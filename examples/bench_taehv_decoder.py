#!/usr/bin/env python3
"""
Step 1 of plan.md: standalone decoder sanity check.

Measures the cost of ONE decode call the way StreamDiffusionV2 issues it (one latent
timestep per call, steady state) for three decoders, plus one large-T point that shows
the time-axis parallelism V2 cannot use, plus the H3 correctness test.

Decoders / modes
  taehv_stream   StreamingTAEHV.decode  : per-frame work queue, MemBlock memory kept across calls.
                                          This is what a TAEHVDecoderWrapper in V2 would call.
  taehv_modeA    decode_video(parallel=True)  : whole call at once, memory reset each call,
                                          first t_upscale-1 = 3 raw frames trimmed. The misuse.
  taehv_modeB    decode_video(parallel=False) : sequential, memory reset each call (no state carry).
  wan_stream     Wan 2.1 VAE stream_decode with feat_cache : what V2 uses today.

Usage (from repo root, inside the V2 venv):
  venv/bin/python examples/bench_taehv_decoder.py                 # default sweep + correctness
  venv/bin/python examples/bench_taehv_decoder.py --profile       # add kernel counts / GPU busy %
  venv/bin/python examples/bench_taehv_decoder.py --skip_wan      # TAEHV only

No dependency on the causvid package import chain: vae.py is loaded by file path.
"""
import argparse
import csv
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Wan 2.1 latent normalization constants (copied from causvid/models/wan/wan_wrapper.py:WanVAEWrapper)
WAN_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
WAN_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


# ----------------------------------------------------------------------------- loading

def load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_taehv(taehv_dir: str, ckpt: str, device, dtype):
    taehv_mod = load_module_from_path("taehv", os.path.join(taehv_dir, "taehv.py"))
    ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(taehv_dir, ckpt)
    model = taehv_mod.TAEHV(checkpoint_path=ckpt_path).to(device=device, dtype=dtype).eval()
    return taehv_mod, model


class WanDecoder:
    """Minimal stand-in for WanVAEWrapper.stream_decode_to_pixel, built from vae.py by path."""

    def __init__(self, device, dtype, model_type="T2V-1.3B"):
        vae_mod = load_module_from_path(
            "wan_vae", os.path.join(REPO_ROOT, "causvid", "models", "wan", "wan_base", "modules", "vae.py"))
        self.model = vae_mod._video_vae(
            pretrained_path=os.path.join(REPO_ROOT, "wan_models", f"Wan2.1-{model_type}", "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False).to(device=device, dtype=dtype)
        self.mean = torch.tensor(WAN_MEAN, device=device, dtype=dtype)
        self.inv_std = 1.0 / torch.tensor(WAN_STD, device=device, dtype=dtype)
        self.reset()

    def reset(self):
        self.model.clear_cache_decode()
        self.model.first_decode = True  # the first call must carry >= 2 latents (see vae.py:stream_decode)

    @torch.no_grad()
    def decode(self, latent_ntchw: torch.Tensor) -> torch.Tensor:
        # mirrors WanVAEWrapper.stream_decode_to_pixel: [B,T,C,H,W] -> [B,T_px,3,H,W] in [-1,1]
        zs = latent_ntchw.permute(0, 2, 1, 3, 4)
        out = self.model.stream_decode(zs, [self.mean, self.inv_std]).float().clamp_(-1, 1)
        return out.permute(0, 2, 1, 3, 4)


# ----------------------------------------------------------------------------- timing

@dataclass
class Result:
    decoder: str
    mode: str
    T: int
    frames_per_call: int
    ms_per_call_mean: float
    ms_per_call_median: float
    ms_per_call_p90: float
    ms_per_frame: float
    peak_mem_MiB: float
    cuda_kernels: Optional[int] = None
    gpu_busy_pct: Optional[float] = None
    note: str = ""


def time_calls(fn: Callable[[], torch.Tensor], warmup: int, iters: int):
    """CUDA-event timing of fn(). Returns (ms list, frames produced by the last call)."""
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # exclude cuDNN autotune workspaces from the peak
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    frames = out.shape[1] if out is not None else 0
    return times, frames


def profile_one_call(fn: Callable[[], torch.Tensor]):
    """One profiled call: number of CUDA kernels launched and GPU busy fraction of wall time."""
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6
    kernels = 0
    busy_us = 0.0
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            kernels += 1
            busy_us += ev.self_device_time_total if hasattr(ev, "self_device_time_total") else ev.cuda_time
    return kernels, 100.0 * busy_us / max(wall_us, 1e-6)


def summarize(name, mode, T, times, frames, peak, kernels=None, busy=None, note=""):
    s = sorted(times)
    return Result(
        decoder=name, mode=mode, T=T, frames_per_call=frames,
        ms_per_call_mean=sum(times) / len(times),
        ms_per_call_median=s[len(s) // 2],
        ms_per_call_p90=s[int(0.9 * (len(s) - 1))],
        ms_per_frame=(sum(times) / len(times)) / max(frames, 1),
        peak_mem_MiB=peak / 2**20,
        cuda_kernels=kernels, gpu_busy_pct=busy, note=note,
    )


# ----------------------------------------------------------------------------- decoder call shapes

def make_latents(T, C, h, w, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, T, C, h, w, generator=g).to(device=device, dtype=dtype)


def stream_call_factory(streaming, latents):
    """One V2-style call: feed T latents, drain every frame they produce. Returns [1, F, 3, H, W]."""
    def fn():
        frames = []
        f = streaming.decode(latents)
        while f is not None:
            frames.append(f)
            f = streaming.decode()
        return torch.cat(frames, 1) if frames else None
    return fn


def run_sweep(args, taehv_mod, taehv, wan: Optional[WanDecoder], device, dtype) -> List[Result]:
    results = []
    C = taehv.latent_channels
    h, w = args.height // 8, args.width // 8
    T_points = [1, args.T_long]

    def measure(name, mode, T, fn, prime: Optional[Callable] = None, note=""):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if prime is not None:
            prime()
        times, frames = time_calls(fn, args.warmup, args.iters)
        peak = torch.cuda.max_memory_allocated(device)
        kernels = busy = None
        if args.profile:
            kernels, busy = profile_one_call(fn)
        r = summarize(name, mode, T, times, frames, peak, kernels, busy, note)
        results.append(r)
        print(f"  {name:13s} {mode:22s} T={T:2d}  {frames:3d} frames/call  "
              f"{r.ms_per_call_mean:8.2f} ms/call  {r.ms_per_frame:7.3f} ms/frame  peak {r.peak_mem_MiB:7.0f} MiB"
              + (f"  kernels={kernels} busy={busy:.0f}%" if kernels is not None else ""))
        return r

    for T in T_points:
        print(f"\n== T = {T} latent(s) per call  (latent [1,{T},{C},{h},{w}], {4*T} pixel frames nominal) ==")
        z = make_latents(T, C, h, w, device, dtype)

        # --- TAEHV streaming (what V2 would call). Prime with one latent so the 3 startup frames are gone.
        streaming = taehv_mod.StreamingTAEHV(taehv)
        z_prime = make_latents(1, C, h, w, device, dtype, seed=99)
        with torch.no_grad():
            measure("taehv", "stream (state kept)", T, stream_call_factory(streaming, z),
                    prime=lambda: stream_call_factory(streaming, z_prime)(),
                    note="steady state after 1 priming latent; MemBlock memory carried across calls")

        # --- TAEHV Mode A per call (the misuse): memory reset every call, 3 raw frames trimmed per call.
        with torch.no_grad():
            measure("taehv", "modeA parallel=True", T,
                    lambda: taehv.decode_video(z, parallel=True, show_progress_bar=False),
                    note="memory reset each call; first 3 raw frames trimmed each call")

        # --- TAEHV Mode B per call: sequential, memory reset every call.
        with torch.no_grad():
            measure("taehv", "modeB parallel=False", T,
                    lambda: taehv.decode_video(z, parallel=False, show_progress_bar=False),
                    note="memory reset each call; sequential work queue")

        # --- Wan 2.1 full VAE streaming decode (today's V2 decoder).
        if wan is not None:
            z_wan_prime = make_latents(2, C, h, w, device, dtype, seed=98)
            def prime_wan():
                wan.reset()
                wan.decode(z_wan_prime)  # first call must have >= 2 latents (vae.py first_decode path)
            measure("wan", "stream feat_cache", T, lambda: wan.decode(z), prime=prime_wan,
                    note="steady state after 2-latent priming call; feat_cache carried across calls")
    return results


# ----------------------------------------------------------------------------- correctness (H3)

@torch.no_grad()
def correctness_check(args, taehv_mod, taehv, device, dtype, T=16):
    """
    H3: streaming decode of a T-latent clip must equal full-clip Mode A (same weights, same math),
    while chunked Mode A (one T=1 call per latent) must differ from it because every call starts
    with zero MemBlock memory and trims 3 of its 4 raw frames.
    """
    C = taehv.latent_channels
    h, w = args.height // 8, args.width // 8
    z = make_latents(T, C, h, w, device, dtype, seed=7)

    ref = taehv.decode_video(z, parallel=True, show_progress_bar=False).float()        # [1, 4T-3, 3, H, W]

    streaming = taehv_mod.StreamingTAEHV(taehv)
    stream_out = stream_call_factory(streaming, z)().float()                             # [1, 4T-3, 3, H, W]

    chunked = [taehv.decode_video(z[:, i:i + 1], parallel=True, show_progress_bar=False).float()
               for i in range(T)]                                                        # T x [1, 1, 3, H, W]
    chunked = torch.cat(chunked, 1)                                                      # [1, T, 3, H, W]
    trimmed_per_call = taehv.frames_to_trim
    wasted = T * trimmed_per_call

    def maxdiff(a, b):
        return (a - b).abs().max().item()

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    print(f"\n== Correctness (H3), T={T} latents, dtype={dtype} ==")
    print(f"  full-clip Mode A frames : {ref.shape[1]}   (4*T - {trimmed_per_call} startup frames trimmed once)")
    print(f"  streaming frames        : {stream_out.shape[1]}")
    print(f"  chunked Mode A frames   : {chunked.shape[1]}   ({wasted} raw frames trimmed and discarded = {trimmed_per_call} per call)")
    same_shape = stream_out.shape == ref.shape
    d_stream = maxdiff(stream_out, ref) if same_shape else float("nan")
    print(f"  streaming vs full-clip  : max|diff| = {d_stream:.3e}  (tol {tol:.0e})  -> {'PASS' if same_shape and d_stream <= tol else 'FAIL'}")

    # chunk i's single surviving frame is raw frame 4i+3 = ref frame 4i. Chunk 0 has zero memory in both
    # paths, so it should match; chunks >= 1 should differ because the memory was reset.
    per_chunk = [maxdiff(chunked[:, i], ref[:, 4 * i]) for i in range(T)]
    print(f"  chunked Mode A vs full-clip, per chunk max|diff|:")
    print("   " + " ".join(f"{d:.2e}" for d in per_chunk))
    chunk0_ok = per_chunk[0] <= tol
    later_differ = all(d > tol for d in per_chunk[1:])
    print(f"  chunk 0 matches (zero memory either way) : {'PASS' if chunk0_ok else 'FAIL'}")
    print(f"  chunks 1..{T-1} differ (memory reset)       : {'PASS' if later_differ else 'FAIL'}")
    return {
        "T": T, "dtype": str(dtype), "ref_frames": ref.shape[1], "stream_frames": stream_out.shape[1],
        "chunked_frames": chunked.shape[1], "wasted_raw_frames": wasted,
        "stream_vs_ref_maxdiff": d_stream, "chunk0_maxdiff": per_chunk[0],
        "chunks_ge1_min_maxdiff": min(per_chunk[1:]), "tol": tol,
        "pass": bool(same_shape and d_stream <= tol and chunk0_ok and later_differ),
    }


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taehv_dir", default="/home/joshua/taehv")
    ap.add_argument("--taehv_ckpt", default="taew2_1.pth", help="file in --taehv_dir, or absolute path")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--T_long", type=int, default=16, help="the one large-T point")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_wan", action="store_true")
    ap.add_argument("--skip_correctness", action="store_true")
    ap.add_argument("--profile", action="store_true", help="add CUDA kernel count and GPU busy %% per mode")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "bench_taehv_decoder.csv"))
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.backends.cudnn.benchmark = True

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(device)}  dtype {dtype}")
    taehv_mod, taehv = load_taehv(args.taehv_dir, args.taehv_ckpt, device, dtype)
    n_dec = sum(p.numel() for p in taehv.decoder.parameters())
    print(f"TAEHV {args.taehv_ckpt}: decoder {n_dec/1e6:.2f}M params, {len(taehv.decoder)} layers, "
          f"latent_channels={taehv.latent_channels}, t_upscale={taehv.t_upscale}, frames_to_trim={taehv.frames_to_trim}")
    wan = None if args.skip_wan else WanDecoder(device, dtype)
    if wan is not None:
        n_wan = sum(p.numel() for p in wan.model.decoder.parameters())
        print(f"Wan 2.1 VAE decoder: {n_wan/1e6:.1f}M params")

    results = run_sweep(args, taehv_mod, taehv, wan, device, dtype)

    corr = None
    if not args.skip_correctness:
        corr = correctness_check(args, taehv_mod, taehv, device, dtype)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        wr.writeheader()
        for r in results:
            wr.writerow(asdict(r))
    print(f"\nwrote {args.out}")
    if corr is not None:
        corr_path = os.path.splitext(args.out)[0] + "_correctness.csv"
        with open(corr_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(corr.keys()))
            wr.writeheader(); wr.writerow(corr)
        print(f"wrote {corr_path}")

    # --- tripwire summary -------------------------------------------------------------
    t1 = {(r.decoder, r.mode): r for r in results if r.T == 1}
    print("\n== T=1 summary (one V2 chunk per call) ==")
    print(f"  {'decoder / mode':36s} {'ms/call':>9s} {'frames':>7s} {'ms/frame':>9s}")
    for (d, m), r in t1.items():
        print(f"  {d + ' ' + m:36s} {r.ms_per_call_mean:9.2f} {r.frames_per_call:7d} {r.ms_per_frame:9.3f}")
    s = t1.get(("taehv", "stream (state kept)"))
    if s is not None:
        print(f"\nTRIPWIRE value: TAEHV streaming T=1 = {s.ms_per_call_mean:.2f} ms per chunk call. "
              f"Step 3 compares this with the final rank's DiT share; if it is not clearly smaller, "
              f"reinstate the launch-overhead hypotheses.")


if __name__ == "__main__":
    main()
