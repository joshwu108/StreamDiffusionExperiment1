#!/usr/bin/env python3
"""
Step 5 of plan.md: is the TAEHV encoder a usable stand-in for the Wan encoder on rank 0?

On a real clip (examples/original.mp4, 480x832, bf16), called exactly the way V2 calls stream_encode
(5 frames on the first call, then 4 per chunk):
  1. Latency per call: Wan stream_encode vs TAEHVFullWrapper.stream_encode (CUDA events).
  2. Latent agreement: TAEHV latents vs Wan raw mu, per latent step (relative L2 error, cosine).
  3. Round trip: decode each latent stream with the Wan decoder (full clip, stateless) and with
     TAEHV, PSNR vs the source and vs the Wan-encoder round trip. The first chunk is reported
     separately because TAEHV's first call is padded (see TAEHVFullWrapper).

Run from the repo root inside the `stream` conda env:
  PYTHONPATH=. /home/joshua/.conda/envs/stream/bin/python examples/check_taehv_encoder.py
"""
import argparse
import csv
import math
import os
import sys
import warnings

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings("ignore")

from causvid.models.wan.wan_wrapper import WanVAEWrapper, TAEHVFullWrapper  # noqa: E402
from streamv2v.inference import load_mp4_as_tensor  # noqa: E402


def psnr(a, b):
    mse = ((a.float() - b.float()) / 2.0).pow(2).mean().item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def v2_encode(fn, src):
    """Call fn like V2: frames [0,5) then [5,9), [9,13), ... Returns [1, 16, T_l, h, w]."""
    outs = [fn(src[:, :, 0:5])]
    e = 5
    while e + 4 <= src.shape[2]:
        outs.append(fn(src[:, :, e:e + 4]))
        e += 4
    return torch.cat(outs, 2)


def timed(fn, warm=3, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    ms.sort()
    return ms[len(ms) // 2]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(REPO_ROOT, "examples", "original.mp4"))
    ap.add_argument("--num_latents", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "check_taehv_encoder.csv"))
    args = ap.parse_args()
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    n_frames = 1 + 4 * (args.num_latents - 1)
    src = load_mp4_as_tensor(args.video, max_frames=n_frames, resize_hw=(480, 832)).unsqueeze(0).to(device, dtype)
    src_ntchw = src.permute(0, 2, 1, 3, 4).float()
    print(f"source clip: {tuple(src.shape)}")

    vae = TAEHVFullWrapper().to(device=device, dtype=dtype)     # Wan encoder+decoder, TAEHV encoder+decoder
    mean, std = vae.mean.to(device, dtype).view(1, -1, 1, 1, 1), vae.std.to(device, dtype).view(1, -1, 1, 1, 1)

    # --- 1. latency of one steady-state 4-frame call
    chunk = src[:, :, 5:9]
    vae.model.first_encode = True
    WanVAEWrapper.stream_encode(vae, src[:, :, 0:5])
    wan_ms = timed(lambda: WanVAEWrapper.stream_encode(vae, chunk))
    vae.reset()
    vae.stream_encode(src[:, :, 0:5])
    tae_ms = timed(lambda: vae.stream_encode(chunk))
    print(f"\nencode latency, 4-frame chunk (median of 20): Wan {wan_ms:.1f} ms   TAEHV {tae_ms:.1f} ms")

    # --- 2. latents, V2 call pattern, raw mu space
    vae.model.first_encode = True
    z_wan = v2_encode(lambda v: WanVAEWrapper.stream_encode(vae, v), src)          # raw mu
    vae.reset()
    z_tae = v2_encode(vae.stream_encode, src)                                       # raw (default)
    # TAEHV emits latent m once frames [4m, 4m+3] have arrived: 1 latent on the first call, then 1 per
    # chunk, so its stream is one latent shorter than Wan's; latent m of both should agree.
    print(f"latent stream: Wan {tuple(z_wan.shape)}  TAEHV {tuple(z_tae.shape)}  (V2 pattern: first call, then 1 per chunk)")
    n_l = min(z_wan.shape[2], z_tae.shape[2])
    z_wan, z_tae = z_wan[:, :, :n_l], z_tae[:, :, :n_l]
    zw, zt = z_wan.float(), z_tae.float()
    per_t_rel = [((zt[:, :, i] - zw[:, :, i]).norm() / zw[:, :, i].norm()).item() for i in range(zw.shape[2])]
    per_t_cos = [torch.nn.functional.cosine_similarity(zt[:, :, i].flatten(), zw[:, :, i].flatten(), dim=0).item()
                 for i in range(zw.shape[2])]
    print("per-latent relative L2 error vs Wan mu: " + " ".join(f"{v:.2f}" for v in per_t_rel))
    print("per-latent cosine vs Wan mu:            " + " ".join(f"{v:.2f}" for v in per_t_cos))
    # normalized-space agreement (what the DiT outputs / TAEHV decodes)
    zn_w, zn_t = (zw - mean.float()) / std.float(), (zt - mean.float()) / std.float()
    print(f"normalized space: Wan std {zn_w.std():.3f}  TAEHV std {zn_t.std():.3f}  "
          f"rel err (latents 2+) {((zn_t - zn_w)[:, :, 2:].norm() / zn_w[:, :, 2:].norm()).item():.3f}")

    # --- 3. round trips through the Wan decoder (stateless full clip) and TAEHV (Mode A full clip)
    def wan_dec(z_raw):
        return WanVAEWrapper.decode_to_pixel(vae, ((z_raw - mean) / std).permute(0, 2, 1, 3, 4).contiguous())

    def tae_dec(z_raw):
        return vae.decode_to_pixel(((z_raw - mean) / std).permute(0, 2, 1, 3, 4).contiguous())

    rows = []
    ref = wan_dec(z_wan)
    n_px = ref.shape[1]
    for name, dec_fn, z in [("wan enc -> wan dec", wan_dec, z_wan), ("taehv enc -> wan dec", wan_dec, z_tae),
                            ("wan enc -> taehv dec", tae_dec, z_wan), ("taehv enc -> taehv dec", tae_dec, z_tae)]:
        out = dec_fn(z)
        r = dict(variant=name, frames=out.shape[1], psnr_vs_source=psnr(out, src_ntchw[:, :n_px]),
                 psnr_vs_source_after_first_chunk=psnr(out[:, 5:], src_ntchw[:, 5:n_px]),
                 psnr_vs_wan_roundtrip=psnr(out, ref))
        rows.append(r)
        print(f"  {name:24s} frames={r['frames']:3d}  PSNR vs source {r['psnr_vs_source']:6.2f} dB "
              f"(frames 5+: {r['psnr_vs_source_after_first_chunk']:6.2f})  vs Wan round trip {r['psnr_vs_wan_roundtrip']:6.2f} dB")
    rows.append(dict(variant="latency_ms_wan_encode_4frames", frames=4, psnr_vs_source=wan_ms,
                     psnr_vs_source_after_first_chunk="", psnr_vs_wan_roundtrip=""))
    rows.append(dict(variant="latency_ms_taehv_encode_4frames", frames=4, psnr_vs_source=tae_ms,
                     psnr_vs_source_after_first_chunk="", psnr_vs_wan_roundtrip=""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
