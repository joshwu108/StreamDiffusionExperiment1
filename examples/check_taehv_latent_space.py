#!/usr/bin/env python3
"""
Step 2 of plan.md: verify the TAEHV decoder wrapper against a real Wan 2.1 latent.

Answers three questions with one real clip (examples/original.mp4, 480x832, bf16):
  1. Latent convention. TAEHV's README says it consumes "exactly what the diffusion model uses"
     (no mean/std). The DiT in V2 works in Wan's normalized space z = (mu - mean) / std, so the
     wrapper feeds the DiT output as-is (convention A). Convention B (un-normalize first,
     z * std + mean) is decoded too; the PSNR against the Wan decoder settles which is right.
  2. Wrapper contract. TAEHVDecoderWrapper.stream_decode_to_pixel, called the way V2 calls it
     (2 latents on the first call, then 1 per chunk), must return the same frame count as
     WanVAEWrapper.stream_decode_to_pixel and match TAEHV's full-clip decode.
  3. H3 misuse. --vae taehv_parallel (decode_video(parallel=True) per chunk) frame count and quality.

Run from the repo root inside the `stream` conda env (needs PyAV for video loading):
  PYTHONPATH=. /home/joshua/.conda/envs/stream/bin/python examples/check_taehv_latent_space.py
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

from causvid.models.wan.wan_wrapper import WanVAEWrapper, TAEHVDecoderWrapper  # noqa: E402
from streamv2v.inference import load_mp4_as_tensor  # noqa: E402


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB between two [-1, 1] tensors, computed on the [0, 1] scale."""
    mse = ((a.float() - b.float()) / 2.0).pow(2).mean().item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def v2_style_stream(decode_fn, z: torch.Tensor) -> torch.Tensor:
    """Call decode_fn the way V2 does: first call with 2 latents, then one latent per chunk."""
    outs = [decode_fn(z[:, :2])]
    for i in range(2, z.shape[1]):
        outs.append(decode_fn(z[:, i:i + 1]))
    return torch.cat(outs, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=os.path.join(REPO_ROOT, "examples", "original.mp4"))
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_latents", type=int, default=9, help="latent timesteps; pixel frames = 1 + 4*(n-1)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "check_taehv_latent_space.csv"))
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    n_frames = 1 + 4 * (args.num_latents - 1)

    src = load_mp4_as_tensor(args.video, max_frames=n_frames, resize_hw=(args.height, args.width))
    src = src.unsqueeze(0).to(device=device, dtype=dtype)          # [1, 3, T, H, W] in [-1, 1]
    src_ntchw = src.permute(0, 2, 1, 3, 4).float()                    # [1, T, 3, H, W]
    print(f"source clip: {tuple(src.shape)}  ({n_frames} frames -> {args.num_latents} latents)")

    # One object carries both decoders: it is a WanVAEWrapper (encoder + Wan decoder) plus TAEHV.
    vae = TAEHVDecoderWrapper().to(device=device, dtype=dtype)
    print(f"TAEHV checkpoint: {vae.taehv_ckpt}")
    mean = vae.mean.to(device=device, dtype=dtype)
    std = vae.std.to(device=device, dtype=dtype)

    # Real Wan latent in the DiT's space: z = (mu - mean) / std, [1, T_lat, 16, h, w]
    z = vae.model.encode(src, [mean, 1.0 / std]).permute(0, 2, 1, 3, 4).contiguous()
    print(f"latent: {tuple(z.shape)}  per-channel mean |{z.float().mean(dim=(0, 1, 3, 4)).abs().mean():.3f}|  std {z.float().std():.3f}")

    rows = []

    def record(name, out, ref_full, note=""):
        same_len = out.shape[1] == ref_full.shape[1]
        r = {
            "variant": name, "frames": out.shape[1],
            "psnr_vs_wan_full": psnr(out, ref_full) if same_len else float("nan"),
            "psnr_vs_source": psnr(out, src_ntchw) if same_len else float("nan"),
            "note": note,
        }
        rows.append(r)
        print(f"  {name:34s} frames={r['frames']:3d}  PSNR vs Wan full-clip {r['psnr_vs_wan_full']:6.2f} dB"
              f"  vs source {r['psnr_vs_source']:6.2f} dB  {note}")
        return r

    print("\n== Wan 2.1 VAE decoder (reference) ==")
    wan_full = WanVAEWrapper.decode_to_pixel(vae, z)                  # stateless full-clip decode
    record("wan full-clip decode", wan_full, wan_full)
    vae.model.first_decode = True
    vae.model.clear_cache_decode()
    wan_stream = v2_style_stream(lambda zz: WanVAEWrapper.stream_decode_to_pixel(vae, zz), z)
    record("wan stream_decode (V2 today)", wan_stream, wan_full, "2 latents first call, then 1 per chunk")

    print("\n== TAEHV latent convention (full-clip decode, Mode A once) ==")
    tae_A = vae.decode_to_pixel(z)
    rA = record("taehv A: normalized z as-is", tae_A, wan_full, "wrapper default")
    tae_B = vae.decode_to_pixel(z * std.view(1, 1, -1, 1, 1) + mean.view(1, 1, -1, 1, 1))
    rB = record("taehv B: z*std+mean (un-normalized)", tae_B, wan_full)
    convention = "A" if rA["psnr_vs_wan_full"] > rB["psnr_vs_wan_full"] else "B"
    print(f"  -> convention {convention} matches the Wan decoder "
          f"(margin {abs(rA['psnr_vs_wan_full'] - rB['psnr_vs_wan_full']):.1f} dB)")

    print("\n== TAEHVDecoderWrapper called the way V2 calls it ==")
    vae.reset()
    tae_stream = v2_style_stream(vae.stream_decode_to_pixel, z)
    rS = record("taehv stream (--vae taehv)", tae_stream, wan_full,
                f"calls={vae.num_decode_calls} latents_in={vae.num_latents_in}")
    d_stream_vs_full = maxdiff(tae_stream, tae_A) if tae_stream.shape == tae_A.shape else float("nan")
    print(f"  streaming vs full-clip TAEHV: max|diff| = {d_stream_vs_full:.3e}")
    print(f"  frame count: taehv stream {tae_stream.shape[1]}  vs  wan stream {wan_stream.shape[1]}  "
          f"-> {'MATCH' if tae_stream.shape[1] == wan_stream.shape[1] else 'MISMATCH'}")

    vae.reset()
    vae.decode_mode = "parallel"
    tae_par = v2_style_stream(vae.stream_decode_to_pixel, z)
    vae.decode_mode = "stream"
    # chunk i (i >= 2) keeps only its raw frame 4i+3, which is full-clip frame 4i; the first call
    # (2 latents) keeps raw frames 3..7 = full-clip frames 0..4.
    matched_idx = list(range(5)) + [4 * i for i in range(2, z.shape[1])]
    rP = {
        "variant": "taehv_parallel (--vae taehv_parallel)", "frames": tae_par.shape[1],
        "psnr_vs_wan_full": psnr(tae_par, wan_full[:, matched_idx]),
        "psnr_vs_source": psnr(tae_par, src_ntchw[:, matched_idx]),
        "note": f"matched frames only; {n_frames - tae_par.shape[1]} of {n_frames} frames missing",
    }
    rows.append(rP)
    print(f"  {rP['variant']:34s} frames={rP['frames']:3d}  PSNR vs Wan full-clip {rP['psnr_vs_wan_full']:6.2f} dB"
          f"  vs source {rP['psnr_vs_source']:6.2f} dB  {rP['note']}")
    per_chunk = [psnr(tae_par[:, 5 + k], tae_A[:, 4 * (k + 2)]) for k in range(z.shape[1] - 2)]
    print("  per-chunk PSNR of taehv_parallel vs correct TAEHV frame:  " + " ".join(f"{p:5.1f}" for p in per_chunk))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}")

    ok = (convention == "A" and tae_stream.shape[1] == wan_stream.shape[1]
          and d_stream_vs_full <= 2e-2 and rS["psnr_vs_wan_full"] > rB["psnr_vs_wan_full"])
    print("STEP 2 CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
