#!/usr/bin/env python3
"""
Write examples/original_x3.mp4 = examples/original.mp4 repeated N times (plan.md Step 3).

The 81-frame clip gives only 20 chunks, i.e. ~12 steady-state pipeline iterations after warm-up.
Looping it 3x gives 243 frames = 60 chunks. There is no ffmpeg on this machine, so the clip is
re-encoded with torchvision (PyAV backend). num_chunks in the inference scripts is derived from the
frame count, the KV cache is a rolling window, so nothing else in the pipeline changes.

Usage (conda env `stream`):
    python examples/make_looped_clip.py [--repeats 3] [--src examples/original.mp4] [--dst examples/original_x3.mp4]
"""
import argparse

import torch
import torchvision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="examples/original.mp4")
    ap.add_argument("--dst", default="examples/original_x3.mp4")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    video, _, info = torchvision.io.read_video(args.src, output_format="THWC", pts_unit="sec")
    fps = float(info["video_fps"])
    print(f"{args.src}: {tuple(video.shape)} @ {fps:g} fps")
    looped = torch.cat([video] * args.repeats, dim=0)
    torchvision.io.write_video(args.dst, looped, fps=round(fps), video_codec="libx264", options={"crf": "12"})
    check, _, _ = torchvision.io.read_video(args.dst, output_format="THWC", pts_unit="sec")
    print(f"{args.dst}: {tuple(check.shape)} -> {(check.shape[0] - 1) // 4} chunks of 4 frames")


if __name__ == "__main__":
    main()
