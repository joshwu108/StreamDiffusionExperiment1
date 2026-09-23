# Investigation plan: why TAEHV decoders are inefficient when parallelized in StreamDiffusionV2

Companion diagram: `streamdiffusion_pipeline.drawio` (open in draw.io / diagrams.net; 3 pages).

| Page | What it shows | Where this plan attaches |
|---|---|---|
| 1. StreamDiffusion v1 (per-frame loop) | The original SD 1.5 pipeline: preprocess → similarity filter → VAE encode + SDEdit noise → Stream Batch → one U-Net call → split → VAE decode. | Nowhere. TAEHV's 16-channel video latents are incompatible with SD 1.5's 4-channel 2D latents; v1 is the reference for *where* the VAE sits on the critical path. |
| 2. StreamDiffusionV2 multi-GPU pipeline | Rank 0 (Wan VAE encode + first DiT blocks) → middle ranks (DiT blocks) → final rank (last DiT blocks → **decode** → host copy), ring send-back per denoising step, block scheduler. | Stage 12 (red) is the decode under study. Steps 2–4 (green) swap the decoder, instrument stages 9–13, and validate fixes. |
| 3. TAEHV decoder & investigation | The tiny causal decoder (Clamp → 3 stages of MemBlock/Upsample/TGrow → head), the MemBlock recurrence, its three execution modes, hypotheses H1–H5. | Step 1 microbenchmark runs here, standalone. |

---

## 1. Context

**Research question (confirmed):** why TAEHV decoders are inefficient in the multi-GPU, rank-pipelined
StreamDiffusionV2 application. The deliverable is an evidence-backed explanation (measurements +
attribution), not a production feature.

**Environment on this machine**

| Path | What | Notes |
|---|---|---|
| `/home/joshua/StreamDiffusion.git` | Bare mirror of StreamDiffusion v1 (SD 1.5) | Reference only; no code changes here. |
| `/home/joshua/StreamDiffusionV2/` | CausVid over Wan 2.1 1.3B, venv with torch 2.10+cu128 | Uses the **full Wan 2.1 causal VAE** today, not TAEHV (`causvid/models/wan/wan_wrapper.py:116-133`). |
| `/home/joshua/taehv/` | TAEHV code + all checkpoints | `safetensors/taew2_1.safetensors` matches Wan 2.1's latent space. |
| GPUs | 2 × A100 80GB PCIe | The README's 2-rank `torchrun` pipeline is runnable as-is. |

---

## 2. Background facts that shape the hypotheses

### TAEHV (`/home/joshua/taehv/taehv.py`)
Tiny causal 3D autoencoder. `taew2_1`: encoder 1.47M / decoder 9.84M params, 16 latent channels,
4× temporal, 8× spatial. Every `MemBlock` is a recurrent conv cell,
`act(conv(cat[x, past])) + skip(x)`, where `past` is the same layer's previous-timestep output.
Two execution modes over identical weights (diagram page 3):

- **Mode A** `decode_video(parallel=True)`: whole clip, layer by layer, memory = zero-padded previous
  timestep. No state survives between calls; the first `t_upscale − 1 = 3` output frames of every call
  are startup frames and get trimmed. Efficient only when T is large.
- **Mode B** `decode_video(parallel=False)` / `StreamingTAEHV.decode()`: per-frame work queue with
  per-layer memory dicts. Correct for streaming, but one small kernel per layer per frame, so it is
  launch-bound at small T.
- **Mode C** batch across streams (N): the only parallelism that does not fight causality; unused by V2.

I/O conventions: NTCHW, RGB in [0, 1], pre-normalized latents (no mean/std scaling, no scaling factor).

### V2 multi-GPU pipeline (`streamv2v/inference_pipe.py`, diagram page 2)
- Rank 0: Wan VAE `stream_encode` of a 4-frame chunk → noise blend → DiT blocks `[0, b0)` → async send.
- Middle ranks: receive → DiT blocks → send.
- Final rank: receive → DiT blocks `[b_k, 30)` → send x0 back to rank 0 for the chunk's next denoising
  step (ring) → **when the chunk is fully denoised, decode synchronously in the same loop iteration**:

```python
video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])   # one chunk
video = (video * 0.5 + 0.5).clamp(0, 1); video = video[0].permute(0, 2, 3, 1).contiguous()
results[save_results] = video.cpu().float().numpy()                       # host sync
```

- Chunk = `4 * num_frame_per_block` pixel frames (= 4 with `configs/wan_causal_dmd_v2v.yaml`), i.e.
  **one latent timestep per decode call** at 480×832.
- Decode runs on the compute stream after that rank's DiT and before the next receive. No overlap.
- `--schedule_block` rebalances DiT blocks across ranks using `t_dit` **only**; the final rank's decode
  time `t_vae` is invisible to the balancer.
- Baseline decoder: Wan `stream_decode` with `feat_cache` (`wan_base/modules/vae.py`, `CACHE_T = 2`), bf16.

---

## 3. Hypotheses

| # | Hypothesis | Test |
|---|---|---|
| H1 | T=1 per chunk makes TAEHV launch-bound: ~40 layers × 4 output frames of tiny kernels; GPU mostly idle; Python work-queue overhead dominates. | Standalone sweep of ms/frame vs T (1…32), kernel count via `torch.profiler`, compare with FLOP roofline. |
| H2 | Mode A per chunk is both incorrect (memory reset, 3 startup frames trimmed per call) and wasteful (re-warm-up each call). | Round-trip equality test: chunked Mode A vs streaming; count wasted frames. |
| H3 | Decode is serialized with DiT and the host copy on the final rank, so the last rank is the longest pipeline stage and extra GPUs do not raise FPS. | Per-rank CUDA-event timeline for 1 and 2 ranks; stage times and pipeline period. |
| H4 | The block scheduler ignores `t_vae`, so the final rank is over-assigned DiT blocks. | Compare `block_num` distribution with and without decode cost included. |
| H5 | Small-tensor overheads (casts, clone/cat of memories, permute/contiguous, `.cpu()` sync) are a large share of decode wall time at T=1. | Profiler op-level breakdown of one decode call. |

---

## 4. Work plan

### Step 1 — Standalone TAEHV decoder microbenchmark (diagram page 3; no V2 dependency)
New script `/home/joshua/StreamDiffusionV2/examples/bench_taehv_decoder.py` (V2 venv, imports
`/home/joshua/taehv/taehv.py`):

- Load `taew2_1` (and `taew2_2_super` as a high-capacity point) in bf16 on one A100.
- Latent shape for 480×832: `[N, T, 16, 60, 104]`. Sweep T ∈ {1, 2, 4, 8, 16, 32}, N ∈ {1, 2, 4, 8}.
- Modes: Mode A, Mode B (`parallel=False`), `StreamingTAEHV.decode` per latent, plus the Wan 2.1 full
  `stream_decode` (from `WanVAEWrapper`) as the baseline.
- Metrics per (mode, T, N): ms per output pixel frame (CUDA events, 10 warm-up + 50 timed), kernels
  launched and GPU-idle fraction (`torch.profiler` with CUDA activity), peak memory, achieved TFLOPs vs
  decoder FLOPs (`torch.utils.flop_counter`).
- Correctness check (H2): 16-latent clip; streaming decode must equal full-clip Mode A within bf16
  tolerance, and differ from chunked Mode A with T=1 calls.
- Output: CSV + printed table. Optional `torch.compile(mode="reduce-overhead")` column as a
  diagnostic bound on launch overhead.

### Step 2 — Swappable TAEHV decoder inside V2 (diagram page 2, stage 12)
- Add `TAEHVDecoderWrapper` in `causvid/models/wan/wan_wrapper.py` exposing the same
  `stream_decode_to_pixel(latent)` contract as `WanVAEWrapper`: input `[B, T, 16, H/8, W/8]` in Wan's
  normalized latent space; output `[B, T_px, 3, H, W]` in [-1, 1]. Internally: permute to NTCHW,
  `StreamingTAEHV.decode` per latent step, collect `t_upscale` frames, map [0,1] → [-1,1]. Carry state
  across chunks; expose `reset()`. Verify the latent normalization against a real Wan latent (PSNR of
  TAEHV decode vs Wan decode).
- Wire a `--vae {wan, taehv, taehv_parallel}` flag in `streamv2v/inference.py` and
  `streamv2v/inference_pipe.py`. `taehv_parallel` = the naive per-chunk Mode A decode, to reproduce
  the inefficient/incorrect variant for H2.
- Encoder stays Wan on rank 0; the question is about the decoder.

### Step 3 — Instrument the pipeline (diagram page 2, stages 9–13)
- In the final-rank loop, add CUDA-event timings for receive-wait, DiT, decode, and host copy; rank 0
  and middle loops already have DiT + comm timings. Dump per-iteration JSON per rank.
- Runs (480×832, `--step 2`, `examples/original.mp4`): single GPU (`inference.py`) and 2-rank pipeline
  (`inference_pipe.py`), each with `--vae wan`, `--vae taehv`, `--vae taehv_parallel`, with and without
  `--schedule_block`.
- Derive: pipeline period (max stage time), decode share of the final-rank stage, achieved FPS, and
  whether FPS improves from 1 → 2 ranks with each decoder.

### Step 4 — Attribute and validate fixes (diagram page 2, green notes)
Only after Steps 1–3 point at a cause, test the matching fix to confirm the attribution:

- H1/H5 → CUDA-graph or `torch.compile` the single-step decoder, or batch 2–4 latents per decode.
- H3 → run decode on a side CUDA stream (overlapping the next receive) or a dedicated rank; pinned
  async host copy instead of `.cpu().numpy()` in the loop.
- H4 → include `t_vae` in `_handle_block_scheduling`.

Report each fix as "decode ms before/after, pipeline FPS before/after".

---

## 5. Files

| Action | Path |
|---|---|
| New | `StreamDiffusionV2/examples/bench_taehv_decoder.py` |
| Edit | `StreamDiffusionV2/causvid/models/wan/wan_wrapper.py` (add `TAEHVDecoderWrapper`) |
| Edit | `StreamDiffusionV2/streamv2v/inference.py`, `streamv2v/inference_pipe.py` (`--vae` flag, event timings) |
| Reuse | `/home/joshua/taehv/taehv.py` (`TAEHV`, `StreamingTAEHV`); `WanVAEWrapper.stream_decode_to_pixel` as baseline |
| Untouched | `/home/joshua/StreamDiffusion.git` (v1) |

## 6. Verification

- Step 1 correctness test passes (streaming == full-clip Mode A within bf16 tolerance; chunked Mode A differs).
- Step 2: `--vae taehv` produces a video with the same frame count as `--vae wan`; PSNR vs Wan decode reported.
- Step 3: per-rank timing JSON exists for every configuration; a summary table reproduces the README-style
  FPS for `--vae wan` as a sanity anchor.
- Final write-up answers the question with numbers: which hypotheses held, decode share of the final-rank
  stage, and the measured effect of each validated fix.
