# Investigation plan: why the TAEHV decoder does not benefit from StreamDiffusionV2's pipeline parallelism

Companion diagram: `streamdiffusion_pipeline.drawio` (open in draw.io / diagrams.net; 5 pages: 0 concepts, 1 v1 reference, 2 V2 pipeline, 3 TAEHV decoder & hypotheses, 4 work breakdown).

| Page | What it shows | Where this plan attaches |
|---|---|---|
| 1. StreamDiffusion v1 (per-frame loop) | The original SD 1.5 pipeline: preprocess → similarity filter → VAE encode + SDEdit noise → Stream Batch → one U-Net call → split → VAE decode. | Nowhere. TAEHV's 16-channel video latents are incompatible with SD 1.5's 4-channel 2D latents; v1 is the reference for *where* the VAE sits on the critical path. |
| 2. StreamDiffusionV2 multi-GPU pipeline | Rank 0 (Wan VAE encode + first DiT blocks) → middle ranks (DiT blocks) → final rank (last DiT blocks → **decode** → host copy), ring send-back per denoising step, block scheduler. | Stage 12 (red) is the decode under study. Steps 2–4 (green) swap the decoder, instrument stages 9–13, and validate fixes. |
| 3. TAEHV decoder & investigation | The tiny causal decoder (Clamp → 3 stages of MemBlock/Upsample/TGrow → head), the MemBlock recurrence, its three execution modes, hypotheses H1–H3. | Step 1 sanity check runs here, standalone. |
| 4. Implementation breakdown | Steps 0–5 with sub-tasks, done-when checks, and dependencies. | Whole plan. |

---

## 1. Context

**Research question (confirmed, narrowed 2026-09-23):** why the TAEHV decoder, which is efficient in
isolation, does not make the multi-GPU, rank-pipelined StreamDiffusionV2 application faster. The
deliverable is an evidence-backed explanation (measurements + attribution), not a production feature.

**Working assumption:** a single TAEHV decode call is cheap. We do *not* investigate TAEHV's own kernel
efficiency beyond one sanity measurement (Step 1). If that measurement shows a T=1 call is expensive on
its own, the assumption is wrong and the launch-overhead hypotheses from the earlier draft come back.

**"Parallelism" means three different things here; this plan is about the first:**

1. **Pipeline parallelism across GPUs.** The 30 DiT blocks are split across ranks like an assembly line;
   the decoder is not split at all. It lands whole on the final rank, after that rank's DiT blocks, and
   blocks the loop until pixels reach the CPU. Throughput = 1 / slowest station, so adding GPUs shortens
   every station except the one that owns decode. **This is the parallelism under study.**
2. **Time-axis parallelism inside TAEHV** (`decode_video(parallel=True)`). Processes all latent timesteps
   of a clip in one pass; it is where TAEHV's offline speed comes from. V2 hands the decoder one latent per
   chunk, so there is no time axis to parallelize, and calling the parallel path per chunk resets the
   MemBlock memory and discards 3 startup frames every call. This is a *misuse* of TAEHV, kept in the plan
   only as a demonstration (variant `taehv_parallel`).
3. **Batch parallelism across streams** (N videos per call). Supported by TAEHV, unused by V2 (one stream).
   Out of scope except as a possible fix.

The common root: MemBlocks are recurrent (layer output at t needs the same layer's output at t−1). A
recurrence can be pipelined across layers or batched across streams but cannot be parallelized across
time without breaking correctness, and V2's chunk size of one latent removes the time axis anyway.

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
- `--schedule_block` rebalances DiT blocks across ranks via `_handle_block_scheduling` →
  `compute_balanced_split(total_blocks, t_total_list, t_dit_list, ...)`. **Verified in code:** the final
  rank's `t_total` *does* include decode + host copy (`inference_pipe.py:415`), and rank 0's includes
  encode (`:291`); middle ranks report DiT only. The rebalance fires once, and every timing is the
  *minimum* observed value, not the mean.
- Baseline decoder: Wan `stream_decode` with `feat_cache` (`wan_base/modules/vae.py`, `CACHE_T = 2`), bf16.

---

## 3. Hypotheses

| # | Hypothesis | Test |
|---|---|---|
| H1 (primary) | Decode is serialized on the final rank: DiT → decode → `.cpu()` run back-to-back on the compute stream, so the final rank is the longest pipeline station and FPS does not improve from 1 → 2 ranks. | Per-rank CUDA-event timeline for 1 and 2 ranks with `--vae wan` and `--vae taehv`; pipeline period = max stage time; decode share of the final-rank stage. |
| H2 | The one-shot, minimum-based block rebalance under-weights decode, so the final rank keeps too many DiT blocks even though the scheduler nominally sees `t_vae`. | Log `block_num` and each rank's contributed `t_total` before/after rebalance; compare with a mean-based or repeated rebalance. |
| H3 (demonstration) | Using TAEHV's time-parallel path per chunk (`decode_video(parallel=True)` with T=1) is both incorrect (memory reset, 3 startup frames trimmed per call) and wasteful (re-warm-up each call). | Round-trip equality test in Step 1; `--vae taehv_parallel` in V2 for frame count and visual artifacts. |

Retired from the earlier draft (kept here so the reasoning is visible): "TAEHV is launch-bound at T=1"
and "small-tensor overheads dominate". Both are about TAEHV's own efficiency and are excluded by the
working assumption. Step 1's sanity check is the tripwire that brings them back if needed.

---

## 4. Work plan

### Step 1 — Standalone decoder sanity check (diagram page 3; no V2 dependency; ~½ day)
New script `/home/joshua/StreamDiffusionV2/examples/bench_taehv_decoder.py` (V2 venv, imports
`/home/joshua/taehv/taehv.py`). Deliberately small: its job is to confirm the working assumption and
to produce the numbers the pipeline analysis needs, not to profile TAEHV.

- Load `taew2_1` in bf16 on one A100. Latent shape for 480×832: `[1, T, 16, 60, 104]`.
- Measure ms per output pixel frame (CUDA events, 10 warm-up + 50 timed) for:
  `StreamingTAEHV.decode` at T=1 (what V2 will call), Mode A at T=1 (the `taehv_parallel` misuse), and
  Wan 2.1 `stream_decode` at T=1 (the baseline V2 uses today). One extra point at T=16 for Mode A and
  streaming, to show the time-axis parallelism V2 cannot use.
- Tripwire: if streaming T=1 decode is not clearly cheaper than the final rank's DiT share (from Step 3),
  the "TAEHV is efficient" assumption fails and the retired launch-overhead hypotheses are reinstated.
- Correctness check (H3): 16-latent clip; streaming decode must equal full-clip Mode A within bf16
  tolerance, and differ from chunked Mode A with T=1 calls. Count trimmed frames.
- Output: small CSV + printed table.

**Step 1 result (2026-09-23, A100 80GB PCIe, bf16, 480×832, 10 warm-up + 50 timed, `--profile`):**

| decoder / mode | T | frames/call | ms/call | ms/frame | CUDA kernels | GPU busy |
|---|---|---|---|---|---|---|
| TAEHV streaming (state kept) | 1 | 4 | **8.7** | 2.16 | 321 | 70 % |
| TAEHV Mode A per call (misuse) | 1 | 1 | 9.0 | 8.98 | 233 | 72 % |
| TAEHV Mode B per call | 1 | 1 | 8.8 | 8.81 | 327 | 73 % |
| Wan 2.1 VAE stream_decode (V2 today) | 1 | 4 | **164.8** | 41.2 | 616 | 97 % |
| TAEHV streaming | 16 | 64 | 136.6 | 2.14 | 5135 | 93 % |
| TAEHV Mode A | 16 | 61 | 131.0 | 2.15 | 232 | 97 % |
| Wan 2.1 VAE stream_decode | 16 | 64 | 2659.5 | 41.6 | 9752 | 99 % |

- **Working assumption holds.** A steady-state TAEHV chunk decode is 8.7 ms, 19× cheaper than the Wan
  decoder V2 uses today, and ms/frame is the same at T=1 and T=16. There is no launch-overhead penalty
  at T=1 to speak of (30 % idle, not 90 %). The retired hypotheses stay retired. Tripwire value = 8.7 ms.
- **Mode A / B per chunk cost the same 9 ms but return 1 usable frame instead of 4** (3 raw frames
  trimmed per call), i.e. 4× the per-frame cost, and the frames are wrong (below).
- **H3 correctness (16-latent clip):** streaming output == full-clip Mode A output exactly (max |diff| 0).
  Chunked Mode A returns 16 frames instead of 61 and discards 48 raw frames; chunk 0 matches the
  reference (zero memory in both paths) and chunks 1–15 differ with max |diff| ≈ 1.0, i.e. garbage.
- Side fact for Step 3: at 16 FPS input a chunk is 250 ms of video; the Wan decoder alone takes 165 ms of
  that budget on the final rank, TAEHV 9 ms. Whatever V2's final-rank stage looks like today, decode is a
  large part of it with Wan and should be small with TAEHV.
- Artifacts: `outputs/bench_taehv_decoder.csv`, `outputs/bench_taehv_decoder_correctness.csv`.
  Venv additions needed: `tqdm`, `einops`, `safetensors` (versions from requirements.txt).

### Step 2 — Swappable TAEHV decoder inside V2 (diagram page 2, stage 12)
- Add `TAEHVDecoderWrapper` in `causvid/models/wan/wan_wrapper.py` exposing the same
  `stream_decode_to_pixel(latent)` contract as `WanVAEWrapper`: input `[B, T, 16, H/8, W/8]` in Wan's
  normalized latent space; output `[B, T_px, 3, H, W]` in [-1, 1]. Internally: permute to NTCHW,
  `StreamingTAEHV.decode` per latent step, collect `t_upscale` frames, map [0,1] → [-1,1]. Carry state
  across chunks; expose `reset()`. Verify the latent normalization against a real Wan latent (PSNR of
  TAEHV decode vs Wan decode).
- Wire a `--vae {wan, taehv, taehv_parallel}` flag in `streamv2v/inference.py` and
  `streamv2v/inference_pipe.py`. `taehv_parallel` = the naive per-chunk Mode A decode, to reproduce
  the incorrect time-parallel misuse for H3.
- Encoder stays Wan on rank 0; the question is about the decoder.

### Step 3 — Instrument the pipeline (diagram page 2, stages 9–13)
- In the final-rank loop, add CUDA-event timings for receive-wait, DiT, decode, and host copy; rank 0
  and middle loops already have DiT + comm timings. Dump per-iteration JSON per rank.
- Runs (480×832, `--step 2`, `examples/original.mp4`): single GPU (`inference.py`) and 2-rank pipeline
  (`inference_pipe.py`), each with `--vae wan`, `--vae taehv`, `--vae taehv_parallel`, with and without
  `--schedule_block`.
- Derive: pipeline period (max stage time), decode share of the final-rank stage, achieved FPS, and
  whether FPS improves from 1 → 2 ranks with each decoder (H1).
- Log `block_num` before/after rebalance and each rank's contributed `t_total` / `t_dit` (H2).

This is the main step. H1 is answered here.

### Step 4 — Attribute and validate fixes (diagram page 2, green notes)
Only after Step 3 points at a cause, test the matching fix to confirm the attribution:

- H1 (primary) → make decode a real pipeline stage instead of a tail on the last one:
  (a) run decode on a side CUDA stream so it overlaps the next receive + DiT;
  (b) a dedicated decode rank (world_size + 1) that receives x0 from the final DiT rank;
  (c) pinned, non-blocking host copy instead of `.cpu().float().numpy()` in the loop.
  Try (a) and (c) first; (b) only if (a) cannot hide decode fully.
- H2 → mean-based (not min-based) timings and periodic re-scheduling in `_handle_block_scheduling`;
  compare `block_num` and final-rank stage time.
- Optional, only if Step 1's tripwire fires → batch 2–4 latents per decode call (Mode C-style batching
  across time is not allowed; batching across streams is) or CUDA-graph the single-step decoder.

Report each fix as "final-rank stage ms before/after, pipeline FPS before/after, 1-rank vs 2-rank".

## 5. Files

| Action | Path |
|---|---|
| New | `StreamDiffusionV2/examples/bench_taehv_decoder.py` |
| Edit | `StreamDiffusionV2/causvid/models/wan/wan_wrapper.py` (add `TAEHVDecoderWrapper`) |
| Edit | `StreamDiffusionV2/streamv2v/inference.py`, `streamv2v/inference_pipe.py` (`--vae` flag, event timings) |
| Reuse | `/home/joshua/taehv/taehv.py` (`TAEHV`, `StreamingTAEHV`); `WanVAEWrapper.stream_decode_to_pixel` as baseline |
| Untouched | `/home/joshua/StreamDiffusion.git` (v1) |

## 6. Verification

- Step 1 ✅ done: streaming == full-clip Mode A (exact); chunked Mode A differs; T=1 streaming decode = 8.7 ms
  (tripwire value), Wan = 165 ms.
- Step 2: `--vae taehv` produces a video with the same frame count as `--vae wan`; PSNR vs Wan decode reported.
- Step 3: per-rank timing JSON exists for every configuration; a summary table reproduces the README-style
  FPS for `--vae wan` as a sanity anchor; the 1-rank vs 2-rank FPS ratio is reported per decoder.
- Final write-up answers the question with numbers: decode share of the final-rank stage, whether FPS
  scales with ranks for each decoder, whether the scheduler's block split changes when decode cost is
  weighted correctly, and the measured effect of each validated fix.
