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
| `/home/joshua/StreamDiffusionV2/` | CausVid over Wan 2.1 1.3B | Uses the **full Wan 2.1 causal VAE** by default (`causvid/models/wan/wan_wrapper.py:116-133`); `--vae taehv` swaps the decoder since Step 2. |
| Python envs | **Pipeline runs: conda env `stream`** (`/home/joshua/.conda/envs/stream/bin/python`, py3.10, torch 2.6+cu124, flash_attn 2.7.4, diffusers, PyAV). The repo `venv/` (py3.13, torch 2.10+cu128) is bare torch and only runs the Step 1 bench. | causvid is not installed in the conda env: launch with `PYTHONPATH=.` from the repo root. Use `examples/prompt.txt` as `--prompt_file_path` (the README passes the mp4, which `TextDataset` reads as text). |
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

**Step 2 result (2026-09-24, A100 80GB PCIe ×2, bf16, 480×832, `examples/original.mp4` = 81 frames @16 fps = 20 chunks, `--step 2`):**

Implemented:
- `TAEHVDecoderWrapper(WanVAEWrapper)` in `causvid/models/wan/wan_wrapper.py`: same `stream_decode_to_pixel`
  contract, Wan encoder kept for rank 0, `StreamingTAEHV` state carried across chunks, `reset()`, call/frame
  counters. `TAEHVParallelDecoderWrapper` = Mode A per chunk (H3). Registered as `taehv` / `taehv_parallel`
  in `causvid/models/__init__.py`; `CausalStreamInferencePipeline` reads `args.vae` (default `wan`, unchanged
  behaviour). `--vae {wan,taehv,taehv_parallel}` added to `streamv2v/inference.py` and `inference_pipe.py`.
  TAEHV is loaded from `$TAEHV_DIR` (default `/home/joshua/taehv`), checkpoint `$TAEHV_CKPT` (default `taew2_1.pth`).
- `examples/check_taehv_latent_space.py`: latent-convention + wrapper-contract check on a real Wan latent
  (9 latents = 33 frames of the clip). Output `outputs/check_taehv_latent_space.csv`.

Latent convention (PSNR vs the Wan decoder's own full-clip decode of the same latent):

| decoder input | frames | PSNR vs Wan decode | PSNR vs source |
|---|---|---|---|
| Wan `stream_decode` (V2 today) | 33 | exact | 33.6 dB |
| TAEHV, DiT latent as-is (normalized, **wrapper default**) | 33 | **29.1 dB** | 28.8 dB |
| TAEHV, un-normalized `z*std+mean` | 33 | 17.0 dB | 16.9 dB |
| TAEHV streaming wrapper, V2 call pattern | 33 | 29.1 dB (== full-clip, max diff 0) | 28.8 dB |
| `taehv_parallel`, V2 call pattern | **12** | 18.2 dB on the 12 surviving frames, 15.6–16.5 dB per chunk vs correct TAEHV | 18.2 dB |

- TAEHV consumes the DiT's normalized latent directly, as its README states; un-normalizing costs 12 dB.
- The streaming wrapper reproduces TAEHV's full-clip decode bit-exactly and returns the same frame count
  as Wan's streaming path under V2's call pattern (2 latents on the first call, then 1 per chunk).

Pipeline smoke runs (whole-loop "Average FPS" as logged by the scripts, no instrumentation yet; the two
single-GPU runs shared the host concurrently, so treat single-GPU numbers as indicative only):

| decoder | 1 GPU frames / FPS | 2 ranks frames / FPS | 2-rank final-rank loop, ms per chunk |
|---|---|---|---|
| `wan` | 81 / 8.8 | 81 / 14.9 | ~279 |
| `taehv` | 81 / **14.3** | 81 / **24.6** | ~191 |
| `taehv_parallel` (H3) | 24 / (14.4 as logged, but 1 frame per chunk, i.e. ~3.6 real) | 24 / 6.1 | ~191 |

- Verification for Step 2 passes: `--vae taehv` gives the same 81 frames as `--vae wan` on 1 and 2 ranks;
  output frames look like the Wan output (same content, slightly sharper/more saturated).
- H3 demonstration: `taehv_parallel` returns 24 of 81 frames (5 from the first two-latent call, then 1 per
  chunk) with visible blocky / blown-out artifacts, at the same per-call cost as the correct streaming decode.
  `inference.py` reports FPS as `chunk_size / t`, so its 14.4 for this variant is fictitious; `inference_pipe.py`
  uses the real frame count (6.1).
- Preliminary observation for Step 3 (not yet attributed): swapping Wan → TAEHV removes ~156 ms of decode per
  chunk in isolation (165 → 9 ms) but the 2-rank final-rank loop only shortens by ~88 ms (279 → 191 ms), and
  FPS does improve 1 → 2 ranks with both decoders on this short clip. Something other than decode now bounds
  the final rank's loop (receive-wait on rank 0's encode + DiT is the obvious candidate). Step 3's per-stage
  CUDA-event timings are needed before drawing conclusions.
- Artifacts: `outputs/check_taehv_latent_space.csv`, `outputs/step2/{1gpu,2gpu}_{wan,taehv,taehv_parallel}/output_000.mp4`.

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

**Step 3 result (2026-09-24, A100 80GB PCIe ×2, bf16, 480×832, `--step 2`, `examples/original_x3.mp4` = 243 frames / 60 chunks,
each config run twice back-to-back on an idle host; repeats agree within 2.1 %):**

Implemented: `streamv2v/stage_timer.py` (CUDA-event marks on the compute stream + `perf_counter`, one JSONL row per
iteration per rank, off unless `--timing_dir` is given; a timed and an untimed run give the same 15.18 FPS),
marks in all four loops of `inference_pipe.py` / `inference.py`, a scheduler dump in `_handle_block_scheduling`,
`examples/run_step3.sh` (the exact 18 commands are in `outputs/step3/commands.txt`), `examples/summarize_step3.py`
(`outputs/step3/summary.csv`, `stages.csv`). Stage intervals are the CUDA-event gaps on the default stream: they sum
to the iteration, whereas host times mislead for async stages (encode launches in 9 ms, its kernels take 101 ms).

Per-rank stage split, medians over ~50 steady-state iterations (ms, share of that rank's period):

| config | rank | period | recv-wait | encode | DiT | decode | host copy | FPS |
|---|---|---|---|---|---|---|---|---|
| 1 GPU `wan` | 0 | 448 | – | 101 (23 %) | 172 (38 %) | 170 (38 %) | 5 | 8.96 |
| 1 GPU `taehv` | 0 | 284 | – | 101 (36 %) | 172 (60 %) | 9 (3 %) | 2 | 14.12 |
| 2 ranks `wan` | 0 | 263–269 | **73–79 (28 %)** | 100 (38 %) | 89 (34 %) | – | – | |
| 2 ranks `wan` | 1 | 263–269 | 0.8 (0 %) | – | 88 (34 %) | **168 (64 %)** | 5–11 | 15.04 |
| 2 ranks `taehv` | 0 | 191 | 0.8 (0 %) | **101 (53 %)** | 88 (46 %) | – | – | |
| 2 ranks `taehv` | 1 | 191 | **84 (44 %)** | – | 87 (45 %) | 9 (5 %) | 11 | 21.08 |
| 2 ranks `wan` + `--schedule_block` → [0,21],[21,30] | 0 | 230 | 4 (2 %) | 101 (44 %) | 123 (54 %) | – | – | |
| 2 ranks `wan` + `--schedule_block` | 1 | 230 | 0.8 (0 %) | – | 55 (24 %) | 169 (74 %) | 5 | 17.38 |
| 2 ranks `taehv` + `--schedule_block` → [0,7],[7,30] | 0 | 149 | 3 (2 %) | 101 (68 %) | 43 (29 %) | – | – | |
| 2 ranks `taehv` + `--schedule_block` | 1 | 149 | 0.8 (0 %) | – | 133 (90 %) | 9 (6 %) | 5 | 26.52 |
| 2 ranks `taehv_parallel` (H3) | 1 | 191 | 92 (48 %) | – | 87 | 9 (5 %) | 3 | 5.28 (1 frame/iter) |

FPS scaling, mean of the two repeats (frames actually produced / wall time):

| decoder | 1 GPU | 2 ranks | ratio | 2 ranks + schedule | ratio |
|---|---|---|---|---|---|
| `wan` | 8.96 | 15.04 | 1.68× | 17.38 | 1.94× |
| `taehv` | 14.12 | 21.08 | 1.49× | 26.52 | 1.88× |
| `taehv_parallel` | 3.52 | 5.28 | 1.50× | 6.81 | 1.93× |

- **H1 confirmed, and the missing 68 ms is attributed.** With `wan`, decode + copy is 64 % of the final rank's
  iteration, the final rank never waits (recv 0.8 ms), and rank 0 idles 73–79 ms per chunk waiting for it: decode
  is the pipeline period. With `taehv`, decode + copy drops to 10 % and the final rank now idles 84 ms (44 %) per
  chunk in recv-wait; rank 0's own work, **Wan encoder 101 ms + DiT blocks 0–14 88 ms = 191 ms**, is the period.
  The 263 → 191 ms change equals rank 0's former wait (73 ms), i.e. the fast decoder removed the whole final-rank
  tail and exposed the next station. Nothing else in the loop is slow: send/send-wait are < 1 ms, host copy 2–11 ms.
- **The decoder is not "bad at parallelism"; it is not split, and neither is the encoder.** Both VAE halves land
  whole on one rank. Once decode is cheap, the Wan *encoder* (101 ms, unchanged in every run) is the largest
  unsplittable stage and sets a floor on the period no matter how many ranks are added: 4 frames / (101 ms + rank 0's
  blocks) ≈ 40 FPS at best with 2+ ranks, unless the encoder is also swapped or overlapped.
- **H2 refuted as stated.** The one-shot, minimum-based scheduler sees encode and decode correctly
  (`t_total_list` = [0.181, 0.249] s with `wan`, [0.182, 0.101] s with `taehv`) and moves blocks the right way:
  6 blocks *to* rank 0 with `wan` (→ 230 ms period, both ranks within 3 ms of each other), 8 blocks *to* rank 1 with
  `taehv` (→ 149 ms, both ranks within 1 ms). The balanced period equals (encode + all DiT + decode + copy) / 2 to
  within 3 ms in both cases, so there is nothing left for a mean-based or repeated rebalance to recover on 2 ranks.
  The scheduler's only cost is the one-shot stall (0.3–0.8 s) at iteration 9–10.
- **H3 numbers:** `taehv_parallel` costs the same 9.2 ms per call as streaming TAEHV but yields 1 frame per
  iteration (64 of 241 frames), so real throughput is 3.5 / 5.3 / 6.8 FPS; the scripts' logged 14.1 FPS on 1 GPU is
  `chunk_size / t` and fictitious.
- Single-GPU anchor: the DiT alone is 172 ms per chunk (both halves of the 2-rank split sum to 176 ms), so
  `taehv` on 1 GPU is encode-and-DiT-bound at 284 ms; the 1 → 2 rank ratio is 1.49× because the encoder cannot move.
- Artifacts: `outputs/step3/<cfg>/{timing_rank*.jsonl, schedule_rank*.json, log.txt, output_000.mp4, gpu_before.txt}`,
  `outputs/step3/{summary.csv, stages.csv, commands.txt, run.log}`; smoke checks in `outputs/step3_smoke/`.
- Consequence for Step 4: fix (a)/(c) (overlap decode, pinned copy) can at most recover the 9 + 5 ms decode + copy on
  the `taehv` scheduled split; the lever is now the encoder (TAEHV encoder on rank 0, or overlap encode of chunk k+1
  with DiT of chunk k on a side stream), and the block scheduler already does its job.

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

**Step 4 result (2026-09-24, same setup as Step 3, one repeat per variant, baselines = Step 3 runs):**

Implemented in `streamv2v/inference_pipe.py`, both opt-in:
- `--overlap_decode` = fixes (a)+(c): `AsyncDecodeSink` runs `stream_decode_to_pixel` on a side CUDA stream and copies the
  pixels into a ring of pinned host buffers with a non-blocking copy; the loop only harvests finished decodes. The
  end-of-iteration sync becomes compute-stream-only so the side stream keeps running into the next receive + DiT.
- `--overlap_encode`: rank 0 encodes chunk k+1 (`_encode_chunk`: Wan `stream_encode` + SDEdit blend) on a side stream
  right before launching the DiT of chunk k, and consumes it next iteration via an event wait.
- Output check on the 20-chunk clip: 81 frames, mean |pixel diff| to a baseline run 9.9/255 vs 11.8/255 between two
  baseline runs (the SDEdit noise is random per run), no NaN. Commands in `outputs/step4/commands.txt`.

| decoder | variant | 2 ranks FPS (period ms) | + `--schedule_block` FPS (period, split) | rank-1 DiT ms on compute stream |
|---|---|---|---|---|
| `wan` | base (Step 3) | 15.04 (266) | 17.38 (231, [0,21]) | 88 |
| `wan` | overlap decode | 15.84 (252) | **13.43** (298, [0,7]) | **232** |
| `wan` | overlap encode | 15.09 (266) | 16.74 (238, [0,20]) | 88 |
| `wan` | both | 15.65 (255) | 14.10 (284, [0,10]) | 236 |
| `taehv` | base (Step 3) | 21.08 (191) | 26.52 (149, [0,7]) | 132 (rank 0 DiT 43 + encode 101) |
| `taehv` | overlap decode | 21.06 (192) | **27.04** (147, [0,6]) | 139 |
| `taehv` | overlap encode | 22.17 (184) | 25.80 (156, [0,10]) | rank 0 DiT 182 (encode hidden inside) |
| `taehv` | both | 22.12 (181) | 25.92 (154, [0,10]) | rank 0 DiT 153 |

- **Fix (a)+(c) validated as attribution, not as a speed-up.** Taking decode off the compute stream removes it from
  rank 1's stage list (decode+copy 173 → 18 ms on the compute stream) but the GPU is still doing it: rank 1's DiT
  stretches from 88 to 232 ms while the 168 ms Wan decode runs beside it (async decode measured at 174 ms). Net gain
  with `wan` is 263 → 252 ms per chunk (+5 % FPS): both stages are GPU-bound and the A100 has no spare capacity to
  overlap them. With `taehv` the decode is 9 ms and rank 0 is the period, so the fix changes nothing (21.06 vs 21.08);
  with scheduling it buys the last 2 % (26.5 → 27.0 FPS, the best configuration measured).
- **Encode overlap, same story.** Rank 0's DiT grows from 88 to 182 ms when the 101 ms encode runs beside it; the
  station shortens only from 191 to 184 ms (+5 % FPS with `taehv`). The Wan encoder saturates the GPU exactly as
  the Wan decoder does.
- **Both overlaps break the block scheduler (H2 addendum).** It measures rank time with `torch.cuda.synchronize()`
  around the compute path, so side-stream decode is invisible (`t_total` rank 1 = 0.094 s with `wan` + overlap decode:
  it moves 8 blocks *onto* the still-busy rank 1 and FPS falls 17.4 → 13.4) and side-stream encode is folded into
  `t_dit` (rank 0 `t_dit` 0.174 s, inflating `dit_time_per_block`, so it moves only 5 blocks and leaves rank 1 waiting
  29 ms; 26.5 → 25.8 FPS). Any overlap fix must give the scheduler the real per-rank iteration period, not stage sums.
- **Ceiling reached with this encoder.** The balanced 2-rank period is (encode 101 + DiT 176 + decode/copy 6) / 2
  ≈ 142 ms → 28 FPS; `taehv` + overlap decode + scheduling measures 147 ms / 27.0 FPS. The only remaining levers are
  ones this plan did not scope: replace the encoder as well (TAEHV encoder, 1.47 M params, untested for quality
  through the DiT), give encode or decode its own rank (fix (b), needs a 3rd GPU), or shrink the DiT.

## 7. Answer to the research question

The TAEHV decoder is exactly as fast inside V2 as in isolation (9 ms per chunk on the final rank). It did not
"underperform in parallelism": the pipeline parallelism splits only the 30 DiT blocks, while both VAE halves sit
whole on one rank each. Swapping the decoder removed the final rank's 168 ms tail, at which point rank 0's
unsplittable Wan encoder (101 ms) plus its DiT blocks became the period, and the one-shot block scheduler then
correctly rebalanced to within 3 ms of the optimum. Overlapping either VAE half with the DiT on the same GPU recovers
only ~5 % because all three are GPU-bound, and it also blinds the scheduler. Measured best: 27.0 FPS on 2 A100s
(`--vae taehv --overlap_decode --schedule_block`) versus 8.96 FPS on 1 GPU with the Wan decoder and 17.4 FPS on
2 GPUs with it.

## 5. Files

| Action | Path |
|---|---|
| New | `StreamDiffusionV2/examples/bench_taehv_decoder.py` |
| New | `StreamDiffusionV2/examples/check_taehv_latent_space.py` (Step 2 latent-convention / contract check) |
| Edit | `StreamDiffusionV2/causvid/models/wan/wan_wrapper.py` (add `TAEHVDecoderWrapper`), `causvid/models/__init__.py` (registry), `causvid/models/wan/causal_stream_inference.py` (reads `args.vae`) |
| New | `StreamDiffusionV2/streamv2v/stage_timer.py` (Step 3: opt-in CUDA-event stage timer, `--timing_dir`) |
| New | `StreamDiffusionV2/examples/make_looped_clip.py`, `examples/run_step3.sh`, `examples/summarize_step3.py` (Step 3: 3× clip, run matrix with recorded commands, summary tables) |
| New | `StreamDiffusionV2/examples/run_step4.sh` (Step 4 variant matrix; `summarize_step3.py` reads Step 3 + Step 4 together) |
| Edit | `StreamDiffusionV2/streamv2v/inference.py`, `streamv2v/inference_pipe.py` (`--vae` flag; Step 3: `--timing_dir` marks in every rank loop, scheduler dump; Step 4: `--overlap_decode`, `--overlap_encode`, `AsyncDecodeSink`) |
| Reuse | `/home/joshua/taehv/taehv.py` (`TAEHV`, `StreamingTAEHV`); `WanVAEWrapper.stream_decode_to_pixel` as baseline |
| Untouched | `/home/joshua/StreamDiffusion.git` (v1) |

## 6. Verification

- Step 1 ✅ done: streaming == full-clip Mode A (exact); chunked Mode A differs; T=1 streaming decode = 8.7 ms
  (tripwire value), Wan = 165 ms.
- Step 2 ✅ done: `--vae taehv` produces 81 frames like `--vae wan` on 1 and 2 ranks; TAEHV decode of a real Wan latent = 29.1 dB vs Wan decode (normalized latent as-is); `taehv_parallel` gives 24/81 frames.
- Step 3 ✅ done: per-rank timing JSONL for all 18 runs; `wan` 2-rank FPS 15.0 reproduces Step 2 / README; 1 → 2 rank
  ratio per decoder reported; timer overhead nil (15.18 FPS timed vs untimed); repeats within 2.1 %.
- Step 4 ✅ done: overlap decode / overlap encode measured against the Step 3 baselines, with and without
  `--schedule_block`; output frame count and pixel statistics unchanged; scheduler interaction documented.
- Final write-up (section 7) answers the question with numbers: decode share of the final-rank stage, whether FPS
  scales with ranks for each decoder, whether the scheduler's block split changes when decode cost is
  weighted correctly, and the measured effect of each validated fix.
