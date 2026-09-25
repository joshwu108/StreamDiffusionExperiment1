#!/usr/bin/env bash
# plan.md Step 3: run the instrumented pipeline over the full configuration matrix, sequentially.
#
# Matrix (each config twice, _r1/_r2):
#   1 GPU  (streamv2v/inference.py)      x --vae {wan,taehv,taehv_parallel}
#   2 rank (streamv2v/inference_pipe.py) x --vae {wan,taehv,taehv_parallel} x {no schedule, --schedule_block}
# = 18 runs. Every run writes output_000.mp4, log.txt, timing_rank{r}.jsonl, schedule_rank{r}.json
# under outputs/step3/<cfg>/. Summarize with: python examples/summarize_step3.py
#
# Usage, from the repo root:   bash examples/run_step3.sh [outputs/step3] [examples/original_x3.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/joshua/.conda/envs/stream/bin/python
TORCHRUN=/home/joshua/.conda/envs/stream/bin/torchrun
export PYTHONPATH=.
export TAEHV_DIR=${TAEHV_DIR:-/home/joshua/taehv}

OUT=${1:-outputs/step3}
VIDEO=${2:-examples/original_x3.mp4}
COMMON=(--config_path configs/wan_causal_dmd_v2v.yaml --checkpoint_folder ckpts/wan_causal_dmd_v2v
        --prompt_file_path examples/prompt.txt --video_path "$VIDEO"
        --height 480 --width 832 --fps 16 --step 2)

if [[ ! -f "$VIDEO" ]]; then
  echo "== making looped clip $VIDEO"
  $PY examples/make_looped_clip.py --dst "$VIDEO"
fi
mkdir -p "$OUT"
echo "run started $(date -Is)" | tee "$OUT/commands.txt"

gpu_state() {
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv,noheader
}

run_1gpu() {  # $1 = vae, $2 = repeat tag
  local cfg="1gpu_$1_$2"; local d="$OUT/$cfg"; mkdir -p "$d"
  echo "== $cfg  $(date -Is)"; gpu_state > "$d/gpu_before.txt"
  local cmd=(env CUDA_VISIBLE_DEVICES=0 $PY streamv2v/inference.py "${COMMON[@]}"
             --vae "$1" --output_folder "$d" --timing_dir "$d")
  printf '%q ' "${cmd[@]}" >> "$OUT/commands.txt"; echo >> "$OUT/commands.txt"
  "${cmd[@]}" > "$d/log.txt" 2>&1
  grep -E "Average FPS|Video shape" "$d/log.txt" | tail -2
}

run_2rank() {  # $1 = vae, $2 = "" | "--schedule_block", $3 = repeat tag
  local tag="2gpu_$1"; [[ -n "$2" ]] && tag="${tag}_sched"
  local cfg="${tag}_$3"; local d="$OUT/$cfg"; mkdir -p "$d"
  echo "== $cfg  $(date -Is)"; gpu_state > "$d/gpu_before.txt"
  local cmd=($TORCHRUN --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py "${COMMON[@]}"
             --vae "$1" --output_folder "$d" --timing_dir "$d" $2)
  printf '%q ' "${cmd[@]}" >> "$OUT/commands.txt"; echo >> "$OUT/commands.txt"
  "${cmd[@]}" > "$d/log.txt" 2>&1
  grep -E "Average FPS|Video shape|Block split" "$d/log.txt" | tail -3
}

for rep in r1 r2; do
  for vae in wan taehv taehv_parallel; do
    run_1gpu "$vae" "$rep"
    run_2rank "$vae" "" "$rep"
    run_2rank "$vae" "--schedule_block" "$rep"
  done
done
echo "run finished $(date -Is)" | tee -a "$OUT/commands.txt"
