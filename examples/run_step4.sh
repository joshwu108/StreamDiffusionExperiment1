#!/usr/bin/env bash
# plan.md Step 4: validate the fixes Step 3 pointed at, on the 2-rank pipeline.
#
#   --overlap_decode  (fix a+c) final rank decodes on a side stream + pinned non-blocking host copy
#   --overlap_encode  rank 0 encodes chunk k+1 on a side stream while the DiT runs chunk k
#
# Matrix: --vae {wan,taehv} x {od, oe, od_oe} x {no schedule, --schedule_block} = 12 runs, one repeat
# (Step 3 repeats agreed within 2 %). Baselines are the Step 3 runs in outputs/step3.
# Summarize both together:  python examples/summarize_step3.py outputs/step3 outputs/step4
#
# Usage, from the repo root:   bash examples/run_step4.sh [outputs/step4] [examples/original_x3.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/joshua/.conda/envs/stream/bin/python
TORCHRUN=/home/joshua/.conda/envs/stream/bin/torchrun
export PYTHONPATH=.
export TAEHV_DIR=${TAEHV_DIR:-/home/joshua/taehv}

OUT=${1:-outputs/step4}
VIDEO=${2:-examples/original_x3.mp4}
COMMON=(--config_path configs/wan_causal_dmd_v2v.yaml --checkpoint_folder ckpts/wan_causal_dmd_v2v
        --prompt_file_path examples/prompt.txt --video_path "$VIDEO"
        --height 480 --width 832 --fps 16 --step 2)

[[ -f "$VIDEO" ]] || $PY examples/make_looped_clip.py --dst "$VIDEO"
mkdir -p "$OUT"
echo "run started $(date -Is)" | tee "$OUT/commands.txt"

run_2rank() {  # $1 = vae, $2 = variant tag (od|oe|od_oe), $3 = "" | "--schedule_block", $4 = repeat tag
  local flags=()
  [[ "$2" == *od* ]] && flags+=(--overlap_decode)
  [[ "$2" == *oe* ]] && flags+=(--overlap_encode)
  local tag="2gpu_$1_$2"; [[ -n "$3" ]] && tag="${tag}_sched"
  local cfg="${tag}_$4"; local d="$OUT/$cfg"; mkdir -p "$d"
  echo "== $cfg  $(date -Is)"
  nvidia-smi --query-gpu=index,clocks.sm,power.draw,temperature.gpu --format=csv,noheader > "$d/gpu_before.txt"
  local cmd=($TORCHRUN --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py "${COMMON[@]}"
             --vae "$1" --output_folder "$d" --timing_dir "$d" "${flags[@]}" $3)
  printf '%q ' "${cmd[@]}" >> "$OUT/commands.txt"; echo >> "$OUT/commands.txt"
  "${cmd[@]}" > "$d/log.txt" 2>&1
  grep -E "Average FPS|Video shape|Block split" "$d/log.txt" | tail -3
}

for vae in wan taehv; do
  for variant in od oe od_oe; do
    run_2rank "$vae" "$variant" "" r1
    run_2rank "$vae" "$variant" "--schedule_block" r1
  done
done
echo "run finished $(date -Is)" | tee -a "$OUT/commands.txt"
