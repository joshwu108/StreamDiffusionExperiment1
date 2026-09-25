#!/usr/bin/env bash
# plan.md Step 5: TAEHV encoder + decoder (--vae taehv_full). Compare with the Step 3/4 `taehv` runs.
#   1 GPU; 2 ranks; 2 ranks + schedule; 2 ranks + overlap decode + schedule (the Step 4 best) = 4 runs.
# Summarize together:  python examples/summarize_step3.py outputs/step3 outputs/step4 outputs/step5
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/joshua/.conda/envs/stream/bin/python
TORCHRUN=/home/joshua/.conda/envs/stream/bin/torchrun
export PYTHONPATH=.
export TAEHV_DIR=${TAEHV_DIR:-/home/joshua/taehv}
OUT=${1:-outputs/step5}
VIDEO=${2:-examples/original_x3.mp4}
COMMON=(--config_path configs/wan_causal_dmd_v2v.yaml --checkpoint_folder ckpts/wan_causal_dmd_v2v
        --prompt_file_path examples/prompt.txt --video_path "$VIDEO"
        --height 480 --width 832 --fps 16 --step 2 --vae taehv_full)
[[ -f "$VIDEO" ]] || $PY examples/make_looped_clip.py --dst "$VIDEO"
mkdir -p "$OUT"
echo "run started $(date -Is)" | tee "$OUT/commands.txt"

run() {  # $1 = cfg name, $2... = extra args
  local cfg=$1; shift; local d="$OUT/$cfg"; mkdir -p "$d"
  echo "== $cfg  $(date -Is)"
  local cmd
  if [[ "$cfg" == 1gpu* ]]; then
    cmd=(env CUDA_VISIBLE_DEVICES=0 $PY streamv2v/inference.py "${COMMON[@]}" --output_folder "$d" --timing_dir "$d" "$@")
  else
    cmd=($TORCHRUN --nproc_per_node=2 --master_port=29501 streamv2v/inference_pipe.py "${COMMON[@]}" --output_folder "$d" --timing_dir "$d" "$@")
  fi
  printf '%q ' "${cmd[@]}" >> "$OUT/commands.txt"; echo >> "$OUT/commands.txt"
  "${cmd[@]}" > "$d/log.txt" 2>&1
  grep -E "Average FPS|Video shape|Block split" "$d/log.txt" | tail -3
}

run 1gpu_taehv_full_r1
run 2gpu_taehv_full_r1
run 2gpu_taehv_full_sched_r1 --schedule_block
run 2gpu_taehv_full_od_sched_r1 --overlap_decode --schedule_block
echo "run finished $(date -Is)" | tee -a "$OUT/commands.txt"
