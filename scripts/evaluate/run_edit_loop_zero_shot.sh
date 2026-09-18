#!/usr/bin/env bash
set -euo pipefail
# Four-arm diagnostic for the Markov edit loop.
# Usage: ./run_edit_loop_zero_shot.sh [OUTPUT_ROOT]
#
# Arms:
#   same_none    edit-noise=same  control=none   (current reconstruction-lock)
#   fresh_none   edit-noise=fresh control=none   (UND instruction, official noise)
#   fresh_gold   edit-noise=fresh control=gold   (strong semantic instruction)
#   fresh_fixed  edit-noise=fresh control=fixed  (watercolor appearance instruction)
OUTPUT_ROOT=${1:-/home/ma-user/work/outputs/edit_loop_diag_v3}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/ma-user/work/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_16.txt}
MAX_PROMPTS=${MAX_PROMPTS:-3}
ROUNDS=${ROUNDS:-3}
DEVICE=${DEVICE:-npu:0}

run_arm() {
  local tag=$1 noise=$2 control=$3
  local out="${OUTPUT_ROOT}/${tag}"
  mkdir -p "$out"
  echo "[arm] ${tag}  edit_noise=${noise} control=${control} -> ${out}"
  "$PYTHON_BIN" scripts/evaluate/edit_loop_zero_shot.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$out" \
    --prompt-file "$PROMPT_FILE" \
    --max-prompts "$MAX_PROMPTS" \
    --rounds "$ROUNDS" \
    --control "$control" \
    --edit-noise "$noise" \
    --device "$DEVICE" \
    --image-height 512 --image-width 512 \
    --reflection-max-tokens 1000
}

mkdir -p "$OUTPUT_ROOT"
run_arm same_none same none
run_arm fresh_none fresh none
run_arm fresh_gold fresh gold
run_arm fresh_fixed fresh fixed
echo "[done] four-arm sweep under ${OUTPUT_ROOT}"
