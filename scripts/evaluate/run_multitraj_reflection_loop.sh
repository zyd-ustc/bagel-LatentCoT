#!/usr/bin/env bash
set -euo pipefail

# Multi-trajectory reflection loop, t=0.9 only, 16 GenEval2-hard prompts.
# Usage: ./run_multitraj_reflection_loop.sh [OUTPUT_DIR] [PROMPT_FILE]

OUTPUT_DIR=${1:-/home/ma-user/work/outputs/multitraj_hard16_t090}
PROMPT_FILE=${2:-experiments/data/geneval2_hard_16.txt}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/ma-user/work/models/Bagel-7B-MoT}

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" scripts/evaluate/multitraj_reflection_loop.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --max-prompts 16 \
  --device npu:0 \
  --image-height 512 \
  --image-width 512 \
  --num-steps 50 \
  --timestep-shift 3.0 \
  --cfg-text-scale 4.0 \
  --cfg-img-scale 1.0 \
  --truncations 0.9 \
  --regen-noise fresh \
  --regen-variants latent,reflection,both \
  --reflection-max-tokens 160
