#!/usr/bin/env bash
# User-run native parity check followed by the seven-arm full Memory matrix.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_full_memory_depth_8gpu.sh output_directory}
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
export MODEL_PATH=${MODEL_PATH:-/private/yida_workspace/models/BAGEL-7B-MoT}
export ARMS=BASE,LAYERWISE_FULL_SEED_REPLACE,LAYERWISE_FULL_MEMORY_REPLACE
export LOOP_DEPTHS=${LOOP_DEPTHS:-1,2,3}
export MAX_PROMPTS=${MAX_PROMPTS:-8}
export MEMORY_SLOTS=0  # Full modes derive their capacity from each complete prompt.
export SEEDS=${SEEDS:-0}
export PROMPTS=${PROMPTS:-/private/yida_workspace/umm-anchored-eval-tools-d126833/data/hard16.jsonl}
export PROBE_STEPS=
export DIAGNOSTICS=0
export FULL_STATIC_CHECK=1
printf '%s\n' 'Check full static R1/R2/R3 velocity parity before generation'
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$MODEL_PYTHON" "$ROOT/scripts/evaluate/validate_full_memory.py" \
    --model-path "$MODEL_PATH" --depths "$LOOP_DEPTHS" \
    --start-layer "${START_LAYER:-0}" --end-layer "${END_LAYER:-8}" --output "$RUN/full_memory_e0.json"
bash "$ROOT/scripts/evaluate/run_memory_loop_8gpu.sh" "$RUN"
"$MODEL_PYTHON" "$ROOT/scripts/evaluate/export_comparison_html.py" --run-dir "$RUN"
