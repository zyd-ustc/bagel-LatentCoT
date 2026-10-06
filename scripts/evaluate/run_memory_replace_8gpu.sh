#!/usr/bin/env bash
# User-run comparison: static compression, append feedback, replacement feedback.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_memory_replace_8gpu.sh output_directory}
export ARMS=BASE,LAYERWISE_SEED_REPLACE,LAYERWISE_MEMORY_KV,LAYERWISE_MEMORY_REPLACE
export LOOP_DEPTHS=${LOOP_DEPTHS:-1}
export SEEDS=${SEEDS:-0}
export PROMPTS=${PROMPTS:-/private/yida_workspace/umm-anchored-eval-tools-d126833/data/hard16.jsonl}
export PROBE_STEPS=
export DIAGNOSTICS=0
bash "$ROOT/scripts/evaluate/run_memory_loop_8gpu.sh" "$RUN"
"${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}" \
    "$ROOT/scripts/evaluate/export_comparison_html.py" --run-dir "$RUN"
