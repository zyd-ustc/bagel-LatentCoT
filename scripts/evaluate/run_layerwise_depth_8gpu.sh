#!/usr/bin/env bash
# User-run 16-prompt depth comparison; no architecture or sampling changes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_layerwise_depth_8gpu.sh output_directory}
export LOOP_DEPTHS=${LOOP_DEPTHS:-1,2,3}
export ARMS=BASE,LAYERWISE_MEMORY_KV
export SEEDS=${SEEDS:-0}
export PROMPTS=${PROMPTS:-/private/yida_workspace/umm-anchored-eval-tools-d126833/data/hard16.jsonl}
export PROBE_STEPS=
export DIAGNOSTICS=0
bash "$ROOT/scripts/evaluate/run_memory_loop_8gpu.sh" "$RUN"
"${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}" \
    "$ROOT/scripts/evaluate/export_comparison_html.py" --run-dir "$RUN"
