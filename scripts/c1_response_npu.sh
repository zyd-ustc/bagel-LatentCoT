#!/usr/bin/env bash
# One jointly optimized C writer, with a frozen native BAGEL replica on each NPU.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/c1_response_npu.sh fresh_output_directory}
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
DATA=${DIAGNOSTIC_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/c1_text80.jsonl}
QUERIES=${QUERY_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/memory_response_queries16.jsonl}
CONFIG=${DIAGNOSTIC_CONFIG:-$ROOT/configs/c1_response.json}
PARENT=${C1_C:-/root/outputs/c1_pilot_20261012_113553/C/best.pt}
C0=${C0_PASS:-/root/outputs/c0_text16_20261012_111241/calibration.json}
mkdir -p "$(dirname "$RUN")"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
unset CUDA_VISIBLE_DEVICES
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    -m qwen_latent_cot.diagnostics.cli distill --distributed --device npu --topology C \
    --model "$MODEL" --data "$DATA" --query-data "$QUERIES" --config "$CONFIG" \
    --checkpoint "$PARENT" --c0-pass "$C0" --output "$RUN"
