#!/usr/bin/env bash
# Each topology is one jointly trained writer across all 16 NPUs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/c1_npu.sh fresh_output_directory}
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
DATA=${DIAGNOSTIC_DATA:-$ROOT/data/c1_text80.jsonl}
CONFIG=${DIAGNOSTIC_CONFIG:-$ROOT/configs/c1_pilot.json}
C0=${C0_PASS:-/root/outputs/c0_text16_20261012_111241/calibration.json}
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
unset CUDA_VISIBLE_DEVICES
for TOPOLOGY in L C; do
    echo "Starting C1-$TOPOLOGY: 16 synchronized NPU workers"
    "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
        -m qwen_latent_cot.diagnostics.cli train --distributed \
        --model "$MODEL" --data "$DATA" --config "$CONFIG" --device npu \
        --stage C1 --topology "$TOPOLOGY" --c0-pass "$C0" --output "$RUN/$TOPOLOGY"
done
echo "Completed C1-L/C: $RUN"
