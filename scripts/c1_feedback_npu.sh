#!/usr/bin/env bash
# Compress explicit P+C feedback into the existing writer; native BAGEL frozen.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/c1_feedback_npu.sh fresh_output_directory [extra arguments]}
shift
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
DATA=${FEEDBACK_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/c1_historical_feedback_pilot.jsonl}
CONFIG=${DIAGNOSTIC_CONFIG:-$ROOT/configs/c1_feedback.json}
PARENT=${C1_C:-/root/outputs/c1_response_expand_20261012_141729/best.pt}
C0=${C0_PASS:-/root/outputs/c0_text16_20261012_111241/calibration.json}
for REQUIRED in "$PYTHON" "$DATA" "$CONFIG" "$PARENT" "$C0"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing required file: $REQUIRED" >&2; exit 1; }
done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=${FEEDBACK_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
unset CUDA_VISIBLE_DEVICES
IFS=',' read -r -a DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
cd "$ROOT"
if [[ ${#DEVICES[@]} == 1 ]]; then
  exec "$PYTHON" -m qwen_latent_cot.diagnostics.cli train --device npu:0 --topology C --stage C1 \
    --model "$MODEL" --data "$DATA" --config "$CONFIG" --checkpoint "$PARENT" \
    --c0-pass "$C0" --output "$RUN" "$@"
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${#DEVICES[@]}" \
  -m qwen_latent_cot.diagnostics.cli train --distributed --device npu --topology C --stage C1 \
  --model "$MODEL" --data "$DATA" --config "$CONFIG" --checkpoint "$PARENT" \
  --c0-pass "$C0" --output "$RUN" "$@"
