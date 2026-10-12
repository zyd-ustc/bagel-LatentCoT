#!/usr/bin/env bash
# E0 single-NPU first; all main visual/loop training and evaluation default 16 NPUs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:?usage: bash scripts/visual_training_npu.sh check|observe|joint|loop|eval|sample|label-states fresh_output [extra arguments]}
RUN=${2:?fresh output directory required}
shift 2
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
DATA=${VISUAL_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/visual_memory640.jsonl}
TEXT_DATA=${TEXT_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/c1_response_text2192.jsonl}
TEACHER=${TEXT_TEACHER:-/root/outputs/c1_response_expand_20261012_141729/best.pt}
C0=${C0_PASS:-/root/outputs/c0_text16_20261012_111241/calibration.json}
case "$MODE" in
  check) COMMAND=check; CONFIG=visual_memory_pilot ;;
  observe) COMMAND=train; CONFIG=visual_memory_pilot ;;
  joint) COMMAND=train; CONFIG=visual_gen_joint ;;
  loop) COMMAND=train; CONFIG=memory_loop_r2 ;;
  eval) COMMAND=eval; CONFIG=${VISUAL_STAGE_CONFIG:-visual_memory_pilot} ;;
  sample) COMMAND=sample; CONFIG=${VISUAL_STAGE_CONFIG:-memory_loop_r2} ;;
  label-states) COMMAND=label-states; CONFIG=visual_memory_pilot ;;
  *) echo 'Unknown visual mode' >&2; exit 1 ;;
esac
CONFIG=${DIAGNOSTIC_CONFIG:-$ROOT/configs/$CONFIG.json}
for PATH_TO_CHECK in "$PYTHON" "$DATA" "$TEACHER" "$C0" "$CONFIG"; do
  [[ -f "$PATH_TO_CHECK" ]] || { echo "Missing required file: $PATH_TO_CHECK" >&2; exit 1; }
done
if [[ "$COMMAND" == train ]]; then
  [[ -f "$TEXT_DATA" ]] || { echo "Missing replay data: $TEXT_DATA" >&2; exit 1; }
fi
ARGS=(-m qwen_latent_cot.diagnostics.visual_cli "$COMMAND" --device npu --model "$MODEL"
      --data "$DATA" --config "$CONFIG" --text-checkpoint "$TEACHER" --c0-pass "$C0" --output "$RUN")
if [[ "$COMMAND" == train ]]; then ARGS+=(--text-data "$TEXT_DATA"); fi
if [[ -n ${VISUAL_PARENT:-} ]]; then ARGS+=(--checkpoint "$VISUAL_PARENT"); fi
if [[ -n ${VISUAL_RESUME:-} ]]; then ARGS+=(--resume "$VISUAL_RESUME"); fi
if [[ "$MODE" == joint || "$MODE" == loop || "$MODE" == eval || "$MODE" == sample ]]; then
  [[ -n ${VISUAL_PARENT:-} || -n ${VISUAL_RESUME:-} ]] || { echo 'Set VISUAL_PARENT or VISUAL_RESUME to the selected visual checkpoint' >&2; exit 1; }
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
if [[ "$MODE" == check ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=${E0_VISIBLE_DEVICE:-0}
  exec "$PYTHON" "${ARGS[@]}" "$@"
fi
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
  "${ARGS[@]}" --distributed "$@"
