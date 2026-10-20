#!/usr/bin/env bash
# Observe/Feedback text supervision on indexed corpus.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:?usage: visual_training_npu.sh observe|feedback|check|eval|source-review output [arguments]}
RUN=${2:?fresh output required}; shift 2
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
CORPUS=${TRAINING_CORPUS:-/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus}
C0=${C0_PASS:-/root/outputs/c0_text16_20261012_111241/calibration.json}
case "$MODE" in
 observe) COMMAND=train; CONFIG=visual_memory_pilot; STAGE=C2-observe ;;
 feedback) COMMAND=train; CONFIG=visual_feedback; STAGE=C2-feedback ;;
 check|eval|source-review) COMMAND=$MODE; CONFIG=${VISUAL_STAGE_CONFIG:-visual_memory_pilot} ;;
 *) echo 'Unknown training mode' >&2; exit 1 ;;
esac
CONFIG=${DIAGNOSTIC_CONFIG:-$ROOT/configs/$CONFIG.json}
ARGS=(-m qwen_latent_cot.diagnostics.visual_cli "$COMMAND" --device npu --model "$MODEL" --config "$CONFIG" --c0-pass "$C0" --output "$RUN")
if [[ "$COMMAND" == train ]]; then
 ARGS+=(--corpus "$CORPUS" --stage "$STAGE" --wandb-mode "${WANDB_MODE:-online}"
        --wandb-project "${WANDB_PROJECT:-bagel-LatentCoT}")
 [[ -z ${WANDB_ENTITY:-} ]] || ARGS+=(--wandb-entity "$WANDB_ENTITY")
 [[ -z ${WANDB_NAME:-} ]] || ARGS+=(--wandb-name "$WANDB_NAME")
else
 ARGS+=(--data "${VISUAL_DATA:?small diagnostic JSONL required}")
fi
if [[ "$COMMAND" == train && -n ${VISUAL_RESUME:-} ]]; then
 ARGS+=(--resume "$VISUAL_RESUME")
elif [[ -n ${VISUAL_PARENT:-} ]]; then
 ARGS+=(--checkpoint "$VISUAL_PARENT")
elif [[ -n ${WRITER_INIT:-} ]]; then
 ARGS+=(--writer-init "$WRITER_INIT")
fi
[[ -z ${TEXT_TEACHER:-} ]] || ARGS+=(--text-checkpoint "$TEXT_TEACHER")
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
NPROC=${NPROC_PER_NODE:-16}
if [[ "$MODE" == check || "$NPROC" == 1 ]]; then
 export ASCEND_RT_VISIBLE_DEVICES=${E0_VISIBLE_DEVICE:-0}
 exec "$PYTHON" "${ARGS[@]}" "$@"
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" "${ARGS[@]}" --distributed "$@"
