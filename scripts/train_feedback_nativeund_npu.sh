#!/usr/bin/env bash
# C2 text feedback starts from Observe best with a fresh optimizer and sampler.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-train}
RUN=${2:-/root/outputs/feedback_nativeund_attn_$(date +%Y%m%d_%H%M%S)}
export NPROC_PER_NODE=${NPROC_PER_NODE:-16}
export DIAGNOSTIC_CONFIG="$ROOT/configs/visual_feedback.json"
export VISUAL_PARENT=${OBSERVE_PARENT:-/root/outputs/observe_nativeund_attn_20261015/best.pt}
unset VISUAL_RESUME WRITER_INIT TEXT_TEACHER
case "$MODE" in
 smoke) export WANDB_MODE=disabled
        exec bash "$ROOT/scripts/visual_training_npu.sh" feedback "$RUN" --steps 2 --validation-max-families 4 --validation-max-rows 8 ;;
 train) export WANDB_MODE=${WANDB_MODE:-online}
        export WANDB_NAME=${WANDB_NAME:-feedback_nativeund_attn_full_20k}
        exec bash "$ROOT/scripts/visual_training_npu.sh" feedback "$RUN" --steps 20000 ;;
 *) echo 'Usage: train_feedback_nativeund_npu.sh smoke|train [output]' >&2; exit 1 ;;
esac
