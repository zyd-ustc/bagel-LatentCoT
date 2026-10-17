#!/usr/bin/env bash
# Two-step 16-NPU smoke, then continue the same native UND experiment to 20000.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:-/root/outputs/observe_nativeund_attn_$(date +%Y%m%d_%H%M%S)}
export NPROC_PER_NODE=${NPROC_PER_NODE:-16}
export DIAGNOSTIC_CONFIG="$ROOT/configs/visual_memory_pilot.json"
export MODEL_PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
unset VISUAL_PARENT WRITER_INIT TEXT_TEACHER VISUAL_RESUME
WANDB_MODE=disabled bash "$ROOT/scripts/visual_training_npu.sh" observe "${RUN}_smoke" \
  --steps 2 --validation-max-families 2
export VISUAL_RESUME="${RUN}_smoke/last.pt"
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_NAME=${WANDB_NAME:-observe_nativeund_attn_full_20k}
exec bash "$ROOT/scripts/visual_training_npu.sh" observe "$RUN" --steps 20000
