#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: train_interaction_npu.sh OUTPUT --parent|--resume PATH --train PATH --val PATH [options]}
shift
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
NPROC=${NPROC_PER_NODE:-16}
NNODES=${NNODES:-1}
ARGS=(-m qwen_latent_cot.interaction.cli sft --model "$MODEL" --device npu:0
      --config "$ROOT/configs/interaction_sft.json" --output "$RUN")
if [[ "$NPROC" == 1 && "$NNODES" == 1 ]]; then
  exec "$PYTHON" "${ARGS[@]}" "$@"
fi
if [[ "$NNODES" == 1 ]]; then
  DIST=(--standalone --nnodes=1)
else
  DIST=(--nnodes="$NNODES" --node-rank="${NODE_RANK:?}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29500}")
fi
exec "$PYTHON" -m torch.distributed.run "${DIST[@]}" --nproc-per-node="$NPROC" "${ARGS[@]}" --distributed "$@"
