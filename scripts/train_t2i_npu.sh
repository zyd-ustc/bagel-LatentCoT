#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
if [[ ${NNODES:-1} == 1 ]]; then DIST=(--standalone --nnodes=1); else
 DIST=(--nnodes="$NNODES" --node-rank="${NODE_RANK:?}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29673}")
fi
exec "$PYTHON" -m torch.distributed.run "${DIST[@]}" --nproc-per-node="${NPROC_PER_NODE:-16}" \
 -m qwen_latent_cot.t2i.cli train --device npu:0 --config "$ROOT/configs/t2i_sft.json" "$@"
