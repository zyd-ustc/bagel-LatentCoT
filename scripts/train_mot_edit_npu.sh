#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?OUTPUT};shift
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export HCCL_CONNECT_TIMEOUT=1800 TOKENIZERS_PARALLELISM=false
ARGS=(-m qwen_latent_cot.mot_edit.cli train --model "$MODEL" --device npu:0 --output "$RUN" --config "${MOT_CONFIG:-$ROOT/configs/mot_edit_8000.json}")
NPROC=${NPROC_PER_NODE:-16};NNODES=${NNODES:-4}
if [[ "$NPROC" == 1 && "$NNODES" == 1 ]]; then exec "$PYTHON" "${ARGS[@]}" "$@"; fi
exec "$PYTHON" -m torch.distributed.run --nnodes="$NNODES" --node-rank="${NODE_RANK:?}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29653}" --nproc-per-node="$NPROC" "${ARGS[@]}" --distributed "$@"
