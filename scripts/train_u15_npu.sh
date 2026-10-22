#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} TOKENIZERS_PARALLELISM=false
PYTHON=${MODEL_PYTHON:?set MODEL_PYTHON to an isolated torch_npu environment}
STAGE=${1:?usage: train_u15_npu.sh t2i\|caption\|correct [CLI options]};shift
NPROC=${NPROC_PER_NODE:-16};NNODES=${NNODES:-1}
ARGS=(-m qwen_latent_cot.u15.cli "$STAGE" --device npu:0 "$@")
if [[ "$NPROC" == 1 && "$NNODES" == 1 ]];then exec "$PYTHON" "${ARGS[@]}";fi
if [[ "$NNODES" == 1 ]];then DIST=(--standalone --nnodes=1);else DIST=(--nnodes="$NNODES" --node-rank="${NODE_RANK:?}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29651}");fi
exec "$PYTHON" -m torch.distributed.run "${DIST[@]}" --nproc-per-node="$NPROC" "${ARGS[@]}" --distributed
