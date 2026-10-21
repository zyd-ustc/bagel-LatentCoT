#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: train_correction_npu.sh OUTPUT --corpus CORPUS [options]};shift
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
NPROC=${NPROC_PER_NODE:-16};NNODES=${NNODES:-1}
ARGS=(-m qwen_latent_cot.correction.cli train --model "$MODEL" --device npu:0 --config "$ROOT/configs/correction.json" --output "$RUN")
if [[ "$NPROC" == 1 && "$NNODES" == 1 ]];then exec "$PYTHON" "${ARGS[@]}" "$@";fi
if [[ "$NNODES" == 1 ]];then DIST=(--standalone --nnodes=1);else DIST=(--nnodes="$NNODES" --node-rank="${NODE_RANK:?}" --master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29641}");fi
exec "$PYTHON" -m torch.distributed.run "${DIST[@]}" --nproc-per-node="$NPROC" "${ARGS[@]}" --distributed "$@"
