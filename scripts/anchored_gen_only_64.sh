#!/usr/bin/env bash
# GEN-only Anchored Dual-State Loop on 64 Ascend cards: 4 workers, 16 devices each.
# Run this same command on every worker. VC_WORKER_HOSTS selects the rank.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
TRAIN=${GEN_ONLY_TRAIN:?caption-image train jsonl with absolute image paths}
RUN=${RUN_DIR:?fresh output directory}
LOOP_CONFIG=${GEN_ONLY_LOOP_CONFIG:-$ROOT/configs/anchored_gen_only.json}
TRAIN_CONFIG=${GEN_ONLY_TRAIN_CONFIG:-$ROOT/configs/anchored_gen_only_train.json}
if [[ -z ${VC_WORKER_HOSTS:-} ]]; then
  echo "VC_WORKER_HOSTS is required for the 64-card launch" >&2
  exit 1
fi
HOSTS=(${VC_WORKER_HOSTS//,/ })
CURRENT=$(hostname)
RANK=-1
for i in "${!HOSTS[@]}"; do
  [[ "${HOSTS[$i]}" == "$CURRENT".* ]] && RANK=$i
done
if [[ $RANK -lt 0 ]]; then
  echo "cannot determine rank for $CURRENT in $VC_WORKER_HOSTS" >&2
  exit 1
fi
MASTER=${HOSTS[${MASTER_TRAIN_INDEX:-0}]}
NNODES=${MA_NUM_HOSTS:-${#HOSTS[@]}}
NPROC=${NPROC_PER_NODE:-16}
WORLD=$((NNODES * NPROC))
if [[ "$WORLD" -ne 64 ]]; then
  echo "GEN-only 64-card entry expects 64 devices, got ${NNODES} x ${NPROC}" >&2
  exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
ARGS=(-m qwen_latent_cot.anchored.cli train-gen-only
  --device npu --model "$MODEL" --train "$TRAIN" --output "$RUN"
  --loop-config "$LOOP_CONFIG" --train-config "$TRAIN_CONFIG")
if [[ -n ${GEN_ONLY_RESUME:-} ]]; then
  ARGS+=(--resume "$GEN_ONLY_RESUME")
fi
exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" --node_rank="$RANK" --nproc_per_node="$NPROC" \
  --master_addr="$MASTER" --master_port="${MASTER_PORT:-29500}" \
  "${ARGS[@]}" --distributed
