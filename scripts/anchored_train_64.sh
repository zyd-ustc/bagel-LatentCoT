#!/usr/bin/env bash
# Two-stage Anchored Dual-State Loop on 64 Ascend cards: 4 workers, 16 devices each.
# Run this same command on every worker. VC_WORKER_HOSTS selects the rank.
# Stage 1 is GEN-only. Stage 2 loads those weights and a frozen C2-observe reader.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
TRAIN=${ANCHORED_TRAIN:?caption-image jsonl, or the annotation-index corpus directory}
RUN=${RUN_DIR:?fresh output directory}
if [[ ! -d "$TRAIN" && -z ${ANCHORED_VAL:-} ]]; then
  echo "ANCHORED_VAL is required when ANCHORED_TRAIN is a JSONL" >&2
  exit 1
fi
OBSERVATION=${OBSERVATION_CHECKPOINT:?C2-observe format-4 checkpoint whose training config resolves to gen_full_query}
LOOP_CONFIG=${ANCHORED_LOOP_CONFIG:-$ROOT/configs/anchored_gen_only.json}
TRAIN_CONFIG=${ANCHORED_TRAIN_CONFIG:-$ROOT/configs/anchored_curriculum_train.json}
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
  echo "anchored curriculum expects 64 devices, got ${NNODES} x ${NPROC}" >&2
  exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
ARGS=(-m qwen_latent_cot.anchored.cli train-anchored
  --device npu --model "$MODEL" --train "$TRAIN" --output "$RUN"
  --loop-config "$LOOP_CONFIG" --train-config "$TRAIN_CONFIG"
  --observation-checkpoint "$OBSERVATION")
if [[ -n ${ANCHORED_VAL:-} ]]; then
  ARGS+=(--val "$ANCHORED_VAL")
fi
if [[ -n ${ANCHORED_RESUME:-} ]]; then
  ARGS+=(--resume "$ANCHORED_RESUME")
fi
exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" --node_rank="$RANK" --nproc_per_node="$NPROC" \
  --master_addr="$MASTER" --master_port="${MASTER_PORT:-29500}" \
  "${ARGS[@]}" --distributed
