#!/usr/bin/env bash
# 64 cards, one comparison strip each. Same seed on every arm and every card.
# This entry is qwen_latent_cot.anchored.sample_compare. It does not call
# scripts/gen_npu.py and it does not load a MemoryBank.
# generate_image owns the Euler loop. Loop arms replace _forward_flow with
# AnchoredDualStateLoop.predict_flow for that call only.
# Columns: prompt, native, gen_only_400, dual_600, dual_shuffled,
# |dual-native|x10, |dual-gen|x10.
# OBSERVATION_CHECKPOINT is passed as --observation-checkpoint.
# Inference does not form a process group. Each visible card is one process.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
PROMPTS=${SAMPLE_PROMPTS:?set SAMPLE_PROMPTS to the prompt jsonl}
OUT=${SAMPLE_OUTPUT:?output directory on shared /data}
GEN=${GEN_ONLY_CHECKPOINT:?gen_only_step_000400.pt}
DUAL=${DUAL_CHECKPOINT:?dual_step_000600.pt}
READER=${OBSERVATION_CHECKPOINT:?C2-observe gen_full_query checkpoint}
if [[ -z ${VC_WORKER_HOSTS:-} ]]; then
  echo "VC_WORKER_HOSTS is required for the 64-card launch" >&2
  exit 1
fi
HOSTS=(${VC_WORKER_HOSTS//,/ })
CURRENT=$(hostname)
NODE=-1
for i in "${!HOSTS[@]}"; do
  [[ "${HOSTS[$i]}" == "$CURRENT".* ]] && NODE=$i
done
if [[ $NODE -lt 0 ]]; then
  echo "cannot determine rank for $CURRENT in $VC_WORKER_HOSTS" >&2
  exit 1
fi
NNODES=${MA_NUM_HOSTS:-${#HOSTS[@]}}
NPROC=${NPROC_PER_NODE:-16}
WORLD=$((NNODES * NPROC))
if [[ "$WORLD" -ne 64 || "$NNODES" -ne 4 || "$NPROC" -ne 16 ]]; then
  echo "sample compare expects 4 nodes x 16 devices, got ${NNODES} x ${NPROC}" >&2
  exit 1
fi
LINES=$(grep -cve '^[[:space:]]*$' "$PROMPTS")
if [[ "$LINES" -ne 64 ]]; then
  echo "need 64 prompt lines, got ${LINES}" >&2
  exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
mkdir -p "$OUT"
pids=()
for local in $(seq 0 15); do
  global=$((NODE * NPROC + local))
  log=$(printf '%s/rank_%02d.log' "$OUT" "$global")
  RANK="$global" WORLD_SIZE="$WORLD" LOCAL_RANK="$local" \
    "$PYTHON" -m qwen_latent_cot.anchored.sample_compare \
    --device npu \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output "$OUT" \
    --gen-only-checkpoint "$GEN" \
    --dual-checkpoint "$DUAL" \
    --observation-checkpoint "$READER" \
    --loop-config "${ANCHORED_LOOP_CONFIG:-$ROOT/configs/anchored_gen_only.json}" \
    --seed "${SAMPLE_SEED:-17}" \
    >"$log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "node ${NODE} had a failed card; see ${OUT}/rank_XX.log" >&2
  exit 1
fi
echo "node ${NODE} wrote ranks $((NODE * NPROC))..$((NODE * NPROC + NPROC - 1))"
