#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:-/root/outputs/c2_2000_early20_L28_R2_bench}
shift || true
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
DEVICES=${NPUS:?Set NPUS to the free physical device IDs, e.g. NPUS=8,10,12,14}
IFS=',' read -ra DEVICES <<< "$DEVICES"
cd "$ROOT"
pids=()
for rank in "${!DEVICES[@]}"; do
  mkdir -p "$RUN/generation"
  ASCEND_RT_VISIBLE_DEVICES=${DEVICES[$rank]} "$PYTHON" scripts/memory_benchmark.py generate \
    --run "$RUN" --device npu:0 --rank "$rank" --world "${#DEVICES[@]}" "$@" \
    > "$RUN/generation/worker_$(printf '%02d' "$rank").log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
