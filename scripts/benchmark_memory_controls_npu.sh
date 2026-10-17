#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?Usage: benchmark_memory_controls_npu.sh RUN [prepare/generate options]}
shift
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
IFS=',' read -ra DEVICES <<< "${NPUS:?Set NPUS to free physical device IDs}"
cd "$ROOT"
if [[ ! -f "$RUN/controls.json" ]]; then
  "$PYTHON" scripts/benchmark_memory_controls.py prepare --run "$RUN" "$@"
fi
mkdir -p "$RUN/generation"
pids=()
for rank in "${!DEVICES[@]}"; do
  ASCEND_RT_VISIBLE_DEVICES=${DEVICES[$rank]} "$PYTHON" scripts/benchmark_memory_controls.py generate \
    --run "$RUN" --device npu:0 --rank "$rank" --world "${#DEVICES[@]}" "$@" \
    > "$RUN/generation/worker_$(printf '%02d' "$rank").log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
