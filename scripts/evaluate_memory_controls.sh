#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?Usage: evaluate_memory_controls.sh RUN [prepare/generate options]}
shift
export NPUS=${NPUS:?Set NPUS to free physical device IDs}
cd "$ROOT"
bash scripts/benchmark_memory_controls_npu.sh "$RUN" "$@"
mkdir -p "$RUN/scores"
# Use each deployed official scorer, on all prepared arms. API-only benchmarks
# retain their existing export flow in memory_benchmark.py.
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  /root/venvs/geneval/bin/python scripts/benchmark_memory_controls.py score --run "$RUN" \
  --scorer geneval --benchmarks geneval --device cpu > "$RUN/scores/geneval.log" 2>&1 &
detector_pid=$!
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  /root/venvs/dpg/bin/python scripts/benchmark_memory_controls.py score --run "$RUN" \
  --scorer mplug --benchmarks dpg dsg --device cpu > "$RUN/scores/mplug.log" 2>&1 &
mplug_pid=$!
ASCEND_RT_VISIBLE_DEVICES=${NPUS%%,*} OMP_NUM_THREADS=4 \
  /root/venvs/qwen3vl-npu/bin/python scripts/benchmark_memory_controls.py score --run "$RUN" \
  --scorer geneval2 --benchmarks geneval2 --device npu:0 > "$RUN/scores/geneval2.log" 2>&1 &
vlm_pid=$!
status=0
for pid in "$detector_pid" "$mplug_pid" "$vlm_pid"; do
  wait "$pid" || status=1
done
${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python} scripts/benchmark_memory_controls.py summary --run "$RUN"
exit "$status"
