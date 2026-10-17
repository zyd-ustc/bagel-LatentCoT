#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:-/root/outputs/c2_2000_early20_L28_R2_bench}
export NPUS=${NPUS:?Set NPUS to free physical device IDs}
cd "$ROOT"
[[ -f "$RUN/plan.json" ]] || /root/venvs/bagel-NPU/bin/python scripts/memory_benchmark.py prepare \
  --run "$RUN" --reuse /root/outputs/c2_2000_generation
mkdir -p "$RUN/scores"
# Validate the NPU judge on one reused image before the full generation job.
ASCEND_RT_VISIBLE_DEVICES=${NPUS%%,*} OMP_NUM_THREADS=4 \
  /root/venvs/qwen3vl-npu/bin/python scripts/memory_benchmark.py score --run "$RUN" \
  --scorer geneval2 --benchmarks geneval2 --device npu:0 --available-only --limit 1 \
  > "$RUN/scores/geneval2_preflight.log" 2>&1
bash scripts/benchmark_memory_npu.sh "$RUN"
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  /root/venvs/geneval/bin/python scripts/memory_benchmark.py score --run "$RUN" \
  --scorer geneval --benchmarks geneval --device cpu > "$RUN/scores/geneval.log" 2>&1 &
detector_pid=$!
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  /root/venvs/dpg/bin/python scripts/memory_benchmark.py score --run "$RUN" \
  --scorer mplug --benchmarks dpg dsg --device cpu > "$RUN/scores/mplug.log" 2>&1 &
mplug_pid=$!
ASCEND_RT_VISIBLE_DEVICES=${NPUS%%,*} OMP_NUM_THREADS=4 \
  /root/venvs/qwen3vl-npu/bin/python scripts/memory_benchmark.py score --run "$RUN" \
  --scorer geneval2 --benchmarks geneval2 --device npu:0 > "$RUN/scores/geneval2.log" 2>&1 &
vlm_pid=$!
status=0
for pid in "$detector_pid" "$mplug_pid" "$vlm_pid"; do
  wait "$pid" || status=1
done
/root/venvs/bagel-NPU/bin/python scripts/memory_benchmark.py summary --run "$RUN"
exit "$status"
