#!/usr/bin/env bash
set -euo pipefail
: "${OUTDIR:?set OUTDIR for this stage}"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
SCORER_PYTHON=${SCORER_PYTHON:-/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IFS=, read -r -a devices <<< "$GPUS"
mkdir -p "$OUTDIR"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM EXIT
for shard in "${!devices[@]}"; do
 CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$SCORER_PYTHON" "$ROOT/scripts/evaluate/quality_report.py" \
   --output-dir "$OUTDIR/worker_$shard" --num-shards "${#devices[@]}" --shard-index "$shard" "$@" \
   > "$OUTDIR/worker_$shard.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
trap - INT TERM EXIT
exit "$failed"
