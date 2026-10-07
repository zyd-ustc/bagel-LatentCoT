#!/usr/bin/env bash
set -euo pipefail
# Run on H200. Set GPUS to the GPUs allocated to this experiment.
: "${MODEL_PATH:?set MODEL_PATH}" "${PROMPTS:?set PROMPTS}" "${OUTDIR:?set OUTDIR}"
PYTHON=${PYTHON:-/private/software/conda/envs/lcot/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IFS=, read -r -a devices <<< "$GPUS"
mkdir -p "$OUTDIR"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM EXIT
for shard in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$PYTHON" "$ROOT/scripts/evaluate/training_free.py" \
        --model-path "$MODEL_PATH" --prompts "$PROMPTS" --output-dir "$OUTDIR/worker_$shard" \
        --num-shards "${#devices[@]}" --shard-index "$shard" "$@" \
        > "$OUTDIR/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
trap - INT TERM EXIT
exit "$failed"
