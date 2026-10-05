#!/usr/bin/env bash
set -euo pipefail
TASK=${1:?usage: launch_offline_8gpu.sh quality|memory-qa|memory-labels arguments}; shift
: "${OUTDIR:?set OUTDIR for this stage}"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
SCORER_PYTHON=${SCORER_PYTHON:-/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
case "$TASK" in
 quality) SCRIPT=quality_report.py; WORKER_PYTHON=$SCORER_PYTHON ;;
 memory-qa) SCRIPT=memory_probe_worker.py; WORKER_PYTHON=$MODEL_PYTHON ;;
 memory-labels) SCRIPT=memory_probe_labels.py; WORKER_PYTHON=$SCORER_PYTHON ;;
 *) printf '%s\n' "unknown task: $TASK" >&2; exit 2 ;;
esac
IFS=, read -r -a devices <<< "$GPUS"
mkdir -p "$OUTDIR"
pids=()
for shard in "${!devices[@]}"; do
 CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$WORKER_PYTHON" "$ROOT/scripts/evaluate/$SCRIPT" \
   --output-dir "$OUTDIR/worker_$shard" --num-shards "${#devices[@]}" --shard-index "$shard" "$@" \
   > "$OUTDIR/worker_$shard.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
