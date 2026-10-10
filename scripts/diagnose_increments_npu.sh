#!/usr/bin/env bash
# User-run diagnostic, no semantic judge or training.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: diagnose_increments_npu.sh fresh_output_directory}
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
CONFIG=${CONFIG:-$ROOT/configs/increment_diagnostics.json}
PROMPTS=${DIAGNOSTIC_PROMPTS:-$ROOT/data/prompts32.jsonl}
NPUS=${NPUS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
IFS=, read -r -a devices <<< "$NPUS"
declare -A seen=()
for id in "${devices[@]}"; do
    [[ "$id" =~ ^[0-9]+$ && -z ${seen[$id]:-} ]] || { echo 'NPUS must contain distinct integer IDs' >&2; exit 1; }
    seen[$id]=1
done
[[ ${#devices[@]} -gt 0 ]] || exit 1
mkdir -p "$RUN/workers"
RUN=$(cd "$RUN" && pwd)
PLAN=$RUN/plan.json
CLI=$ROOT/scripts/diagnose_increments.py
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
env CUDA_VISIBLE_DEVICES= ASCEND_RT_VISIBLE_DEVICES= "$PYTHON" "$CLI" prepare \
    --model-path "$MODEL" --prompts "$PROMPTS" --config "$CONFIG" --plan "$PLAN"
SHARDS=$("$PYTHON" -c 'import json,sys;p=json.load(open(sys.argv[1]));print(min(int(sys.argv[2]),len(p["prompt_ids"])*len(p["config"]["seeds"])))' "$PLAN" "${#devices[@]}")
for ((shard=0; shard<SHARDS; shard++)); do
    env -u CUDA_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES="${devices[$shard]}" "$PYTHON" "$CLI" run \
        --plan "$PLAN" --device npu:0 --output-dir "$RUN/workers/worker_$shard" \
        --shard-index "$shard" --num-shards "$SHARDS" > "$RUN/workers/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "Diagnostic failed; inspect $RUN/workers/worker_*.log" >&2; exit 1; }
done
pids=()
env CUDA_VISIBLE_DEVICES= ASCEND_RT_VISIBLE_DEVICES= "$PYTHON" "$CLI" report --plan "$PLAN" --output-dir "$RUN"
trap - EXIT INT TERM
printf 'Completed: %s\n' "$RUN/diagnostics.html"
