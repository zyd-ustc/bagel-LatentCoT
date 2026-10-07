#!/usr/bin/env bash
# User-invoked diagnostic. The coding agent must not start GPU workers.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_memory_round_diagnostic_8gpu.sh output_directory}
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
MODEL_PATH=${MODEL_PATH:-/private/yida_workspace/models/BAGEL-7B-MoT}
PROMPTS=${PROMPTS:-$ROOT/data/prompts32.jsonl}
IFS=, read -r -a devices <<< "$GPUS"
declare -A seen=()
for device in "${devices[@]}"; do
    [[ "$device" =~ ^[0-9]+$ && -z ${seen[$device]:-} ]] || { echo 'GPUS must contain distinct integer GPU IDs' >&2; exit 1; }
    seen[$device]=1
done
args=(--model-path "$MODEL_PATH" --prompts "$PROMPTS" --prompt-count 32
    --seed "${SEED:-0}" --image-size "${IMAGE_SIZE:-512}" --num-timesteps "${NUM_TIMESTEPS:-50}"
    --probe-steps "${PROBE_STEPS:-0,24,48}" --timestep-shift "${TIMESTEP_SHIFT:-3.0}"
    --cfg-text-scale "${CFG_TEXT_SCALE:-4.0}" --start-layer "${START_LAYER:-0}"
    --end-layer "${END_LAYER:-8}" --num-shards "${#devices[@]}")
printf '%s\n' 'Check native projection and persistent-state contracts before diagnosis'
CUDA_VISIBLE_DEVICES="${devices[0]}" "$MODEL_PYTHON" "$ROOT/scripts/evaluate/validate_und_state.py" \
    --model-path "$MODEL_PATH" --depths 1,2,3 --start-layer "${START_LAYER:-0}" --end-layer "${END_LAYER:-8}" \
    --output "$RUN/e0.json" > "$RUN/e0.log" 2>&1
printf '%s\n' 'Prepare 32-prompt diagnostic plan and weight/source hashes'
"$MODEL_PYTHON" "$ROOT/scripts/evaluate/diagnose_memory_rounds.py" "${args[@]}" --prepare-only --output-dir "$RUN"
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT
for shard in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$MODEL_PYTHON" "$ROOT/scripts/evaluate/diagnose_memory_rounds.py" \
        "${args[@]}" --shard-index "$shard" --plan "$RUN/plan.json" --output-dir "$RUN/worker_$shard" \
        > "$RUN/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
printf '%s\n' 'Workers launched. Each GPU processes its own prompts; no cross-GPU tensor communication.'
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
trap - INT TERM EXIT
if (( failed )); then
    echo 'A worker failed. Inspect worker_*.log; incomplete runs will not be merged.' >&2
    exit 1
fi
"$MODEL_PYTHON" "$ROOT/scripts/evaluate/merge_memory_rounds.py" --run-dir "$RUN"
printf 'Completed: %s\n' "$RUN/summary.md"
