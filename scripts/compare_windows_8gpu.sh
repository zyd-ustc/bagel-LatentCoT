#!/usr/bin/env bash
# User-run formal evaluation. Each GPU handles independent prompt/seed pairs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: compare_windows_8gpu.sh fresh_output_directory}
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
SCORER_PYTHON=${SCORER_PYTHON:-/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/private/yida_workspace/models/BAGEL-7B-MoT}
PROMPTS=${PROMPTS:-$ROOT/data/prompts32.jsonl}
CONFIG=${CONFIG:-$ROOT/configs/window_comparison.json}
JUDGE=${JUDGE_MODEL:-/private/yida_workspace/models/Qwen3-VL-8B-Instruct}
OFFICIAL=${GENEVAL2_SOURCE:-/private/yida_workspace/umm-anchored-eval-tools-d126833/GenEval2/evaluation.py}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
IFS=, read -r -a devices <<< "$GPUS"
declare -A seen=()
for device in "${devices[@]}"; do
    [[ "$device" =~ ^[0-9]+$ && -z ${seen[$device]:-} ]] || { echo 'GPUS must contain distinct integer IDs' >&2; exit 1; }
    seen[$device]=1
done
[[ ${#devices[@]} -gt 0 ]] || { echo 'Select at least one GPU' >&2; exit 1; }
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
PLAN=$RUN/plan.json
CLI=$ROOT/scripts/compare_windows.py
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM EXIT
wait_workers() {
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "Worker failed. Inspect worker logs under $RUN" >&2
            exit 1
        fi
    done
    pids=()
}
printf '%s\n' '1/5: bind denoising windows, native weights, source, prompts and sampling'
CUDA_VISIBLE_DEVICES= "$MODEL_PYTHON" "$CLI" prepare --model-path "$MODEL_PATH" \
    --prompts "$PROMPTS" --config "$CONFIG" --plan "$PLAN"
NUM_SHARDS=$("$MODEL_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(min(int(sys.argv[2]),len(p["prompt_ids"])*len(p["seeds"])))' "$PLAN" "${#devices[@]}")
printf '%s\n' '2/5: real-weight numerical checks for five loop arms and inactive-window parity; stop on failure'
CUDA_VISIBLE_DEVICES="${devices[0]}" "$MODEL_PYTHON" "$CLI" validate \
    --plan "$PLAN" --output "$RUN/e0.json" > "$RUN/e0.log" 2>&1
printf '%s\n' '3/5: generate Base + five R2 time windows; progress in generation/worker_*.log'
mkdir -p "$RUN/generation" "$RUN/quality"
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$MODEL_PYTHON" "$CLI" generate \
        --plan "$PLAN" --output-dir "$RUN/generation/worker_$shard" \
        --num-shards "$NUM_SHARDS" --shard-index "$shard" \
        > "$RUN/generation/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
wait_workers
printf '%s\n' '4/5: score semantic constraints and quality; progress in quality/worker_*.log'
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$SCORER_PYTHON" "$CLI" score \
        --manifests "$RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
        --judge-model "$JUDGE" --geneval2-source "$OFFICIAL" \
        --output-dir "$RUN/quality/worker_$shard" --num-shards "$NUM_SHARDS" --shard-index "$shard" \
        > "$RUN/quality/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
wait_workers
printf '%s\n' '5/5: validate complete pairs, merge metrics and export portable HTML'
CUDA_VISIBLE_DEVICES= "$SCORER_PYTHON" "$CLI" report \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --score-dirs "$RUN"/quality/worker_*/ \
    --output-dir "$RUN/quality_report" --run-dir "$RUN"
trap - INT TERM EXIT
printf 'Completed: %s\n' "$RUN/comparison.html"
