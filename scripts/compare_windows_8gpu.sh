#!/usr/bin/env bash
# User-run formal evaluation. Each CUDA/Ascend device handles independent pairs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: compare_windows_8gpu.sh fresh_output_directory}
[[ ! -e "$RUN" || ${RESUME:-0} == 1 ]] || { echo 'Use a fresh output directory or RESUME=1 for the identical run' >&2; exit 1; }
if [[ -x /root/venvs/bagel-NPU/bin/python ]]; then
    default_model_python=/root/venvs/bagel-NPU/bin/python
elif [[ -x /private/software/conda/envs/lcot/bin/python ]]; then
    default_model_python=/private/software/conda/envs/lcot/bin/python
else
    default_model_python=python3
fi
MODEL_PYTHON=${MODEL_PYTHON:-$default_model_python}
BACKEND=${BACKEND:-auto}
if [[ "$BACKEND" == auto ]]; then
    BACKEND=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$MODEL_PYTHON" -c 'from qwen_latent_cot.bagel.accelerator import resolve_device; print(resolve_device().type)')
fi
[[ "$BACKEND" == cuda || "$BACKEND" == npu ]] || { echo 'Formal evaluation requires BACKEND=cuda or npu' >&2; exit 1; }
if [[ "$BACKEND" == npu ]]; then
    default_model=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
    default_judge=/data/model/Qwen3-VL-8B-Instruct
    default_official=/root/npu-eval-tools/GenEval2/evaluation.py
    default_scorer_python=$MODEL_PYTHON
else
    default_model=/private/yida_workspace/models/BAGEL-7B-MoT
    default_judge=/private/yida_workspace/models/Qwen3-VL-8B-Instruct
    default_official=/private/yida_workspace/umm-anchored-eval-tools-d126833/GenEval2/evaluation.py
    default_scorer_python=/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python
fi
SCORER_PYTHON=${SCORER_PYTHON:-$default_scorer_python}
MODEL_PATH=${MODEL_PATH:-$default_model}
# Do not inherit generic PROMPTS from an unrelated experiment.
PROMPTS=${COMPARISON_PROMPTS:-}
CONFIG=${CONFIG:-$ROOT/configs/repeat_r1_pilot.json}
JUDGE=${JUDGE_MODEL:-$default_judge}
OFFICIAL=${GENEVAL2_SOURCE:-$default_official}
GPUS=${NPUS:-${GPUS:-0,1,2,3,4,5,6,7}}
IFS=, read -r -a devices <<< "$GPUS"
declare -A seen=()
for device in "${devices[@]}"; do
    [[ "$device" =~ ^[0-9]+$ && -z ${seen[$device]:-} ]] || { echo 'NPUS/GPUS must contain distinct integer IDs' >&2; exit 1; }
    seen[$device]=1
done
[[ ${#devices[@]} -gt 0 ]] || { echo 'Select at least one device' >&2; exit 1; }
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
PLAN=$RUN/plan.json
CLI=$ROOT/scripts/compare_windows.py
if [[ -z "$PROMPTS" ]]; then
    PROMPTS=$("$MODEL_PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(json.load(open(sys.argv[1]))["benchmark"]); print(p if p.is_absolute() else pathlib.Path(sys.argv[2])/p)' "$CONFIG" "$ROOT")
fi
set_worker_environment() {
    local physical_id=$1
    if [[ "$BACKEND" == npu ]]; then
        worker_environment=(env -u CUDA_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES="$physical_id")
    else
        worker_environment=(env -u ASCEND_RT_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="$physical_id")
    fi
}
run_cpu() { env CUDA_VISIBLE_DEVICES= ASCEND_RT_VISIBLE_DEVICES= "$@"; }
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
run_cpu "$MODEL_PYTHON" "$CLI" prepare --model-path "$MODEL_PATH" \
    --prompts "$PROMPTS" --config "$CONFIG" --plan "$PLAN"
NUM_SHARDS=$("$MODEL_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(min(int(sys.argv[2]),len(p["prompt_ids"])*len(p["seeds"])))' "$PLAN" "${#devices[@]}")
printf '%s\n' '2/5: real-weight numerical checks for configured arms and native-path numerical checks; stop on failure'
set_worker_environment "${devices[0]}"
"${worker_environment[@]}" "$MODEL_PYTHON" "$CLI" validate \
    --plan "$PLAN" --device "$BACKEND:0" --output "$RUN/e0.json" > "$RUN/e0.log" 2>&1
printf '%s\n' '3/5: generate all configured paired arms; progress in generation/worker_*.log'
mkdir -p "$RUN/generation" "$RUN/quality"
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    set_worker_environment "${devices[$shard]}"
    "${worker_environment[@]}" "$MODEL_PYTHON" "$CLI" generate \
        --plan "$PLAN" --device "$BACKEND:0" --output-dir "$RUN/generation/worker_$shard" \
        --num-shards "$NUM_SHARDS" --shard-index "$shard" \
        > "$RUN/generation/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
wait_workers
printf '%s\n' '4/5: score semantic constraints and quality; progress in quality/worker_*.log'
for ((shard=0; shard<NUM_SHARDS; shard++)); do
    set_worker_environment "${devices[$shard]}"
    "${worker_environment[@]}" "$SCORER_PYTHON" "$CLI" score \
        --manifests "$RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
        --device "$BACKEND:0" --judge-model "$JUDGE" --geneval2-source "$OFFICIAL" \
        --output-dir "$RUN/quality/worker_$shard" --num-shards "$NUM_SHARDS" --shard-index "$shard" \
        > "$RUN/quality/worker_$shard.log" 2>&1 &
    pids+=("$!")
done
wait_workers
printf '%s\n' '5/5: validate complete pairs, merge metrics and export portable HTML'
run_cpu "$SCORER_PYTHON" "$CLI" report \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --score-dirs "$RUN"/quality/worker_*/ \
    --output-dir "$RUN/quality_report" --run-dir "$RUN"
trap - INT TERM EXIT
printf 'Completed: %s\n' "$RUN/comparison.html"
