#!/usr/bin/env bash
# Formal GPU evaluation is user-run; the coding agent must not launch it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_und_state_8gpu.sh fresh_output_directory}
[[ ! -e "$RUN" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
export SCORER_PYTHON=${SCORER_PYTHON:-/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python}
export MODEL_PATH=${MODEL_PATH:-/private/yida_workspace/models/BAGEL-7B-MoT}
export PROMPTS=${PROMPTS:-$ROOT/data/prompts32.jsonl}
export MAX_PROMPTS=${MAX_PROMPTS:-32}
export SEEDS=${SEEDS:-0}
export LOOP_DEPTHS=${LOOP_DEPTHS:-2}
export START_LAYER=${START_LAYER:-0} END_LAYER=${END_LAYER:-8}
JUDGE=${JUDGE_MODEL:-/private/yida_workspace/models/Qwen3-VL-8B-Instruct}
OFFICIAL=${GENEVAL2_SOURCE:-/private/yida_workspace/umm-anchored-eval-tools-d126833/GenEval2/evaluation.py}
IFS=, read -r -a devices <<< "$GPUS"
declare -A seen=()
for device in "${devices[@]}"; do
    [[ "$device" =~ ^[0-9]+$ && -z ${seen[$device]:-} ]] || { echo 'GPUS must contain distinct integer IDs' >&2; exit 1; }
    seen[$device]=1
done
printf '%s\n' 'Check native projection, R0, full prompt capacity, state and cache contracts'
CUDA_VISIBLE_DEVICES="${devices[0]}" "$MODEL_PYTHON" "$ROOT/scripts/evaluate/validate_und_state.py" \
    --model-path "$MODEL_PATH" --depths "$LOOP_DEPTHS" --start-layer "$START_LAYER" --end-layer "$END_LAYER" \
    --output "$RUN/e0.json" > "$RUN/e0.log" 2>&1
export OUTDIR=$RUN/generation PYTHON=$MODEL_PYTHON
printf '%s\n' '1/3: native Base and persistent UND Memory generation'
bash "$ROOT/scripts/evaluate/launch_training_free_8gpu.sh" \
    --stage evaluation --max-prompts "$MAX_PROMPTS" --seeds "$SEEDS" --loop-depths "$LOOP_DEPTHS" \
    --start-layer "$START_LAYER" --end-layer "$END_LAYER"
export OUTDIR=$RUN/quality
printf '%s\n' '2/3: paired semantic and quality scoring'
bash "$ROOT/scripts/evaluate/launch_offline_8gpu.sh" \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
    --judge-model "$JUDGE" --geneval2-source "$OFFICIAL"
printf '%s\n' '3/3: merge paired report and portable image comparison'
"$SCORER_PYTHON" "$ROOT/scripts/evaluate/merge_quality.py" \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --score-dirs "$RUN"/quality/worker_*/ \
    --output-dir "$RUN/quality_report"
"$MODEL_PYTHON" "$ROOT/scripts/evaluate/export_comparison_html.py" --run-dir "$RUN"
printf 'Completed: %s\n' "$RUN/quality_report/summary.md"
