#!/usr/bin/env bash
# User-invoked formal evaluation. Never start this driver from the coding agent.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_memory_loop_8gpu.sh output_directory}
mkdir -p "$RUN"
RUN=$(cd "$RUN" && pwd)
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export MODEL_PYTHON=${MODEL_PYTHON:-/private/software/conda/envs/lcot/bin/python}
export SCORER_PYTHON=${SCORER_PYTHON:-/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python}
export PYTHON=$MODEL_PYTHON
export MODEL_PATH=${MODEL_PATH:-/private/yida_workspace/models/BAGEL-7B-MoT}
TOOLS=${EVAL_TOOLS:-/private/yida_workspace/umm-anchored-eval-tools-d126833}
JUDGE=${JUDGE_MODEL:-/private/yida_workspace/models/Qwen3-VL-8B-Instruct}
OFFICIAL=${GENEVAL2_SOURCE:-$TOOLS/GenEval2/evaluation.py}
if [[ -z ${PROMPTS:-} ]]; then
    export PROMPTS=$RUN/prompts.jsonl
    "$MODEL_PYTHON" "$ROOT/scripts/evaluate/build_early_eval_manifest.py" \
        --hard "$TOOLS/data/hard128.jsonl" --ordinary "$TOOLS/data/easy16.jsonl" --output "$PROMPTS"
fi
export OUTDIR=$RUN/generation
extra=()
[[ ${DIAGNOSTICS:-0} == 1 ]] && extra+=(--diagnostics)
[[ -n ${PROBE_STEPS:-} ]] && extra+=(--probe-steps "$PROBE_STEPS")
printf '%s\n' '1/3: paired native Base / hidden-state Memory loop generation'
bash "$ROOT/scripts/evaluate/launch_training_free_8gpu.sh" \
    --stage evaluation --start-layer "${START_LAYER:-0}" --end-layer "${END_LAYER:-8}" \
    --seeds "${SEEDS:-0,1}" --loop-rounds "${LOOP_ROUNDS:-1}" --memory-slots "${MEMORY_SLOTS:-8}" \
    --progress-start "${PROGRESS_START:-0}" --progress-end "${PROGRESS_END:-1}" \
    --arms "${ARMS:-BASE,MEMORY_LOOP}" "${extra[@]}"
printf '%s\n' '2/3: paired final-image semantic and quality scoring'
export OUTDIR=$RUN/quality
bash "$ROOT/scripts/evaluate/launch_offline_8gpu.sh" quality \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
    --judge-model "$JUDGE" --geneval2-source "$OFFICIAL"
printf '%s\n' '3/3: validate paired coverage and merge quality report'
"$SCORER_PYTHON" "$ROOT/scripts/evaluate/merge_quality.py" \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl \
    --score-dirs "$RUN"/quality/worker_*/ --output-dir "$RUN/quality_report"
printf 'Completed: %s\n' "$RUN/quality_report/summary.md"
