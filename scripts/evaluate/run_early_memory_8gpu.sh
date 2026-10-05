#!/usr/bin/env bash
# User-invoked formal evaluation; the coding agent must not execute this driver.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: run_early_memory_8gpu.sh output_directory}
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
printf '%s\n' '1/6: eight-GPU generation with diagnostic Memory snapshots'
bash "$ROOT/scripts/evaluate/launch_training_free_8gpu.sh" \
    --stage evaluation --start-layer "${START_LAYER:-0}" --end-layer "${END_LAYER:-8}" \
    --seeds "${SEEDS:-0,1}" --evaluations "${EVALUATIONS:-2}" --memory-slots 16 \
    --progress-start 0 --progress-end .5 --probe-steps "${PROBE_STEPS:-8,16,24}" \
    --arms BASE,GEN_LAYERWISE,MEMORY_DYNAMIC,MEMORY_STATIC,MEMORY_NO_READ
printf '%s\n' '2/6: eight-GPU final-image semantic and quality scoring'
export OUTDIR=$RUN/quality
bash "$ROOT/scripts/evaluate/launch_offline_8gpu.sh" quality \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
    --judge-model "$JUDGE" --geneval2-source "$OFFICIAL"
"$SCORER_PYTHON" "$ROOT/scripts/evaluate/merge_quality.py" \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl \
    --score-dirs "$RUN"/quality/worker_*/ --output-dir "$RUN/quality_report"
printf '%s\n' '3/6: eight-GPU native UND Memory QA and native ViT readout control'
export OUTDIR=$RUN/memory_qa
bash "$ROOT/scripts/evaluate/launch_offline_8gpu.sh" memory-qa \
    --manifests "$RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
    --model-path "$MODEL_PATH" --max-questions "${MAX_QUESTIONS:-4}" --max-count 12
printf '%s\n' '4/6: eight-GPU observed-image labeling (prompt and desired answer hidden)'
export OUTDIR=$RUN/memory_labels
bash "$ROOT/scripts/evaluate/launch_offline_8gpu.sh" memory-labels \
    --qa-dirs "$RUN"/memory_qa/worker_*/ --judge-model "$JUDGE" \
    --geneval2-source "$OFFICIAL" --minimum-confidence .8
printf '%s\n' '5/6: merge and validate full paired coverage'
"$SCORER_PYTHON" "$ROOT/scripts/evaluate/merge_memory_probe.py" \
    --qa-dirs "$RUN"/memory_qa/worker_*/ --label-dirs "$RUN"/memory_labels/worker_*/ \
    --output-dir "$RUN/memory_probe_report"
printf '%s\n' '6/6: completed (no training launched)'
printf 'Final-image report: %s\nMemory report: %s\n' "$RUN/quality_report/summary.md" "$RUN/memory_probe_report/summary.md"
