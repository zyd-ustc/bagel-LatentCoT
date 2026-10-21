#!/usr/bin/env bash
# Reuse a prepared GenEval plan. Full default: 553 prompts, four seeds, two arms.
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: benchmark_stage_a.sh PREPARED_RUN}
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
SCORER=${GENEVAL_PYTHON:-/root/venvs/geneval/bin/python}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
cd "$CODE"
CHECKPOINT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN/plan.json")
SEEDS=$("$PYTHON" -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["seeds"])))' "$RUN/plan.json")
for seed in $SEEDS; do
  printf -v directory 'seed_%04d' "$seed"
  SAMPLE_OUTPUT="$RUN/samples/$directory" SAMPLE_CHECKPOINT="$CHECKPOINT" SAMPLE_PROMPTS="$RUN/prompts.jsonl" SAMPLE_SEED="$seed" \
    bash scripts/stage_a_sample_compare_16.sh
done
"$PYTHON" scripts/benchmark_stage_a.py export --run "$RUN"
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "$SCORER" scripts/benchmark_stage_a.py score --run "$RUN"
