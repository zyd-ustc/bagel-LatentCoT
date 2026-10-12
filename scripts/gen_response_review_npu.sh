#!/usr/bin/env bash
# Fixed response step 100, original four images, three t values, sixteen NPUs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/gen_response_review_npu.sh fresh_output_directory}
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
DATA=${GEN_DATA:-/data/zyd_workspace/bagel-LatentCoT/datasets/memory_gen_recheck4.jsonl}
CHECKPOINT=${GEN_CHECKPOINT:-/root/outputs/c1_response_20261012_134848/last.pt}
BASELINE=${GEN_BASELINE:-/root/outputs/gen_recheck_C_t789_20261012}
cd "$ROOT"
"$PYTHON" -u scripts/gen_npu.py --data "$DATA" --checkpoint "$CHECKPOINT" \
    --eval-t 0.7 0.8 0.9 --noise-seeds 0 1 2 3 --time-groups 2 --output "$RUN"
"$PYTHON" scripts/render_gen_review.py --run "$RUN" --data "$DATA" --baseline "$BASELINE"
tar -czf "$RUN/review.tar.gz" -C "$RUN" review summary.json plan.json gen.jsonl pixel_comparison.json
printf 'Review archive: %s/review.tar.gz\n' "$RUN"
