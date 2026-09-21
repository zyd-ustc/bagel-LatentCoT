#!/usr/bin/env bash
set -euo pipefail

ROOT_OUTPUT=${1:-/data/outputs/bagel_loop_t2i_full800_multiseed}
RUN_SCORE=${RUN_SCORE:-1}
BASE_RUNNER=scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh

mkdir -p "$ROOT_OUTPUT"

echo "[experiment] full GenEval2 800 prompts"
echo "[experiment] seed=42 arms=Z0,Z2,Z3,Z4,Z6"
SEED=42 ARMS=Z0,Z2,Z3,Z4,Z6 SCORE=$RUN_SCORE \
  bash "$BASE_RUNNER" "$ROOT_OUTPUT/seed_42_main"

for seed in 43 44; do
  echo "[experiment] seed=$seed arms=Z0"
  SEED=$seed ARMS=Z0 SCORE=$RUN_SCORE \
    bash "$BASE_RUNNER" "$ROOT_OUTPUT/seed_${seed}_z0"
done

if [[ "$RUN_SCORE" != "0" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}
  "$PYTHON_BIN" -u scripts/evaluate/summarize_bagel_loop_t2i_multiseed.py \
    --root-dir "$ROOT_OUTPUT"
fi

echo "[experiment] done root=$ROOT_OUTPUT"
