#!/usr/bin/env bash
set -euo pipefail

# One independent frozen-BAGEL process per NPU; prompts are round-robin sharded.
OUTPUT_DIR=${1:-/data/outputs/phase05_dynamic_prompt_hard16}
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_16.txt}
MODE=${MODE:-probe}
NUM_SHARDS=${NUM_SHARDS:-16}
MAX_PROMPTS=${MAX_PROMPTS:-16}
ALPHAS=${ALPHAS:-0,0.1,-0.1,0.2}

mkdir -p "$OUTPUT_DIR"
echo "[launch] mode=$MODE shards=$NUM_SHARDS model=$MODEL_PATH"
echo "[launch] prompts=$PROMPT_FILE output=$OUTPUT_DIR"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "[launch] NPU $i -> shard $i"
  ASCEND_RT_VISIBLE_DEVICES=$i "$PYTHON_BIN" -u \
    scripts/evaluate/bagel_dynamic_prompt_phase05.py \
    --mode "$MODE" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device npu:0 \
    --prompt-file "$PROMPT_FILE" \
    --alphas "$ALPHAS" \
    --body-start 12 \
    --body-end 20 \
    --step-fraction 0.35 \
    --probe-timestep 0.8 \
    --max-prompts "$MAX_PROMPTS" \
    --shard-id "$i" \
    --num-shards "$NUM_SHARDS" \
    > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "[launch] at least one shard failed; inspect $OUTPUT_DIR/shard_*.log" >&2
  exit "$fail"
fi

"$PYTHON_BIN" -u scripts/evaluate/bagel_dynamic_prompt_phase05.py \
  --merge-only \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR"

echo "[launch] done: $OUTPUT_DIR/manifest.json"
