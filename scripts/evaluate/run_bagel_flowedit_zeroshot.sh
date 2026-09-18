#!/usr/bin/env bash
set -euo pipefail
# 16-NPU FlowEdit 4.1. One process per card. Extra pairs round-robin.
OUTPUT_DIR=${1:-/root/outputs/bagel_flowedit_zeroshot_v1}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=${PYTHON_BIN_FALLBACK:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
fi
MODEL_PATH=${MODEL_PATH:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}
PAIRS_FILE=${PAIRS_FILE:-experiments/data/geneval2_hard_16_flowedit.jsonl}
PREFIX_MODE=${PREFIX_MODE:-text}
NUM_SHARDS=${NUM_SHARDS:-16}

mkdir -p "$OUTPUT_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found: $PYTHON_BIN" >&2
  exit 1
fi

n_pairs=$(grep -cve '^[[:space:]]*$' "$PAIRS_FILE")
if [[ "$n_pairs" -lt 1 ]]; then
  echo "no pairs in $PAIRS_FILE" >&2
  exit 1
fi
if [[ "$n_pairs" -lt "$NUM_SHARDS" ]]; then
  NUM_SHARDS=$n_pairs
fi

echo "[launch] pairs=$n_pairs shards=$NUM_SHARDS python=$PYTHON_BIN"
echo "[launch] model=$MODEL_PATH"
echo "[launch] out=$OUTPUT_DIR"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "[launch] shard $i -> $OUTPUT_DIR/shard_${i}.log"
  ASCEND_RT_VISIBLE_DEVICES=$i "$PYTHON_BIN" -u scripts/evaluate/bagel_flowedit_zeroshot.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device npu:0 \
    --pairs-file "$PAIRS_FILE" \
    --prefix-mode "$PREFIX_MODE" \
    --n-min 0.2 --n-max 0.8 --n-avg 1 \
    --shard-id "$i" \
    --num-shards "$NUM_SHARDS" \
    > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &
  pids+=($!)
done

tail -F "$OUTPUT_DIR"/shard_*.log &
tail_pid=$!
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
kill "$tail_pid" 2>/dev/null || true
wait "$tail_pid" 2>/dev/null || true

"$PYTHON_BIN" -u scripts/evaluate/bagel_flowedit_zeroshot.py \
  --merge-only \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device npu:0

echo "[launch] done fail=$fail gallery=$OUTPUT_DIR/index.html"
exit "$fail"
