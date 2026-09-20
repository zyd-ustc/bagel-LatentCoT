#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-/root/outputs/bagel_loop_t2i_phase05}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}
MODEL_PATH=${MODEL_PATH:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_16.txt}
NUM_SHARDS=${NUM_SHARDS:-16}
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
SEED=${SEED:-42}
ARMS=${ARMS:-}
MAX_PROMPTS=${MAX_PROMPTS:-0}

mkdir -p "$OUTPUT_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/ema.safetensors" ]]; then
  echo "missing $MODEL_PATH/ema.safetensors" >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "missing prompt file: $PROMPT_FILE" >&2
  exit 1
fi

n_prompts=$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE")
if [[ "$MAX_PROMPTS" -gt 0 && "$MAX_PROMPTS" -lt "$n_prompts" ]]; then
  n_prompts=$MAX_PROMPTS
fi
if [[ "$n_prompts" -lt 1 ]]; then
  echo "no prompts in $PROMPT_FILE" >&2
  exit 1
fi
if [[ "$n_prompts" -lt "$NUM_SHARDS" ]]; then
  NUM_SHARDS=$n_prompts
fi

echo "[launch] T2I prompts=$n_prompts shards=$NUM_SHARDS geometry=${HEIGHT}x${WIDTH}"
echo "[launch] model=$MODEL_PATH out=$OUTPUT_DIR arms=${ARMS:-all}"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "[launch] shard $i -> $OUTPUT_DIR/shard_${i}.log"
  ASCEND_RT_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH:-}:$(pwd)" \
    "$PYTHON_BIN" -u scripts/evaluate/bagel_loop_t2i_zeroshot.py \
      --model-path "$MODEL_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --device npu:0 \
      --prompt-file "$PROMPT_FILE" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --seed "$SEED" \
      --num-loop-tokens 8 \
      --shard-id "$i" \
      --num-shards "$NUM_SHARDS" \
      --max-prompts "$MAX_PROMPTS" \
      --arms "$ARMS" \
      > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" "$PYTHON_BIN" -u \
  scripts/evaluate/bagel_loop_t2i_zeroshot.py \
    --merge-only \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device npu:0 \
    --arms "$ARMS"

echo "[launch] done fail=$fail gallery=$OUTPUT_DIR/index.html"
exit "$fail"
