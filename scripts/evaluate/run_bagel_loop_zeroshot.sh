#!/usr/bin/env bash
set -euo pipefail
# 16-NPU Phase-0 A0–A5. One hard prompt per card; six arms share ε on that card.
OUTPUT_DIR=${1:-/root/outputs/bagel_loop_zeroshot_v1}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=${PYTHON_BIN_FALLBACK:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
fi
MODEL_PATH=${MODEL_PATH:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_16.txt}
NUM_SHARDS=${NUM_SHARDS:-16}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
SEED=${SEED:-42}
ARMS=${ARMS:-}

mkdir -p "$OUTPUT_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/ema.safetensors" ]]; then
  echo "missing $MODEL_PATH/ema.safetensors" >&2
  ls -lh "$MODEL_PATH" | head -n 40 >&2
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "missing prompt file: $PROMPT_FILE" >&2
  exit 1
fi

n_prompts=$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE")
if [[ "$n_prompts" -lt 1 ]]; then
  echo "no prompts in $PROMPT_FILE" >&2
  exit 1
fi
if [[ "$n_prompts" -lt "$NUM_SHARDS" ]]; then
  NUM_SHARDS=$n_prompts
fi

echo "[launch] prompts=$n_prompts shards=$NUM_SHARDS python=$PYTHON_BIN"
echo "[launch] model=$MODEL_PATH"
echo "[launch] out=$OUTPUT_DIR image_size=$IMAGE_SIZE seed=$SEED arms=${ARMS:-all}"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "[launch] shard $i -> $OUTPUT_DIR/shard_${i}.log"
  ASCEND_RT_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
    scripts/evaluate/bagel_loop_zeroshot.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device npu:0 \
    --prompt-file "$PROMPT_FILE" \
    --image-size "$IMAGE_SIZE" \
    --seed "$SEED" \
    --cfg-interval-min 0.0 \
    --cfg-interval-max 1.0 \
    --num-loop-tokens 8 \
    --shard-id "$i" \
    --num-shards "$NUM_SHARDS" \
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

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/evaluate/bagel_loop_zeroshot.py \
  --merge-only \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device npu:0 \
  --arms "$ARMS"

echo "[launch] done fail=$fail gallery=$OUTPUT_DIR/index.html"
exit "$fail"
