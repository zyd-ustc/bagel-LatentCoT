#!/usr/bin/env bash
set -euo pipefail
# Phase-0.5 paired semantic edits. All selected arms share ε for each case.
OUTPUT_DIR=${1:-/root/outputs/bagel_loop_zeroshot_v1}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=${PYTHON_BIN_FALLBACK:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
fi
MODEL_PATH=${MODEL_PATH:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_16.txt}
EDIT_FILE=${EDIT_FILE:-experiments/data/semantic_edit_phase05.jsonl}
SOURCE_IMAGE=${SOURCE_IMAGE:-}
SOURCE_PROMPT=${SOURCE_PROMPT:-}
NUM_SHARDS=${NUM_SHARDS:-16}
SEED=${SEED:-42}
ARMS=${ARMS:-}
K_VALUES=${K_VALUES:-}

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
case_args=()
if [[ -n "$EDIT_FILE" ]]; then
  if [[ ! -f "$EDIT_FILE" ]]; then
    echo "missing edit file: $EDIT_FILE" >&2
    exit 1
  fi
  case_args+=(--edit-file "$EDIT_FILE")
  n_prompts=$(grep -cve '^[[:space:]]*$' "$EDIT_FILE")
else
  if [[ ! -f "$PROMPT_FILE" || -z "$SOURCE_IMAGE" ]]; then
    echo "legacy mode requires PROMPT_FILE and SOURCE_IMAGE" >&2
    exit 1
  fi
  case_args+=(--prompt-file "$PROMPT_FILE" --source-image "$SOURCE_IMAGE" --source-prompt "$SOURCE_PROMPT")
  n_prompts=$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE")
fi
if [[ "$n_prompts" -lt 1 ]]; then
  echo "no edit cases" >&2
  exit 1
fi
if [[ "$n_prompts" -lt "$NUM_SHARDS" ]]; then
  NUM_SHARDS=$n_prompts
fi

echo "[launch] prompts=$n_prompts shards=$NUM_SHARDS python=$PYTHON_BIN"
echo "[launch] model=$MODEL_PATH"
echo "[launch] out=$OUTPUT_DIR source_geometry=native seed=$SEED arms=${ARMS:-all} k_values=${K_VALUES:-none}"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "[launch] shard $i -> $OUTPUT_DIR/shard_${i}.log"
  ASCEND_RT_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
    scripts/evaluate/bagel_loop_zeroshot.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device npu:0 \
    "${case_args[@]}" \
    --seed "$SEED" \
    --cfg-interval-min 0.0 \
    --cfg-interval-max 1.0 \
    --num-loop-tokens 8 \
    --shard-id "$i" \
    --num-shards "$NUM_SHARDS" \
    --arms "$ARMS" \
    --k-values "$K_VALUES" \
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
  --arms "$ARMS" \
  --k-values "$K_VALUES"

echo "[launch] done fail=$fail gallery=$OUTPUT_DIR/index.html"
exit "$fail"
