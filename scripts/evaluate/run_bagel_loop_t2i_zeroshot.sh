#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${1:-/data/outputs/bagel_loop_t2i_phase05_128}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}
MODEL_PATH=${MODEL_PATH:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_128.txt}
BENCHMARK_DATA=${BENCHMARK_DATA:-experiments/data/geneval2_hard_128.jsonl}
NUM_SHARDS=${NUM_SHARDS:-16}
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
SEED=${SEED:-42}
ARMS=${ARMS:-}
K_VALUES=${K_VALUES:-}
MAX_PROMPTS=${MAX_PROMPTS:-0}
SCORE=${SCORE:-auto}
SCORE_SERVER_URL=${SCORE_SERVER_URL:-http://127.0.0.1:18086}
SCORE_PORT=${SCORE_PORT:-18086}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-4}
VLM_PATH=${VLM_PATH:-/data/bagel-LatentCoT/models/Qwen3-VL-8B-Instruct}
SCORE_DEVICE=${SCORE_DEVICE:-npu:0}

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
if [[ ! -f "$BENCHMARK_DATA" ]]; then
  echo "missing GenEval2 benchmark: $BENCHMARK_DATA" >&2
  exit 1
fi
case "$SCORE" in
  0|1|auto) ;;
  *) echo "SCORE must be 0, 1, or auto" >&2; exit 1 ;;
esac
if [[ "$SCORE" == "1" ]] && \
   ! curl -fsS "$SCORE_SERVER_URL" >/dev/null 2>&1 && \
   [[ ! -f "$VLM_PATH/config.json" ]]; then
  echo "scoring requested but no server at $SCORE_SERVER_URL and no VLM at $VLM_PATH" >&2
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
echo "[launch] model=$MODEL_PATH out=$OUTPUT_DIR arms=${ARMS:-all} k_values=${K_VALUES:-none}"
echo "[launch] GenEval2 score=$SCORE benchmark=$BENCHMARK_DATA"

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

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" "$PYTHON_BIN" -u \
  scripts/evaluate/bagel_loop_t2i_zeroshot.py \
    --merge-only \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device npu:0 \
    --arms "$ARMS" \
    --k-values "$K_VALUES"

score_server_pid=""
cleanup_score_server() {
  if [[ -n "$score_server_pid" ]] && kill -0 "$score_server_pid" 2>/dev/null; then
    kill "$score_server_pid" 2>/dev/null || true
    wait "$score_server_pid" 2>/dev/null || true
  fi
}
trap cleanup_score_server EXIT

if [[ "$fail" == "0" && "$SCORE" != "0" ]]; then
  score_ready=0
  if curl -fsS "$SCORE_SERVER_URL" >/dev/null 2>&1; then
    score_ready=1
  elif [[ -f "$VLM_PATH/config.json" ]] && \
       [[ "$SCORE_SERVER_URL" == "http://127.0.0.1:${SCORE_PORT}" ]]; then
    echo "[score] starting GenEval2 Soft-TIFA on $SCORE_DEVICE"
    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u \
      scripts/evaluate/serve_geneval2_soft_tifa.py \
      --model-path "$VLM_PATH" \
      --device "$SCORE_DEVICE" \
      --port "$SCORE_PORT" \
      > "$OUTPUT_DIR/geneval2_server.log" 2>&1 &
    score_server_pid=$!
    echo "$score_server_pid" > "$OUTPUT_DIR/geneval2_server.pid"
    for _ in $(seq 1 90); do
      if curl -fsS "$SCORE_SERVER_URL" >/dev/null 2>&1; then
        score_ready=1
        break
      fi
      sleep 5
    done
  fi

  if [[ "$score_ready" == "1" ]]; then
    echo "[score] evaluating all arms on $n_prompts GenEval2 prompts"
    PYTHONPATH="${PYTHONPATH:-}:$(pwd)" "$PYTHON_BIN" -u \
      scripts/evaluate/score_bagel_loop_t2i_geneval2.py \
      --output-dir "$OUTPUT_DIR" \
      --benchmark-data "$BENCHMARK_DATA" \
      --server-url "$SCORE_SERVER_URL" \
      --batch-size "$SCORE_BATCH_SIZE" \
      2>&1 | tee "$OUTPUT_DIR/geneval2_score.log"
  elif [[ "$SCORE" == "1" ]]; then
    echo "GenEval2 Soft-TIFA did not start; see $OUTPUT_DIR/geneval2_server.log" >&2
    exit 1
  else
    echo "[score] skipped: set SCORE=1 with a live server or VLM_PATH" >&2
  fi
fi

echo "[launch] done fail=$fail gallery=$OUTPUT_DIR/index.html"
exit "$fail"
