#!/usr/bin/env bash
set -euo pipefail
# v7: every I_r is x0_hat @ t_A. Official UND + official edit CFG, fresh ε, prefix stop.
# 3-prompt smoke first. SCORE=1 later when Qwen3-VL is on the box.
OUTPUT_DIR=${1:-/home/ma-user/work/outputs/draft_prefix_loop_v7_tstop}
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/ma-user/work/models/Bagel-7B-MoT}
PROMPT_FILE=${PROMPT_FILE:-experiments/data/geneval2_hard_16.txt}
BENCH=${BENCH:-experiments/data/geneval2_hard_16.jsonl}
SCORE=${SCORE:-0}
VLM_PATH=${VLM_PATH:-/home/ma-user/work/models/Qwen3-VL-8B-Instruct}
SCORE_DEVICE=${SCORE_DEVICE:-npu:0}
MAX_PROMPTS=${MAX_PROMPTS:-3}
GEN_TEXT=${GEN_TEXT:-a_only}
EDIT_STOP=${EDIT_STOP:-prefix}

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" -u scripts/evaluate/draft_prefix_loop.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --max-prompts "$MAX_PROMPTS" \
  --rounds 3 \
  --truncations 0.8 \
  --gen-text "$GEN_TEXT" \
  --edit-stop "$EDIT_STOP" \
  --device npu:0 \
  --image-height 512 --image-width 512 \
  --num-steps 50 --timestep-shift 3.0 \
  --cfg-text-scale 4.0 --cfg-img-scale 1.0 \
  --cfg-interval-min 0.4 --cfg-interval-max 1.0 \
  --cfg-renorm-type global \
  --reflection-max-tokens 1000

if [[ "$SCORE" == "1" ]]; then
  if [[ ! -d "$VLM_PATH" ]]; then
    echo "[score] skip: VLM not found at $VLM_PATH" >&2
    exit 0
  fi
  if ! curl -fsS http://127.0.0.1:18086 >/dev/null 2>&1; then
    echo "[score] starting Soft-TIFA on $SCORE_DEVICE"
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
      scripts/evaluate/serve_geneval2_soft_tifa.py \
      --model-path "$VLM_PATH" --device "$SCORE_DEVICE" --port 18086 \
      > "$OUTPUT_DIR/geneval2_server.log" 2>&1 &
    echo $! > "$OUTPUT_DIR/geneval2_server.pid"
    ready=0
    for _ in $(seq 1 90); do
      if curl -fsS http://127.0.0.1:18086 >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 5
    done
    if [[ "$ready" != "1" ]]; then
      echo "[score] Soft-TIFA did not come up; see $OUTPUT_DIR/geneval2_server.log" >&2
      exit 1
    fi
  fi
  PYTHONPATH="${PYTHONPATH:-}:$(pwd)" "$PYTHON_BIN" -u scripts/evaluate/score_draft_prefix_geneval2.py \
    --output-dir "$OUTPUT_DIR" \
    --benchmark-data "$BENCH"
fi
