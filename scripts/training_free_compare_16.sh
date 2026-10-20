#!/usr/bin/env bash
# One resident model per visible device; workers shard a fixed prompt list.
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${SAMPLE_OUTPUT:?Set a fresh sampling output directory}
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
PARENT=${OBSERVATION_CHECKPOINT:-/data/zyd_workspace/bagel-LatentCoT/checkpoints/observe_highnoise_full_20261017/best.pt}
PROMPTS=${SAMPLE_PROMPTS:-$CODE/configs/training_free_prompts_64.jsonl}
FIRST=${FIRST_WORKER:-0}
LAST=${LAST_WORKER:-15}
mkdir -p "$OUT/launch"
cd "$CODE"
for ((rank=FIRST;rank<=LAST;rank++)); do
  printf -v name 'worker_%02d' "$rank"
  if [[ -f "$OUT/launch/$name.pid" ]] && kill -0 "$(cat "$OUT/launch/$name.pid")" 2>/dev/null; then
    echo "Worker $rank is already running" >&2; exit 1
  fi
  if [[ -e "$OUT/launch/$name.log" ]]; then
    echo "Worker $rank already has a launch log; use a fresh output directory" >&2; exit 1
  fi
  setsid nohup env ASCEND_RT_VISIBLE_DEVICES="$rank" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" scripts/sample_training_free_compare.py --model "$MODEL" --parent "$PARENT" \
    --prompts "$PROMPTS" --output "$OUT" --rank "$rank" --world 16 --seed "${SAMPLE_SEED:-17}" \
    > "$OUT/launch/$name.log" 2>&1 < /dev/null &
  echo "$!" > "$OUT/launch/$name.pid"
  echo "Started worker=$rank pid=$!"
done
