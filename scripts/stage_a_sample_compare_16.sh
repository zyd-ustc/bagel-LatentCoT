#!/usr/bin/env bash
# Independent resident-model workers. Does not join the 64-card training group.
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${SAMPLE_OUTPUT:?Set a fresh output directory}
CHECKPOINT=${SAMPLE_CHECKPOINT:?Set a Stage A checkpoint file or training directory}
PROMPTS=${SAMPLE_PROMPTS:?Set a prepared Stage A prompt suite}
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
WORLD=${SAMPLE_WORKERS:-16}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
cd "$CODE"
[[ ! -e "$OUT" ]] || { echo "Use a fresh output directory: $OUT" >&2; exit 1; }
# Resolve once, before workers start, so later saves cannot change the comparison.
CHECKPOINT=$("$PYTHON" -c 'import sys; from qwen_latent_cot.diagnostics.inference import resolve_checkpoint; print(resolve_checkpoint(sys.argv[1]))' "$CHECKPOINT")
mkdir -p "$OUT/launch"
cp "$PROMPTS" "$OUT/prompts.jsonl"
printf '%s\n' "$CHECKPOINT" > "$OUT/checkpoint.txt"
pids=()
for ((rank=0;rank<WORLD;rank++)); do
  printf -v name 'worker_%02d' "$rank"
  env ASCEND_RT_VISIBLE_DEVICES="$rank" OMP_NUM_THREADS=4 \
    "$PYTHON" scripts/sample_stage_a_compare.py --model "$MODEL" --checkpoint "$CHECKPOINT" \
    --prompts "$OUT/prompts.jsonl" --output "$OUT" --rank "$rank" --world "$WORLD" --seed "${SAMPLE_SEED:-17}" \
    > "$OUT/launch/$name.log" 2>&1 < /dev/null &
  pids+=("$!"); echo "$!" > "$OUT/launch/$name.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo 'A sampling worker failed; inspect worker_*.json and launch logs' >&2; exit 1; }
exec "$PYTHON" scripts/render_stage_a_compare.py --output "$OUT"
