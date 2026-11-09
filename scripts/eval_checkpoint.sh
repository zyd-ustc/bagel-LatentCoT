#!/bin/bash
# Run the two development albums for one checkpoint on a single 16-chip node.
# usage: eval_checkpoint.sh <checkpoint> <output-dir> [linear-limit] [prompt-limit]
set -eu
CODE=${CODE:-/data/zyd_workspace/bagel-LatentCoT/code/isochronic-state-flow-20261107}
PY=${PY:-/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python}
MODEL=${MODEL:-/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT}
SRC=${SRC:-/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b}
DATA=${DATA:-/data/zyd_workspace/isochronic_data_20261107}
LINEAR=${LINEAR:-/data/zyd_workspace/isochronic_data_20261106/linear/L_30k.json}
PROMPTS=${PROMPTS:-/data/zyd_workspace/isochronic_data_20261106/eval/dev64_prompts.jsonl}
CKPT=$1
OUT=$2
LIMIT_LINEAR=${3:-32}
LIMIT_PROMPT=${4:-64}
PORT_BASE=${PORT_BASE:-29611}

mkdir -p "$OUT"
cd "$CODE"

echo "== eval-linear =="
PYTHONPATH="$CODE" "$PY" -m torch.distributed.run --nproc_per_node=16 --master_port=$PORT_BASE \
  -m qwen_latent_cot.u15_target_correction.cli eval-linear \
  --distributed --device npu:0 --model "$MODEL" --source "$SRC" \
  --config "$CODE/configs/u15_isochronic_state/sample36.json" \
  --linear "$LINEAR" --checkpoint "$CKPT" --output "$OUT/linear" \
  --limit "$LIMIT_LINEAR" --seed 20261107

echo "== sample =="
PYTHONPATH="$CODE" "$PY" -m torch.distributed.run --nproc_per_node=16 --master_port=$((PORT_BASE+2)) \
  -m qwen_latent_cot.u15_target_correction.cli sample \
  --distributed --device npu:0 --model "$MODEL" --source "$SRC" \
  --config "$CODE/configs/u15_isochronic_state/sample36.json" \
  --checkpoint "$CKPT" --prompts "$PROMPTS" --output "$OUT/sample" --limit "$LIMIT_PROMPT"

echo "== done =="
ls "$OUT/linear/review.html" "$OUT/sample" | head -5
