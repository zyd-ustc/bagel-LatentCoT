#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/shallow_memory_npu.sh fresh_output [sampling arguments]}
shift
[[ ! -e "$RUN" ]] || { echo 'Output already exists' >&2; exit 1; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
NPROC=${NPROC_PER_NODE:-16}
ARGS=(-m qwen_latent_cot.diagnostics.shallow_sampling
  --model "${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}"
  --checkpoint "${VISUAL_PARENT:-/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt}"
  --prompts "${PROMPTS:-$ROOT/data/prompts32.jsonl}"
  --output "$RUN" --count 16 --passes 2 3 4 --early 10 20 30 full --loop-layers full --seed 0 "$@")
"$PYTHON" "${ARGS[@]}" --prepare-only
if [[ "$NPROC" == 1 ]]; then
 "$PYTHON" "${ARGS[@]}"
else
 "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" "${ARGS[@]}"
 "$PYTHON" -m qwen_latent_cot.diagnostics.shallow_review "$RUN"
fi
