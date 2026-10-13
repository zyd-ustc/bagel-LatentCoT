#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/shallow_memory_npu.sh fresh_output [extra arguments]}
shift
[[ ! -e "$RUN" ]] || { echo 'Output already exists' >&2; exit 1; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
unset CUDA_VISIBLE_DEVICES
cd "$ROOT"
exec "${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=16 \
  -m qwen_latent_cot.diagnostics.shallow_sampling \
  --model "${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}" \
  --checkpoint "${VISUAL_PARENT:-/root/outputs/visual_observe_full_20261012/best.pt}" \
  --text-checkpoint "${TEXT_TEACHER:-/root/outputs/c1_response_expand_20261012_141729/best.pt}" \
  --prompts "${PROMPTS:-$ROOT/data/prompts32.jsonl}" \
  --output "$RUN" --count 16 --passes 2 3 4 --early 10 20 --seed 0 "$@"
