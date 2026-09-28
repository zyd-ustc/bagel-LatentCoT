#!/usr/bin/env bash
# Foreground launcher: run inside tmux; never starts tmux or detaches itself.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/private/yida_workspace/models/BAGEL-7B-MoT}"
DATA_PATH="${DATA_PATH:-/private/yida_workspace/datasets/memory_grounding_v2/stage12_stage2_prompt_unique_20260927/prompts.jsonl}"
CONFIG="${CONFIG:-configs/training/memory_reader_grounding_8gpu.yaml}"
OUTDIR="${OUTDIR:-/private/yida_workspace/outputs/memory_reader_v2_8gpu_$(date +%Y%m%d_%H%M%S)}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUTDIR" || -e "${OUTDIR}.launcher.log" || -e "${OUTDIR}.workers" ]]; then
    echo "Refusing to overwrite an existing run: $OUTDIR" >&2
    exit 1
fi
"$PYTHON_BIN" - "$NUM_PROCESSES" "$BATCH_SIZE" <<'PY'
import sys, torch
world, batch = map(int, sys.argv[1:])
count = torch.cuda.device_count()
if world < 1 or world > count or batch < 2:
    raise SystemExit(f"requested {world} GPUs / batch {batch}; visible CUDA GPUs={count}; batch must be >=2")
print(f"Visible GPUs={count}; ranks={world}; local batch={batch}; global batch={world*batch}", flush=True)
PY

# Validate the full data once before launching eight heavy model replicas.
WORLD_SIZE="$NUM_PROCESSES" RANK=0 LOCAL_RANK=0 "$PYTHON_BIN" \
    scripts/train/bagel_gen_memory_grounding.py --config "$CONFIG" \
    --model-path "$MODEL_PATH" --data-path "$DATA_PATH" --output-dir "$OUTDIR" \
    --max-steps "$MAX_STEPS" --batch-size "$BATCH_SIZE" --validate-only "$@"

mkdir -p -- "$(dirname -- "$OUTDIR")"
echo "Output: $OUTDIR"
echo "Launcher log: ${OUTDIR}.launcher.log"
echo "Rank stdout/stderr: ${OUTDIR}.workers"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$NUM_PROCESSES" --max_restarts=0 \
    --log_dir="${OUTDIR}.workers" --tee=3 \
    scripts/train/bagel_gen_memory_grounding.py --config "$CONFIG" \
    --model-path "$MODEL_PATH" --data-path "$DATA_PATH" --output-dir "$OUTDIR" \
    --max-steps "$MAX_STEPS" --batch-size "$BATCH_SIZE" "$@" \
    2>&1 | tee "${OUTDIR}.launcher.log"
