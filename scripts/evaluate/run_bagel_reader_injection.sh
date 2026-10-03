#!/usr/bin/env bash
# Independent frozen-model inference workers; no distributed training or waiting.
set -euo pipefail
cd "$(dirname "$0")/../.."
reader_output=${1:?Usage: bash scripts/evaluate/run_bagel_reader_injection.sh NEW_OUTPUT_DIR}
PYTHON_BIN=${PYTHON_BIN:-python}
READER_CHECKPOINT=${READER_CHECKPOINT:?Set READER_CHECKPOINT to the step-5000 safetensors}
BENCHMARK_DATA=${BENCHMARK_DATA:-experiments/data/phase1a_semantic_hard64.jsonl}
GATE_SCALE=${GATE_SCALE:-1.0}
MAX_PROMPTS=${MAX_PROMPTS:-16}
NUM_SHARDS=${NUM_SHARDS:-8}
MIN_FREE_GIB=${MIN_FREE_GIB:-45}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
if [[ -e "$reader_output" ]]; then
  echo "Refusing to overwrite output: $reader_output" >&2; exit 2
fi
args=(--checkpoint "$READER_CHECKPOINT" --benchmark-data "$BENCHMARK_DATA"
      --output-dir "$reader_output" --gate-scale "$GATE_SCALE"
      --max-prompts "$MAX_PROMPTS" --num-shards "$NUM_SHARDS")
if [[ -n "${MODEL_PATH:-}" ]]; then args+=(--model-path "$MODEL_PATH"); fi
if [[ -n "${SEED:-}" ]]; then args+=(--seed "$SEED"); fi
# CPU preflight before touching GPU contexts or creating output.
"$PYTHON_BIN" scripts/evaluate/bagel_reader_injection.py "${args[@]}" --dry-run >/dev/null
reader_device_lines=$("$PYTHON_BIN" scripts/evaluate/mechanism_runtime.py --backend cuda)
reader_devices=()
while IFS= read -r reader_token; do reader_devices+=("$reader_token"); done <<<"$reader_device_lines"
if (( NUM_SHARDS > ${#reader_devices[@]} )); then
  echo "Need $NUM_SHARDS visible GPUs; found ${#reader_devices[@]}" >&2; exit 2
fi
"$PYTHON_BIN" - "$NUM_SHARDS" "$MIN_FREE_GIB" <<'PY'
import math, sys, torch
count, minimum = int(sys.argv[1]), float(sys.argv[2])
if not math.isfinite(minimum) or minimum <= 0:
    raise ValueError('MIN_FREE_GIB must be finite and positive')
for index in range(count):
    free, total = torch.cuda.mem_get_info(index)
    print(f'visible GPU {index}: {free / 2**30:.1f} GiB free', flush=True)
    if free < minimum * 2**30:
        raise RuntimeError(f'GPU {index} has less than {minimum} GiB free; retry later')
PY
mkdir -p "$reader_output"
"$PYTHON_BIN" scripts/evaluate/bagel_reader_injection.py "${args[@]}" --dry-run >"$reader_output/launch_plan.json"
nvidia-smi >"$reader_output/gpu_preflight.txt"
printf '%q ' "$PYTHON_BIN" scripts/evaluate/bagel_reader_injection.py "${args[@]}" >"$reader_output/command.txt"
printf '\n' >>"$reader_output/command.txt"
reader_pids=()
reader_stop() {
  for reader_pid in "${reader_pids[@]}"; do kill -TERM "$reader_pid" 2>/dev/null || true; done
  wait || true
  exit 130
}
trap reader_stop INT TERM
for ((reader_shard=0; reader_shard<NUM_SHARDS; reader_shard++)); do
  echo "[launch] shard=$reader_shard CUDA_VISIBLE_DEVICES=${reader_devices[$reader_shard]}"
  CUDA_VISIBLE_DEVICES="${reader_devices[$reader_shard]}" "$PYTHON_BIN" -u \
    scripts/evaluate/bagel_reader_injection.py "${args[@]}" --shard-id "$reader_shard" \
    >"$reader_output/shard_${reader_shard}.log" 2>&1 &
  reader_pids+=("$!")
done
reader_failed=0
for reader_pid in "${reader_pids[@]}"; do if ! wait "$reader_pid"; then reader_failed=1; fi; done
if (( reader_failed )); then
  echo "Worker failed; inspect shard logs. No completed gallery was produced." >&2; exit 1
fi
"$PYTHON_BIN" scripts/evaluate/bagel_reader_injection.py "${args[@]}" --merge-only
echo "[done] $reader_output/index.html"
