#!/usr/bin/env bash
# CUDA/H200 normal R2/4/6/8. Hard8 uses at most 4 workers (2 prompts/worker).
set -euo pipefail
cd "$(dirname "$0")/../.."

MECHANISM_OUT=${1:?usage: bash scripts/evaluate/run_bagel_memory_mechanism.sh /absolute/path/NEW_RUN}
PYTHON_BIN=${PYTHON_BIN:-python}
BACKEND=${BACKEND:-cuda}
MODEL_PATH=${MODEL_PATH:-$PWD/models/Bagel-7B-MoT}
BENCHMARK_DATA=${BENCHMARK_DATA:-experiments/data/geneval2_hard_128.jsonl}
MAX_PROMPTS=${MAX_PROMPTS:-8}
NUM_STEPS=${NUM_STEPS:-50}
SEEDS=${SEEDS:-42}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-512}

if [[ -e "$MECHANISM_OUT" ]]; then
  printf 'Output already exists; choose a new run directory: %s\n' "$MECHANISM_OUT" >&2
  exit 2
fi
if [[ "$BACKEND" != cuda && "$BACKEND" != npu ]]; then
  printf 'BACKEND must be cuda or npu\n' >&2
  exit 2
fi
if [[ ! "$MAX_PROMPTS" =~ ^[1-9][0-9]*$ ]] || (( MAX_PROMPTS < 2 || MAX_PROMPTS % 2 )); then
  printf 'MAX_PROMPTS must be a positive even integer >= 2\n' >&2
  exit 2
fi
# Keep inherited visibility masks (e.g. SLURM/container allocation), never select
# GPUs outside them. Each worker gets exactly one of those original tokens.
device_lines=$("$PYTHON_BIN" scripts/evaluate/mechanism_runtime.py --backend "$BACKEND")
devices=()
while IFS= read -r token; do devices+=("$token"); done <<<"$device_lines"
pair_count=$((MAX_PROMPTS / 2))
if [[ -z "${NUM_SHARDS:-}" ]]; then
  NUM_SHARDS=${#devices[@]}
  if (( NUM_SHARDS > pair_count )); then NUM_SHARDS=$pair_count; fi
fi
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || (( NUM_SHARDS > pair_count || NUM_SHARDS > ${#devices[@]} )); then
  printf 'NUM_SHARDS must be <= prompt pairs (%s) and visible devices (%s)\n' "$pair_count" "${#devices[@]}" >&2
  exit 2
fi
if [[ "$BACKEND" == cuda ]]; then visibility_var=CUDA_VISIBLE_DEVICES; else visibility_var=ASCEND_RT_VISIBLE_DEVICES; fi
args=(--model-path "$MODEL_PATH" --output-dir "$MECHANISM_OUT"
      --benchmark-data "$BENCHMARK_DATA" --max-prompts "$MAX_PROMPTS"
      --num-shards "$NUM_SHARDS" --num-steps "$NUM_STEPS" --seeds "$SEEDS"
      --height "$HEIGHT" --width "$WIDTH")
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/evaluate/bagel_memory_mechanism.py "${args[@]}" --dry-run >/dev/null
mkdir -p "$MECHANISM_OUT"
"$PYTHON_BIN" scripts/evaluate/bagel_memory_mechanism.py "${args[@]}" --dry-run >"$MECHANISM_OUT/launch_plan.json"

pids=()
stop_workers() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait || true
  exit 130
}
trap stop_workers INT TERM
for ((i=0; i<NUM_SHARDS; i++)); do
  printf '[launch] pair shard=%s %s=%s\n' "$i" "$BACKEND" "${devices[$i]}"
  env "$visibility_var=${devices[$i]}" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u scripts/evaluate/bagel_memory_mechanism.py "${args[@]}" \
      --shard-id "$i" --device "$BACKEND:0" >"$MECHANISM_OUT/shard_${i}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then
  printf 'At least one shard failed; no complete gallery was produced. Inspect shard logs.\n' >&2
  exit 1
fi
"$PYTHON_BIN" scripts/evaluate/bagel_memory_mechanism.py "${args[@]}" --merge-only
printf '[done] %s/index.html\n' "$MECHANISM_OUT"
