#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: bash scripts/explicit_feedback_npu.sh fresh_output [extra inference arguments]}
shift
[[ ! -e "$RUN" ]] || { echo 'Output already exists' >&2; exit 1; }
PYTHON=${MODEL_PYTHON:-/root/venvs/bagel-NPU/bin/python}
MODEL=${MODEL_PATH:-/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT}
CHECKPOINT=${TEXT_TEACHER:-/root/outputs/c1_response_expand_20261012_141729/best.pt}
PROMPT_FILE=${PROMPTS:-$ROOT/data/prompts32.jsonl}
mkdir -p "$RUN/inputs"
cd "$ROOT"
"$PYTHON" - "$PROMPT_FILE" "$RUN/inputs" "${SELECTION_SEED:-7456276010569376034}" "${NOISE_SEED:-0}" <<'PY'
import hashlib, json, random, sys
from pathlib import Path
source, output, selection_seed, noise_seed = sys.argv[1:]
output = Path(output)
pool = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
if len(pool) < 16:
    raise ValueError('require at least 16 prompts for 16 workers')
indexes = random.Random(int(selection_seed)).sample(range(len(pool)), 16)
rows = [dict(sample_id=pool[i]['prompt_id'], prompt=pool[i]['prompt'],
             noise_seed=int(noise_seed), source_index=i) for i in indexes]
for rank, row in enumerate(rows):
    (output/f'device_{rank:02d}.jsonl').write_text(json.dumps(row, ensure_ascii=False)+'\n')
(output/'feedback16.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(output/'selection.json').write_text(json.dumps(dict(selection_seed=int(selection_seed),
    noise_seed=int(noise_seed), source=source, source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
    rows=rows), ensure_ascii=False, indent=2)+'\n')
print(f'Selected {len(rows)} prompts: {output}', flush=True)
PY
pids=()
for i in $(seq 0 15); do
  shard=$(printf '%s/inputs/device_%02d.jsonl' "$RUN" "$i")
  log=$(printf '%s/device_%02d.log' "$RUN" "$i")
  work=$(printf '%s/device_%02d' "$RUN" "$i")
  env -u CUDA_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES="$i" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m qwen_latent_cot.diagnostics.feedback_cli \
      --model "$MODEL" --text-checkpoint "$CHECKPOINT" \
      --config "$ROOT/configs/explicit_feedback.json" --data "$shard" \
      --device npu:0 --output "$work" --auto-reflect --early 10 20 "$@" > "$log" 2>&1 &
  pids+=("$!")
done
printf 'Launched 16 workers: %s\n' "$RUN"
failed=0
for i in $(seq 0 15); do
  if ! wait "${pids[$i]}"; then
    printf 'Worker %02d failed; inspect %s/device_%02d.log\n' "$i" "$RUN" "$i" >&2
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit 1
"$PYTHON" - "$RUN" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>显式固定 memory：16 卡结果</title>',
         '<h1>显式固定 memory：16 卡结果</h1><ul>']
for i in range(16):
    page = root/f'device_{i:02d}'/'review.html'
    if not page.exists():
        raise FileNotFoundError(page)
    parts.append(f'<li><a href="device_{i:02d}/review.html">device_{i:02d}</a></li>')
parts.append('</ul></html>')
(root/'index.html').write_text('\n'.join(parts))
print(f'Completed: {root}/index.html', flush=True)
PY
