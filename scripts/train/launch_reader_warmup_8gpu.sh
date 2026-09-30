#!/usr/bin/env bash
# Fresh prompt export -> validated 8-card smoke -> fresh Phase 1A.0 main run.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "Usage: bash $0 SOURCE_JSONL OFFICIAL_VAL_JSONL MODEL_PATH RUN_ROOT PYTHON_BIN" >&2
  exit 2
fi
reader_source=$1
reader_heldout_source=$2
reader_model=$3
reader_run_root=$4
reader_python=$5
if [[ -e "$reader_run_root" ]]; then
  echo "Refusing to overwrite run root: $reader_run_root" >&2
  exit 2
fi
mkdir -p "$reader_run_root"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$reader_python" -c 'import torch; assert torch.cuda.device_count()==8; print("8 CUDA devices verified")'
nvidia-smi > "$reader_run_root/gpu_preflight.txt"
"$reader_python" -m pip freeze > "$reader_run_root/package_freeze.txt"
"$reader_python" scripts/data/prepare_reader_warmup_prompts.py \
  --source "$reader_source" --heldout-source "$reader_heldout_source" \
  --output-dir "$reader_run_root/data" --heldout-count 64 \
  | tee "$reader_run_root/data_export.log"
reader_args=(--config configs/training/memory_reader_warmup.yaml
  --model-path "$reader_model"
  --prompt-data "$reader_run_root/data/reader_train.jsonl"
  --heldout-prompt-data "$reader_run_root/data/reader_heldout.jsonl"
  --device cuda)
"$reader_python" scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_run_root/main" --validate-only \
  | tee "$reader_run_root/preflight.json"
echo "[phase1a0] Starting 8-card, 2-step smoke"
"$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_run_root/smoke" --max-steps 2 --eval-max-prompts 2 \
  2>&1 | tee "$reader_run_root/smoke.log"
"$reader_python" - "$reader_run_root/smoke" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
status=json.loads((root/'status.json').read_text())
assert status['status']=='complete' and status['step']==2,status
rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
assert len(rows)==2 and all(row['world_size']==8 for row in rows)
assert all(math.isfinite(row['loss_reader_mse']) and math.isfinite(row['grad_norm']) for row in rows)
report=json.loads((root/'warmup_gate.json').read_text())
assert report['native_parity_max_abs']<=1e-6
assert (root/'reader_warmup_step_0000002.safetensors').is_file()
print('[phase1a0] Smoke passed; starting fresh 5000-step main training')
PY
"$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_run_root/main" --max-steps 5000 --eval-max-prompts 8 \
  2>&1 | tee "$reader_run_root/train.log"
echo "[phase1a0] Main training completed. Inspect main/warmup_gate.json before OPD."
