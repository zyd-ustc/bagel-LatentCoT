#!/usr/bin/env bash
# Two resumed updates validate real H200 recovery, then continue original750->5000.
set -euo pipefail
reader_root=/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000_resume750
reader_code=/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-resume-20260930
reader_python=/private/software/conda/envs/lcot/bin/python
reader_data=/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/data
reader_parent=/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000/main
cd "$reader_code"
test -d "$reader_root"
test ! -e "$reader_root/main"
test ! -e "$reader_root/smoke"
export BAGEL_CODE_COMMIT=58cb0a62b04701ffe5f8ed46e36f562655702283
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

reader_check_idle() {
  "$reader_python" - <<'PY'
import subprocess
import torch
assert torch.cuda.device_count() == 8, 'eight visible CUDA devices required'
raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True)
rows = [tuple(map(int, row.split(','))) for row in raw.strip().splitlines()]
assert len(rows) == 8 and all(used < 1024 and free > 60000 and util < 10
                              for _, used, free, util in rows), raw
print('Eight idle H200 devices verified')
PY
}

reader_check_idle
nvidia-smi > "$reader_root/gpu_preflight.txt"
"$reader_python" -m pip freeze > "$reader_root/package_freeze.txt"
sha256sum "$reader_data/reader_train.jsonl" "$reader_data/reader_heldout.jsonl" \
  "$reader_parent/reader_warmup_step_0000750.safetensors" \
  "$reader_parent/reader_warmup_step_0000750.optimizer.pt" \
  qwen_latent_cot/bagel/reader_warmup.py scripts/train/bagel_memory_reader_warmup.py \
  configs/training/memory_reader_warmup.yaml > "$reader_root/input_sha256.txt"
reader_args=(--config configs/training/memory_reader_warmup.yaml
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT
  --prompt-data "$reader_data/reader_train.jsonl"
  --heldout-prompt-data "$reader_data/reader_heldout.jsonl"
  --resume-checkpoint "$reader_parent/reader_warmup_step_0000750.safetensors"
  --device cuda --eval-max-prompts 8)
CUDA_VISIBLE_DEVICES= WORLD_SIZE=8 "$reader_python" scripts/train/bagel_memory_reader_warmup.py \
  "${reader_args[@]}" --output-dir "$reader_root/main" --max-steps 5000 --validate-only \
  > "$reader_root/preflight.json"
printf '%q ' "$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_root/smoke" --max-steps 752 > "$reader_root/smoke_command.txt"
printf '\n' >> "$reader_root/smoke_command.txt"
echo '[resume] Starting two-update pilot:751/752, eight cards'
"$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_root/smoke" --max-steps 752 > "$reader_root/smoke.log" 2>&1

"$reader_python" - "$reader_root" "$reader_parent" <<'PY'
import json, math, sys
from pathlib import Path
import torch
root, parent = map(Path, sys.argv[1:])
smoke = root/'smoke'
status = json.loads((smoke/'status.json').read_text())
assert status['status'] == 'complete' and status['step'] == 752, status
rows = [json.loads(row) for row in (smoke/'metrics.jsonl').read_text().splitlines()]
assert [row['step'] for row in rows] == [751, 752]
assert all(row['world_size'] == 8 and len(row['prompt_ids']) == 8 for row in rows)
assert all(math.isfinite(row['loss_reader_mse']) and math.isfinite(row['grad_norm']) for row in rows)
initial = (smoke/'heldout_diagnostics.jsonl').read_text().splitlines()[0]
assert initial == (parent/'heldout_diagnostics.jsonl').read_text().splitlines()[0]
manifest = json.loads((smoke/'run_manifest.json').read_text())
assert manifest['start_step'] == 750 and manifest['effective_batch_size'] == 8
assert manifest['resume_mode'] == 'legacy_seeded_rollout'
gate = json.loads((smoke/'warmup_gate.json').read_text())
assert gate['native_parity_max_abs'] <= 1e-6
prefix = smoke/'reader_warmup_step_0000752'
for suffix in ('.safetensors','.json','.optimizer.pt','.resume.pt','.resume.json'):
    assert prefix.with_suffix(suffix).is_file(), suffix
state = torch.load(prefix.with_suffix('.optimizer.pt'), map_location='cpu', weights_only=True)
assert state['step'] == 752
assert all(float(value['step']) == 752 for value in state['optimizer']['state'].values())
summary = dict(status='passed', steps=[751,752], world_size=8,
    native_parity_max_abs=gate['native_parity_max_abs'], losses=[row['loss_reader_mse'] for row in rows],
    resume_mode=manifest['resume_mode'], sidecars_present=True)
(root/'smoke_verified.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
PY

reader_check_idle
printf '%q ' "$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_root/main" --max-steps 5000 > "$reader_root/command.txt"
printf '\n' >> "$reader_root/command.txt"
echo '[resume] Pilot passed; formal continuation from original750 to total5000'
"$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --output-dir "$reader_root/main" --max-steps 5000 > "$reader_root/train.log" 2>&1
echo '[resume] Training exited successfully; inspect main/status.json and checkpoint5000'
