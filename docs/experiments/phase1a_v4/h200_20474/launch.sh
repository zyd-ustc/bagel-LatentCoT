#!/usr/bin/env bash
# Direct fresh restart: reuse validated shared exports and passed smoke evidence.
set -euo pipefail
reader_root=/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000
reader_code=/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-8gpu-20260930-r2
reader_python=/private/software/conda/envs/lcot/bin/python
reader_data=/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/data
cd "$reader_code"
test -d "$reader_root"
test ! -e "$reader_root/main"
test ! -e "$reader_root/train.pid"
export BAGEL_CODE_COMMIT=4148a70bfd318d506a2b261ca9aaa56b8de31075
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$reader_python" -c 'import torch; assert torch.cuda.device_count()==8; print(torch.__version__, "8 CUDA devices verified")'
nvidia-smi > "$reader_root/gpu_preflight.txt"
"$reader_python" -m pip freeze > "$reader_root/package_freeze.txt"
sha256sum "$reader_data/reader_train.jsonl" "$reader_data/reader_heldout.jsonl" \
  qwen_latent_cot/bagel/inferencer.py configs/training/memory_reader_warmup.yaml \
  > "$reader_root/input_sha256.txt"
cp configs/training/memory_reader_warmup.yaml "$reader_root/config_source.yaml"
reader_args=(--config configs/training/memory_reader_warmup.yaml
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT
  --prompt-data "$reader_data/reader_train.jsonl"
  --heldout-prompt-data "$reader_data/reader_heldout.jsonl"
  --output-dir "$reader_root/main" --device cuda --max-steps 5000 --eval-max-prompts 8)
"$reader_python" scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  --validate-only > "$reader_root/preflight.json"
printf '%q ' "$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" > "$reader_root/command.txt"
printf '\n' >> "$reader_root/command.txt"
nohup "$reader_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/bagel_memory_reader_warmup.py "${reader_args[@]}" \
  > "$reader_root/train.log" 2>&1 < /dev/null &
reader_pid=$!
printf '%s\n' "$reader_pid" > "$reader_root/train.pid"
printf 'TRAIN_PID=%s\nOUTDIR=%s\n' "$reader_pid" "$reader_root"
