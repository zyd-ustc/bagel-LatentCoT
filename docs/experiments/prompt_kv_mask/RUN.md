# NPU commands: 16 prompts × keep/mask prompt KV × four Write arms

The user runs these commands after the code is on GitHub main. This experiment
has **not** been run yet. It is frozen/training-free, K=8, R=2, same-depth body
from `configs/training/loop_pair_memory_early.yaml`, 50 steps at 512×512 by
default. The two KV policies share the same packed prompt pair, prompt cache,
initial noise, model weights, CFG and schedule. The mask blocks cached prompt
keys only for non-memory conditional generation queries in every layer; memory
queries still see the cache. The unconditioned CFG branch is unchanged.

## 1. Enter the existing NPU checkout

Connect from the Mac:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -i /Users/zyd/Downloads/KeyPair-zyd.pem root@dev.modelarts.cnszaismartcity01.api-ai.smartcitysz.com -p 32692
```

On NPU, inspect local edits before updating; do not discard any:

```bash
git -C /root/bagel-LatentCoT-phase1-calibrated status --short
git -C /root/bagel-LatentCoT-phase1-calibrated fetch origin main
git -C /root/bagel-LatentCoT-phase1-calibrated switch --detach origin/main
cd /root/bagel-LatentCoT-phase1-calibrated
```

If `switch` refuses because local edits overlap, stop and use a new checkout;
do not reset the existing tree.

## 2. Two-prompt pilot

```bash
set -euo pipefail
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -m pytest -q \
  tests/test_bagel_write_sensitivity_t2i.py tests/test_mot_loop_phase0.py

PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -u \
  scripts/evaluate/bagel_write_sensitivity_t2i.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --output-dir /data/outputs/bagel_prompt_kv_mask_hard16_v3_smoke \
  --device npu:0 --max-prompts 2 --num-steps 2 \
  2>&1 | tee /data/outputs/bagel_prompt_kv_mask_hard16_v3_smoke.log
```

Gate: `run_manifest.json` has `complete=true`, both `keep` and
`mask_nonmemory` policies, 4 arms per policy, 16 PNGs, 8 probe files, and
`index.html`. Do not treat pilot image differences as semantic evidence.

## 3. Full hard-16 comparison

Only after the pilot passes, use a new output path:

```bash
set -euo pipefail
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -u \
  scripts/evaluate/bagel_write_sensitivity_t2i.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --output-dir /data/outputs/bagel_prompt_kv_mask_hard16_v3 \
  --device npu:0 --max-prompts 16 \
  2>&1 | tee /data/outputs/bagel_prompt_kv_mask_hard16_v3.log
```

Expected: 128 PNGs, 8 columns in `index.html`, per-prompt keep-vs-mask MAE
for each Write arm, and 8 GenEval2 image maps under `geneval2/`. The manifest
uses schema `bagel_prompt_memory_kv_visibility_write_hard16_v3` and records
prompt/noise hashes, policies, probes, config and code commit. Pixel MAE is
only a sensitivity measure; inspect count/relation/attribute/composition in
the gallery before making a quality claim.
