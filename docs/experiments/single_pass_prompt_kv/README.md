# Single-pass prompt-KV visibility (hard 4)

This is a frozen BAGEL MoT inference ablation, not a trained-loop evaluation.
Each denoising timestep performs one full decoder forward. The eight native
SOI/EOI memory slots are reset to their frozen embedding at the next timestep.

| Path | Layers 0–11 | Layers 12–15 | Layers 16–19 | Layers 20+ |
| --- | --- | --- | --- | --- |
| Memory state | update | update | frozen | frozen |
| `keep`: non-memory → prompt KV | visible | visible | visible | visible |
| `mask_body`: non-memory → prompt KV | visible | masked | masked | visible |

The mask bounds come from `configs/training/loop_pair_memory_early.yaml`
(currently `[12,20)`). Memory queries can still read prompt KV at every layer;
GEN can read memory throughout. There is no second body pass, memory
replacement, prompt-aware memory initializer, or adapter checkpoint.

Both arms use the same four prompts, initial noise, image size, guidance, and
schedule. `run_manifest.json` records those settings and each prompt's seed;
`index.html` shows the side-by-side images and pixel MAE. The evaluator refuses
an already nonempty output directory.

```bash
cd /root/bagel-LatentCoT-v1
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -m pytest -q \
  tests/test_bagel_single_pass_prompt_kv_t2i.py tests/test_mot_loop_phase0.py
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -u \
  scripts/evaluate/bagel_single_pass_prompt_kv_t2i.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --benchmark-data experiments/data/geneval2_hard_16.jsonl \
  --output-dir /data/zyd_workspace/outputs/bagel_single_pass_prompt_kv_hard4 \
  --device npu:0 --max-prompts 4
```
