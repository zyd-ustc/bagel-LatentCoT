# BAGEL native-head MoT internal editing

Training is8000 total optimizer steps: observation caption alignment1000, then two-round editing7000. Native BAGEL stays frozen. Three role-local Q/K/V/O LoRA sets are saved; native lm_head and llm2vae remain the only output heads.

## Data and run

- [Design](../bagel_mot_internal_edit_loop_plan_20261023.md)
- [Fixed recipe](PLAN.md), [execution record](RUN.md), [checklist](CHECKLIST.md)
- Config: `configs/mot_edit_8000.json`
- Code: `qwen_latent_cot/mot_edit/`
- New entry point: `python -m qwen_latent_cot.mot_edit.cli`

On modelarts-64, use the existing bagel-NPU Python. The pinned code/run/corpus paths are in PLAN.md. Read progress with:

```bash
/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python \
  /data/zyd_workspace/bagel-LatentCoT/code/bagel-mot-edit-8000-20261023/scripts/status_mot_edit.py \
  --run /data/zyd_workspace/bagel-LatentCoT/outputs/bagel_mot_edit_64npu_20261023_8000
```

Checkpoint8000 belongs to this new method, not the old spatial-correction CLI. Sample it with the new CLI's `sample --checkpoint ... --prompt ... --output ... --model ... --rounds 2`. `--rounds 0` runs the native baseline; `--rounds 1` truncates the same trained loop after one edit at each active node.

Exact resume uses `train --resume ...` with the same data/config/world size. Do not launch a second worker group over an active run. The checkpoint includes both stages' adapters, active optimizer state, deterministic sampler and per-rank RNG.
