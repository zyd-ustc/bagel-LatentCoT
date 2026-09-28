# Extra-Loop Prompt-KV Ablation (v4)

This is the training-free hard-16 Write-sensitivity comparison. The main
gallery contains `keep` and `mask_loop_only`, each with four Write sources:
`correct_M`, `shuffled_across_sample_M`, `prompt_init_M` (the prompt-aware
`M_init(P)`, formerly named `m0`), and `zero_M`.

| Stage | `keep` | `mask_loop_only` |
|---|---|---|
| Prompt prefill, prefix, first body/Read pass, suffix | prompt KV visible | prompt KV visible |
| Second body/Write pass, non-memory query | prompt KV visible | cached prompt KV blocked |
| Second body/Write pass, memory query | prompt KV visible | prompt KV visible |

Both policies share the same paired prompts, prompt cache, initial noise,
frozen weights, K=8, R=2, body position from the training config, strict Read,
fresh memory, CFG, and denoising schedule. Only the conditional image branch
receives this intervention; the unconditioned CFG branch stays at `none`.
`all_generation` is available solely as an optional legacy destructive control
via `--include-all-generation-control`; it is excluded from the main gallery by
default. Outputs use a new v4 schema and must not be pooled with v3.

The experiment asks whether the *extra recurrent Write pass* needs to re-read
cached prompt keys directly. It is not a strict information bottleneck: the
Write pass begins from a prompt-conditioned prefix state. Pixel MAE is a
sensitivity measure, not a semantic-quality score.

## NPU verification and run

From the NPU checkout after transferring these code changes, first run:

```bash
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -m pytest -q \
  tests/test_bagel_write_sensitivity_t2i.py tests/test_mot_loop_phase0.py
```

Then use a fresh pilot output directory:

```bash
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -u \
  scripts/evaluate/bagel_write_sensitivity_t2i.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --output-dir /data/outputs/bagel_extra_loop_prompt_kv_v4_smoke \
  --device npu:0 --max-prompts 2 --num-steps 2
```

Gate: `run_manifest.json` must have `complete=true`, two policies × four arms,
16 PNGs, eight probe files, and `index.html`. Only then run the full 16 prompts
in another fresh directory, with `--max-prompts 16` and default 50 steps.
Do not overwrite or merge prior v3 results.
