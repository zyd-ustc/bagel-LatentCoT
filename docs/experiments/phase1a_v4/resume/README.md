# Phase 1A.0 continuation — prepared, not started

`--resume-checkpoint` restores adapter weights, AdamW moments/counters and global
step. `--max-steps 5000` means finish at total step 5000, not run 5000 more updates.
The validated old checkpoint is step 750: first resumed update will be 751.
Old 751–755 were not checkpointed and must be recomputed. Never overwrite the
old run directory or truncate its historical logs/metrics.

## Guards and compatibility

Require the same model path, data SHA256, seed, world_size/global batch, reader
contract, native rollout sampling/resolution, learning rate/clipping and eval/save
settings. Restore the original step-zero heldout baseline; do not evaluate the
trained reader as a new initial model. An incomplete or invalid optimizer is fatal,
not a weights-only fallback. `--validate-only` performs CPU artifact/provenance
checks without loading BAGEL, allocating CUDA or creating a training output.

Existing three-file checkpoints, including the real step-750 artifact, can recover
with their original `resolved_config.json`, `run_manifest.json`,
`trainable_routes.json` and `heldout_diagnostics.jsonl` still adjacent. They are
marked `legacy_seeded_rollout`: explicit per-step/rank rollout seeds and optimizer
are restored, but old global RNG was never saved and cannot be reconstructed.
Do not claim exact global-RNG or bitwise H200 reproducibility for this legacy path.

New checkpoints add `.resume.pt` (per-rank CPU/CUDA/Python/NumPy RNG after eval)
and `.resume.json` (completion marker, world/order and content hashes). New-format
checkpoints require both files; a missing completion marker is fatal. Original
initial-baseline hash and immutable provenance files are checked. CPU tiny MoT
tests cover uninterrupted/resumed bitwise equality, including stochastic losses,
two-rank mean-gradient optimizer state and disjoint rank sample continuation.

## Safe preflight only (does not start training)

Use the prepared code directory on the shared H200 filesystem. This command is
safe even while CUDA cards are busy because it hides CUDA and returns before
distributed/model initialization:

```bash
cd /private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-resume-20260930
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" \
  /private/software/conda/envs/lcot/bin/python scripts/train/bagel_memory_reader_warmup.py \
  --config configs/training/memory_reader_warmup.yaml \
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT \
  --prompt-data /private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/data/reader_train.jsonl \
  --heldout-prompt-data /private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/data/reader_heldout.jsonl \
  --resume-checkpoint /private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000/main/reader_warmup_step_0000750.safetensors \
  --output-dir /private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000_resume750/main \
  --max-steps 5000 --eval-max-prompts 8 --device cuda --validate-only
```

Expected resume summary: step=750, world_size=8, mode=legacy_seeded_rollout.
Formal continuation remains deferred until a separate user instruction and eight
usable H200 cards. No resumed BAGEL training or GPU pilot has been launched.
