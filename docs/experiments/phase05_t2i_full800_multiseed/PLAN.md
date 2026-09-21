# Phase 0.5 full-800 multi-seed plan

## Contract

- Dataset: the official 800-prompt GenEval2 set at commit
  `a6e82d2289e8d418f27f0adee77908b07060eea3`.
- Main comparison: seed 42 with `Z0,Z2,Z3,Z4,Z6` under identical prompt,
  noise derivation, image geometry, CFG, NFE, and timestep schedule.
- Randomness control: Z0-only repeats at seeds 43 and 44.
- Generation: frozen BAGEL-7B-MoT, 1024×1024, 50 steps, 16 NPU shards.
- Evaluation: GenEval2 Soft-TIFA with Qwen3-VL-8B-Instruct.

## Success criteria

1. Produce 5,600 images with complete manifests and no missing prompt-arm pairs.
2. Score all generated images and write per-run AM, GM, skills, and atomicity.
3. Report the three-seed Z0 mean, sample standard deviation, and range.
4. Compare loop arms only against same-seed Z0; use extra Z0 seeds only as a
   baseline-variability diagnostic.

## Outputs

The root output is `/data/outputs/bagel_loop_t2i_full800_multiseed` with
`seed_42_main`, `seed_43_z0`, `seed_44_z0`, and root-level
`multiseed_summary.{json,md}`.

## Estimated runtime and storage

- Generation and scoring: 6–8 hours on 16 Ascend NPUs.
- Images and reports: approximately 7 GB.
