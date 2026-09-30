# Phase 1A.0 on H200 20470

## Preparation

2026-09-30: Authorized GitHub push/upload/8-card Phase 1A.0 training. Code
`bb5fc09d5b6bba79c1304f7372408b63989cfa8c` pushed to source and dedicated run branches.
Local and remote CPU suites each passed 305 tests (remote 130.39 s).
Isolated code: `/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-8gpu-20260930`.
Python: `/private/software/conda/envs/lcot/bin/python`, torch 2.5.1+cu124.
Base: `/private/yida_workspace/models/BAGEL-7B-MoT`.

## Data export

24,547 training prompts (21,004 spatial, 3,543 count), 64 heldout (52 spatial,
12 count). Heuristic tags with matched phrase provenance; not human-reviewed.
22 normalized train/official-val overlaps excluded from training. No images
consumed and no original manifests changed.
Export report and SHA256 live under the run root's `data/export_report.json`.

## Smoke attempt 1 — failed before training

Root: `/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322`.
8 ranks reached real model loading and initial native rollout. At 05:34:20 UTC,
rank 0 failed with `ValueError: expected [B,L,D] hidden and positive num_slots`.
The packed text-cache capture returned [L,D], but inferencer saved it unchanged
while providing [1,L] masks to the initializer. Synthetic runtime tests used
already batched hidden and had not exercised this real capture boundary.
Torchrun terminated only its workers; launcher exited and did not start main.
Failure log/status preserved. No optimizer steps, no smoke success claim.

Fix: add the singleton batch axis at the inferencer's one-prompt capture
boundary, with row-count validation; add an actual tiny-MoT text-cache capture
regression. Do not loosen the initializer, masks, parity or heldout guards.
Retry uses a fresh output root, unchanged data hashes and numerical settings.

See PLAN.md for protocol and CHECKLIST.md for execution frontier.
