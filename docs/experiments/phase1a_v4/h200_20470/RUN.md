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

## Smoke attempt 2 — complete; main training launched

Training code: `4148a70bfd318d506a2b261ca9aaa56b8de31075`, pushed to both
`codex/phase1a-v4-reader-warmup` and `codex/run/phase1a0-h2008-20260930`.
Local full suite: 306 passed. Remote targeted native capture and two-rank
tests: 10 passed in 49.60 s with CUDA disabled for these CPU tests.
Code directory: `/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-8gpu-20260930-r2`.
Root: `/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2`.
Detached launcher PID on 20470: 411045. No tmux dependency and SSH can disconnect.
Data hashes and all numerical settings match attempt 1.

8-card smoke completed 2 updates; losses 6.6853085160, 6.4241779447, final
pre-clipping gradient norm 0.0363959558; every recorded loss/gradient finite.
Native parity max abs=0. Four existing operational checks passed on only two
heldout prompts. Correct MSE 8.7343239784 vs initial 8.7352927923 and shuffled
8.8187544346. Zero MSE 2.6766021252 is substantially lower than correct;
this is a diagnostic caveat, not evidence of useful semantic memory. The
machine readiness flag is not a robust/semantic go decision from this tiny smoke.

Fresh 5000-step main run started automatically after smoke validation. At
2026-09-30 11:39 UTC the artifact confirms 183 optimizer steps, world_size=8,
per-rank B=1, global B=8, generation_injection=false. Latest loss=5.0966878682,
latest pre-clipping gradient norm=1.8724541664, last-10 mean loss=4.8418878764;
all 183 recorded losses/gradient norms finite. Different training prompts mean
these raw losses are not a controlled improvement estimate. First formal
heldout/checkpoint at step 250 remains pending. Main status is running, not complete.

## Current resource audit

All three endpoints share the same code/data filesystem device and inode;
inferencer SHA256 also matches. At 11:39 UTC, 20344 and 20474 each had eight
idle H200 with 143167 MiB free per card and no compute processes. 20470 had
our main worker (~33 GiB/card) plus unrelated jobs, with ~25.8–26.3 GiB free.
No migration or extra training copy has been started. Current runner has no
exact resume CLI; moving must not silently discard the completed updates.
See [SSH_PORT_AUDIT.md](SSH_PORT_AUDIT.md) for per-card memory snapshot.
