# Validation — 2026-09-27

Implementation verification only; not a model-training result.

## Executed locally

Interpreter: `/tmp/bagel-mechanism-tests.WOiF4o/bin/python` (isolated existing
PyTorch 2.8.0 environment). Original user Python installation was not modified.

`python -m pytest -q`: **255 passed in 3.72 seconds** (226 existing + 29 new).
`python -m compileall -q qwen_latent_cot/bagel scripts/train scripts/evaluate`: passed.
A reader and held-out eval `--validate-only`: passed with model-path `.` as a
path-existence-only fixture; this does NOT validate BAGEL model weights.
B/C/D `--help`: passed without loading weights or contacting reward services.

## Evidence covered

- Actual tiny BF16 MoT and Bagel: native adapter-off exact parity; cache/input
  immutability; strict Read mask; Write-only prompt mask; whole-sample shuffle.
- Stage A: GEN-Q LoRA gradients, no UND-Q/m0 gradients, detached Read state.
  Stage B: Read UND-Q gradients, frozen GEN reader, memory gradient retained.
- Full suffix outputs for W=1/2/3; final velocity is last output; intermediate
  velocities equal independently run shallow models; DS gradient propagation.
- Distillation teacher detach and acceptance guard; monotonic objective;
  normalized dependency losses, mask curriculum and reward composition.
- Toy orchestration using real loss/SDE math: native state reuse with shared
  batch timesteps, deterministic paired four-arm trajectories, 8-image HTML
  and metric output, metadata rejection, output protection and provenance.

One new test initially found that the distillation sum initializer retained a
zero-gradient graph edge to the final teacher. Removed that initializer; the
teacher now has no gradient edge from the distillation term. Final suite passed.

## Not verified here

Real H200/NPU memory use, model loading, kernel performance, optimizer dynamics,
reward-service availability, real dataset target-flow convergence and real
semantic/quality gains. GRPO orchestration and optional scorer integration have
not run against real services. No superiority or self-correction claim is made.
The old saved images/results were not changed or used as evidence for v2 gains.

## Recovery / next action

Source snapshot before edits:
`/Users/zyd/Documents/LCoT-codex/bagel-v2-backup.5LeFuK/source-before-v2.tar.gz`.
No files/results were deleted. Legacy hidden-space scripts remain as ablations.
No remote upload, training launch, commit or push was performed.
Next: review the Stage-A preflight command in README.md, then request transfer.
