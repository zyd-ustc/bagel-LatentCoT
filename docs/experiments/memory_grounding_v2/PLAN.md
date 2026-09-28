# Memory grounding v2 implementation contract

## Objective and constraints
Implement the supplied Memory Grounding → GEN Reader → Semantic RL → Loop
Supervision plan locally in v1. No remote upload, GPU run, or deletion of results.
K8, body [12,20), fresh memory, strict one Read, same-depth Write recurrence.
Old hidden-space grounding is a legacy ablation, not a prerequisite.

## Baseline and comparability
Native means K0, adapters off, original `_forward_flow`. Counterfactuals share
prompt, x_t, t, initial noise and schedule; only first-Write memory differs.
Shuffle swaps whole samples with no fixed point; zero is dynamic after injection.
Metrics: teacher error, shuffle/zero dependency gaps, direction cosine, relative
velocity, attention mass; per-Write full suffix velocities for supervision.
No semantic improvement or self-correction claim without held-out evidence.

## Code translation
Decoder: explicit Write-entry override and Write-only prompt mask; per-Write
suffix outputs. Bagel: typed supervised output. New math/runtime and A/B/C/D
entrypoints/configs plus four-arm evaluation. Legacy entrypoints stay readable.
Stage A trains GEN-Q only; B trains memory-row UND-Q with frozen reader;
C adds paired causal rewards; D uses DS and optional gated distillation.
SMA is future work, not an implementation claim.

## Execution and recovery
CPU tiny-model numeric/gradient tests, CLI preflight, then full pytest suite.
No real main run in this request; GPU pilot and stage acceptance remain pending.
Stop on parity, cache mutation, gradient routing or non-finite failures.
Backup: `/Users/zyd/Documents/LCoT-codex/bagel-v2-backup.5LeFuK/source-before-v2.tar.gz`.
Git pointer references an unavailable Linux worktree; do not repair or claim commits.
Experiment skill's bash_exec/artifact/memory tools are unavailable; use local
terminal plus this plan/checklist and a durable validation record instead.

## Revision log
2026-09-27: initial implementation contract; reference plan read completely.
2026-09-27: v2 explicitly locks single-device conditional velocity (CFG=1);
historical normal-R CFG=4 remains a separate protocol. No DDP or exact optimizer
resume claim. Optional real semantic/quality scorers are available for held-out
four-arm evaluation; SMA/gating remains future work as specified by the plan.
2026-09-27: CPU validation complete (255 tests); no real training was launched.

## Checklist
See CHECKLIST.md and VALIDATION.md. Next: remote Stage-A preflight, then a 2-step
GPU pilot in a fresh output directory after code transfer is requested.
