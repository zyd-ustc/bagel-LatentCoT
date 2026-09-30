# Reader warm-up resume implementation (no real training launch)

## Objective and constraints

User requests checkpoint continuation support but explicitly forbids launching
continuation now. Restore adapter, AdamW and global step, retaining data/seed/
evaluation protocol. Preserve interrupted outputs in a read-only parent directory;
resume always writes a fresh child directory. `max_steps` is total final step,
not an additional update budget. No OPD, hardware allocation or full BAGEL run.
Run/implementation branch: `codex/run/phase1a0-h2008-20260930`.

## Comparability contract

Interrupted 20474 job completed 755 updates, received SIGTERM at 12:42:22 UTC,
and has a validated complete step-750 adapter/optimizer checkpoint. Old status
is stale. Required continuation: first new update 751, same world_size=8,
rank-specific record formula `(step-1)*world+rank`, same rollout seed formula,
same base/model/dataset hashes, sampling, optimizer and eval/save settings.
Preserve original zero-B heldout evaluation rather than treating trained reader
as the new baseline. Legacy checkpoints lack global RNG state; allow them only
with original resolved config, manifest and initial heldout evidence, and label
deterministic-seeded legacy recovery explicitly (not exact global-RNG restoration).
New checkpoints capture per-rank RNG and verified resume sidecars.

## Code translation

- `reader_warmup.py`: lightweight resume inspection, strict source/config/hash/
  optimizer/global-batch checks; restore adapters/optimizer/step and per-rank RNG;
  preserve initial evidence; per-checkpoint resumability sidecars.
- Training CLI: `--resume-checkpoint`; validation-only inspects checkpoint without
  loading full BAGEL or creating output.
- Tests: uninterrupted vs resumed CPU tiny MoT adapter/optimizer/metrics parity,
  two-rank gloo continuation and rank sample sequence, legacy-750-format recovery,
  tampered/missing artifacts and changed config/world/datasets rejected.
- Documentation: explain fresh output requirement and total-step semantics.

## Execution, outputs and budget

Only CPU synthetic tests and read-only real checkpoint inspection. Local isolated
test Python: `/tmp/bagel-v4-tests.nWADtD/venv/bin/python`.
Expected time: 15–25 minutes implementation/testing; no H200 training time.
No main run command is executed. Test logs and validation are recorded here.
Acceptance: resumed updates/optimizer match uninterrupted within tight numerical
tolerance, same next prompt IDs/seeds, original initial metric unchanged, guards
fail before full model load. Startup correctness is not semantic utility evidence.

## Fallbacks and recovery

Reject invalid or incomplete checkpoints; never silently fall back to weights-only
or restart optimizer/step. Do not delete/truncate original metrics (including 751–755).
If legacy provenance is absent, require the correct original run artifacts.
Specialized experiment execution/artifact tools unavailable: local tools and files.
No formal resume launch until a separate explicit user request.

## Checklist and revision

See CHECKLIST.md. 2026-09-30: user authorizes implementation only; supersedes
previous no-resume-code instruction, but not authorization to start training.
