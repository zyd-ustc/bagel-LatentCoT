# Phase 1A T0 implementation validation — 2026-09-28

## Completed locally

- Backed up the complete pre-change v1 source at
  `/Users/zyd/Documents/LCoT-codex/bagel-phase1a-backup.aRDlRK/before-phase1a.tar.gz`.
  This was created before editing; no old v2 scripts or results were deleted.
- Full local CPU suite: **282 passed in 7.18s** with
  `/tmp/bagel-mechanism-tests.WOiF4o/bin/python` (PyTorch CPU). Changed Python
  modules and all three new CLIs also passed bytecode compilation.
- New tiny-MoT tests assert: exact native velocity parity at zero-effect O_mem;
  only O_mem adapter has gradients; strict Read receives sampled prompt hidden
  **at the body entry**, not at token embedding; K0 native forward remains
  untouched; eval-mode activation checkpointing preserves reader gradients.
- Data/CLI tests assert: semantic category and teacher-cache schema gating;
  count contradiction rejection; cache/prompt id and text matching; ranking
  settings rejected; baseline evidence bound to cache/model/schedule; train
  CLI and cache-builder preflight load no full BAGEL weights; fixed-state
  native/correct/shuffled/zero controls handle multiple prompt pairs.

## Not claimed

- No full BAGEL weights were loaded; no H200/NCCL pilot, CoT cache generation,
  training, image generation or semantic scorer was run. CPU tests demonstrate
  mechanism wiring, not numerical/visual quality at 7B scale.
- The automatic CoT quality gate is intentionally conservative: it checks
  seven sections, token length, refusals and explicit count conflicts. It
  cannot certify all object/attribute/relation semantics. Manual or external
  evaluation of the frozen teacher is still required.
- Pre-training field difference is a gate to attempt OPD, **not** evidence of
  semantic improvement. Final Phase 1A go requires held-out field ordering,
  end-to-end semantic gain and no obvious image-quality collapse.
- The checkout's `.git` points to an unavailable Linux worktree. No commit,
  GitHub push or remote upload was attempted or claimed.

## Decision

Implementation is ready for a curated-data/teacher-baseline pilot, not for a
research-success claim or automatic full training. The code keeps T1 and loop
stages deferred until the T0 gates are measured.
