# Continuation implementation validation

## Local CPU evidence

2026-09-30: full repository suite, 329 passed in 18.71 s.
Command: `PYTHONPATH="$PWD" /tmp/bagel-v4-tests.nWADtD/venv/bin/python -m pytest -q`.
No full-model H200 computation or formal continuation was performed.

Covered: checkpoint-1/2 continuation vs uninterrupted 4-step tiny MoT, exact
adapter tensors and AdamW counters/moments, identical metrics/next samples,
CPU/Python/NumPy stochastic stream restoration, two-rank gloo averaged updates
with distinct per-rank RNG, preserved zero-step evaluation and read-only parent.
Legacy three-file layout restores optimizer/step with explicit seeded rollouts.
CLI validation exits without loading BAGEL or creating output; wrong world,
data/config/budget, missing/tampered optimizer, incomplete sidecars, invalid RNG,
changed source/baseline hashes fail. Explicit reader/loss defaults normalize to
the same effective contract as omitted defaults.

## Real artifact preflight

Pending isolated code upload and CPU-only validation of real step-750 checkpoint.
This preflight is not a GPU continuation pilot and cannot establish bitwise H200
equivalence of legacy RNG state that was never saved. Formal continuation deferred.

## Reusable recovery lessons

Outcome: implementation validated, full training partial/interrupted, no continuation.
Idea: Phase1A.0 Reader Warm-up; branch: codex/run/phase1a0-h2008-20260930;
parent run: phase1a0_reader8_20474_20260930_115000.
SIGTERM can leave status.json stale without a Python exception; inspect processes
and torchrun's root-cause signal, not just status text. A weights-only artifact is
not optimizer continuation. Preserve original initial evaluation and global batch;
starting at the checkpoint's next global step preserves rank sample/rollout seeds.
Write new outputs separately instead of rewriting uncheckpointed parent metrics.
