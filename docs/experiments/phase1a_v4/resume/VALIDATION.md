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

Isolated code uploaded to shared filesystem:
`/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-resume-20260930`.
Deployed code commit: `58cb0a62b04701ffe5f8ed46e36f562655702283` (later local
documentation commits do not change the deployed Python implementation).
20474 validation-only succeeded with CUDA hidden against actual
`phase1a0_reader8_20474_20260930_115000/main/reader_warmup_step_0000750.safetensors`:
step=750, world_size=8, mode=legacy_seeded_rollout; train24547/heldout64.
Code/CLI SHA256 match local files:
`62a8d97666008d16e06e021c9b437fd98470867224df3bf381cd3879dfe7fd6a` and
`cc8d0d5b5a50f0f07c8f1aaa9902daef81bbac54398e7bbaacdb393c00847e77`.

Linux/PyTorch2.5.1+cu124/Python3.11 CPU test suite: 23 passed in 28.31 s.
Command: CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
PYTHONPATH=code directory, existing lcot Python, pytest -q
`tests/test_reader_warmup_resume.py`. Full remote test log is
`docs/experiments/phase1a_v4/resume/linux_cpu_tests.log` under the deployed code.

Proposed continuation output `phase1a0_reader8_20474_20260930_115000_resume750/main`
does not exist after validation. Original interrupted status running/755 remains
unchanged and stale; no old artifact was rewritten. No real continuation started.
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
