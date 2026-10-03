# Repository maintenance

## Current line

Repository: `zyd-ustc/bagel-LatentCoT`; current/default branch: `main`.
The Phase 1A v4 Reader Warm-up / Self-CoT OPD implementation is promoted by a
fast-forward from the existing main; no forced rewrite or squashed history.
New branch/tag names do not use a `codex/` prefix. Old experiment documents retain
their original branch names and commit identities as historical provenance.

## Cleanup on 2026-10-03

- README focuses on the current entrypoints; historical routes have their own index.
- Removed two launchers with missing train/generation/config dependencies and an
  unreferenced common shell helper. They remain recoverable from Git history.
- Deduplicated the reflection design, retaining an old-path pointer to its full copy.
- Corrected transfer-script usage examples and package description.
- Recorded completed step-5000 training and its evaluation scope, without publishing
  weights/datasets or changing training, generation or checkpoint compatibility.

Historical modules with tests and reusable entrypoints remain in place. No raw
training loss or smoke result is represented as semantic image-quality evidence.

## Validation

`PYTHONPATH="$PWD" /tmp/bagel-v4-tests.nWADtD/venv/bin/python -m pytest -q`:
403 passed in 40.32 seconds on local CPU. The temporary Python path identifies
the validation environment; users can run `python -m pytest -q` in their own
compatible environment. New layout tests check shell syntax, local navigation
links and literal repository entrypoints. `git diff --check` also passed.

No H200 model execution was performed as part of this repository cleanup.
