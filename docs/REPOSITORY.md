# Repository maintenance

## Current line

Repository: `zyd-ustc/bagel-LatentCoT`; current/default branch: `main`.
The Phase 1A v4 Reader Warm-up / Self-CoT OPD implementation is promoted by a
fast-forward from the existing main; no forced rewrite or squashed history.
New branch/tag names do not use a `codex/` prefix. Old experiment documents retain
their original branch names and commit identities as historical provenance.

## Archived remote branches

On 2026-10-03 the user confirmed keeping only `main` as a remote branch. Each
old branch tip was published as a lightweight tag and its exact SHA verified
before deleting the branch. Deletion used expected-SHA guards and one atomic
push, so a concurrently changed branch would stop the operation.

| Historical branch | Preserved tag | Commit |
|---|---|---|
| `codex/phase1-structured-reflection` | `phase1` | `8f9c784638d2742e54bab6784c4b7c1f1c232710` |
| `codex/phase1a-selfcot-opd` | `phase1a-opd` | `600d195cdc7b8595c1060d674fe4c8519c93b0b4` |
| `codex/phase1a-v4-reader-warmup` | `phase1a-warmup` | `3a20321bed1442ddf59b0eb196d1e6b9016f047d` |
| `codex/run/phase1a0-h2008-20260930` | `phase1a0` | `084182db0353c0c90044eb9d4edaaa537a2ee644` |

`phase1a0` preserves the previously published branch tip, not the later cleanup
and completed-run documentation now on `main`. Tags are immutable historical
anchors; continue development from `main`. Other local worktrees and local-only
changes were not deleted or reset. The old unrelated local Qwen `main` remains
as `qwen-main`; its remote repository was not modified.

To inspect a historical version without moving the current checkout:

```bash
git fetch --tags bagel
git show phase1a0:README.md
```

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
