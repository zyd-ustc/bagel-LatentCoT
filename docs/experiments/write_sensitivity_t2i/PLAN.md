# Write Sensitivity T2I — implementation contract

## 1. Objective

- Run ID: `write_sensitivity_t2i_hard16_pair2_v1`.
- Tier: auxiliary/dev causal mechanism probe; no quality-improvement claim.
- User requirement: implement the attached four-arm plan, push code to GitHub, provide NPU commands; **do not run NPU evaluation in this task**.
- Question: does the generation Write round respond to the *content* of Read memory?
- Null: fixed prompt/noise/CFG/schedule and batch pairing yield no stable image differences across memory sources.
- Alternative: replacing only the Write input changes output images, with correct Read memory differing from wrong/initial/zero memory.

## 2. Baseline and comparability

- Baseline: `correct_M` in the **same paired-batch implementation**, not the old single-prompt hard-16 images.
- Data: `experiments/data/geneval2_hard_16.jsonl`, file order preserved, adjacent pairs `[0,1]` through `[14,15]`.
- All arms: same two prompts, per-prompt initial noises, image geometry, CFG, shifted schedule, K/R/body, frozen model/adapter, pair order, and batch size 2.
- Single changed variable: memory handed from strict Read to the one Write round. `shuffle` is a deterministic pair swap with no fixed point. `m0` is the original loop embedding (not the prefix-transformed state).
- Primary observations: per-prompt image MAE against `correct_M` and side-by-side gallery. These measure **sensitivity**, not semantic superiority. GenEval2 scoring is supplementary and user-run only.
- Comparability risk: BAGEL's original `global` CFG renorm is shared over a batch and would cause cross-sample output coupling. Use `sample_global`: the same global-norm formula applied separately to each packed sample. It equals `global` at batch size 1. Do not compare absolute images to the earlier single-prompt run without a parity check.

## 3. Code translation

| Path | Change | Why / guard |
|---|---|---|
| `qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py` | swap memory only after round-0 Read, before round-1 Write | fail closed for batch size 1, unsupported R/mode |
| `qwen_latent_cot/bagel/modeling/bagel/bagel.py` | pass a write-source selector through generation; add per-sample global CFG normalization | default `correct` + old `global` preserves old behavior; new mode prevents batch coupling |
| `scripts/evaluate/bagel_write_sensitivity_t2i.py` | fixed pair-batch four-arm hard-16 runner | identical prompt/noise/config; gallery and manifest |
| `tests/test_bagel_write_sensitivity_t2i.py` | validate derangement, arm contracts, no-op default | prevent an accidental identity shuffle |

## 4. Execution design

- Minimal run: one pair, two denoising steps, four arms, output under a new directory.
- Full run: eight pairs × four arms × 50 steps at 512×512; not run by this agent per user request.
- Expected outputs: four PNGs per prompt, `index.html`, image maps, `run_manifest.json`, probe summaries, and benchmark subset.
- Stop/fail conditions: missing prompt/checkpoint, incorrect batch size, R≠2, non-strict Read, persistence enabled, non-`same_depth` mode, incomplete images, nonfinite diagnostics.
- Strongest alternative explanation for small deltas: frozen GEN-Q Write path is weak despite functional memory routing.

## 5. Runtime and recovery

- NPU command is documented in `RUN.md`; single NPU card, one process, batch size 2.
- Keep each run's output directory empty/new; do not overwrite previous hard-16 or partial results.
- If batch-2 OOMs, do not silently fall back to batch size 1: that would invalidate `shuffle`. Record the failure and decide on a lower image resolution as a *new* protocol.
- No monitoring by this agent: user will run the NPU commands.

## 6. Revision log

| Date | Decision | Reason |
|---|---|---|
| 2026-09-23 | Use native packed batch size 2 and one Read→Write swap point | The existing single-image inferencer cannot perform a real within-batch shuffle. |
| 2026-09-23 | Use `sample_global` CFG normalization for all four arms | Original `global` renorm pooled both samples, confounding per-prompt memory sensitivity. |
