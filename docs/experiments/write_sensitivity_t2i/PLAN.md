# Prompt-as-Memory Write Sensitivity T2I — v2 implementation contract

## 1. Objective

- Run ID: `prompt_as_memory_write_sensitivity_t2i_hard16_pair2_v2`.
- Tier: auxiliary/dev causal mechanism probe; no quality-improvement claim.
- User requirement: destructively refactor the training-free path per `Prompt_as_Memory_Training_Free_Write_Sensitivity.md`; no NPU run by this agent.
- Question: does a same-depth prompt-aware memory initialization, followed by strict Read, produce a Write-sensitive prompt-specific dynamic state?
- Null: fixed prompt/noise/CFG/schedule and batch pairing yield no stable image differences across memory sources.
- Alternative: replacing only the Write input changes output images; `correct_M` differs from swapped, prompt-aware `m0`, and zero memory.

## 2. Baseline and comparability

- Baseline: `correct_M` in the **same paired-batch implementation**, not the old single-prompt hard-16 images.
- Data: `experiments/data/geneval2_hard_16.jsonl`, file order preserved, adjacent pairs `[0,1]` through `[14,15]`.
- All arms: same two prompts, per-prompt initial noises, image geometry, CFG, shifted schedule, K/R/body, frozen model/adapter, pair order, and batch size 2.
- Single variable **within v2**: memory handed from strict Read to one Write round. `shuffle` is a deterministic pair swap with no fixed point. `m0` is now the prompt-aware body-entry initialization, not the old boundary embedding.
- The model captures each prompt's causal EOS hidden at the **body-entry depth** while building its normal prompt KV cache. `M_init(P) = A_s(P) + 0.05 * b_k`; `b_k` is a centered, RMS-normalized deterministic sin/cos slot pattern, identical for every prompt. No new encoder, projector, parameter, target image, or final-layer-to-early-layer splice.
- Every denoising step reinjects the same `M_init(P)` immediately after prefix `[0,s)`; it does not persist across timesteps. The prompt KV cache remains intact. Unconditional CFG retains its original boundary memory, avoiding prompt leakage into the unconditioned branch.
- The Write-source intervention applies to the **conditional** branch only. CFG's unconditioned branch keeps `correct` Read→Write for all arms, so the unconditional velocity is a fixed control rather than a second intervention.
- Primary observations: per-prompt image MAE against `correct_M`, side-by-side gallery, Read-vs-initial probes, and per-prompt initialization hashes. These measure **sensitivity**, not semantic superiority. GenEval2 scoring is supplementary and user-run only.
- This breaks the v1 arm semantics and manifest schema. Do not pool v1/v2 metrics or load a trained Phase 1.1 adapter into this training-free script. The separate unified checkpoint evaluator keeps its original initialization.
- Comparability risk: BAGEL's original `global` CFG renorm is shared over a batch and would cause cross-sample output coupling. Use `sample_global`: the same global-norm formula applied separately to each packed sample. It equals `global` at batch size 1. Do not compare absolute images to the earlier single-prompt run without a parity check.

## 3. Code translation

| Path | Change | Why / guard |
|---|---|---|
| `qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py` | swap memory only after round-0 Read, before round-1 Write | fail closed for batch size 1, unsupported R/mode |
| `qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py` | capture unnormalized prompt hidden after prefix depth and inject `M_init` at image body entry | no depth mismatch; trained/default path unchanged |
| `qwen_latent_cot/bagel/write_sensitivity.py` | fixed prompt-memory initializer | centered deterministic slots; no trainable weights |
| `qwen_latent_cot/bagel/modeling/bagel/bagel.py` | pass a write-source selector through generation; add per-sample global CFG normalization | default `correct` + old `global` preserves old behavior; new mode prevents batch coupling |
| `scripts/evaluate/bagel_write_sensitivity_t2i.py` | training-free-only pair-batch four-arm hard-16 runner | `--adapter` removed; identical prompt/noise/config; gallery and v2 manifest |
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
| 2026-09-23 | Break v1 `m0` semantics and remove optional adapter from this script | The new hypothesis is training-free prompt-as-memory; trained Phase 1.1 checkpoints retain their original evaluation path. |
