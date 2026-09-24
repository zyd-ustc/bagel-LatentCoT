# Prompt-KV visibility — hard-16 paired experiment

## 1. Objective

- Run ID: `prompt_memory_kv_visibility_hard16_v3`.
- Tier: auxiliary/dev mechanism ablation; no image-quality claim without scoring.
- Selected idea: compare the same frozen prompt-as-memory loop with native prompt KV visible to all image-generation queries versus prompt KV hidden from every **non-memory** query. Memory queries keep prompt access in both groups.
- User requirements: publish code to GitHub main and provide NPU commands; the user runs the experiment.
- Question: can prompt-aware, Read-updated memory carry prompt information to GEN without a direct prompt-KV route?
- Null: masking direct prompt KV does not alter images or four-arm Write sensitivity.
- Alternative: image content or four-arm sensitivity changes under masking.

## 2. Baseline and comparability

- Baseline: `keep` policy in this **same v3 executable**, not old v1/v2 runs.
- Data: `experiments/data/geneval2_hard_16.jsonl`, file order, adjacent pairs, all 16 prompts.
- Fixed: model weights, frozen LoRA, K=8, R=2, body from training YAML (`[12,20)` in the early config), same-depth, strict Read, fresh memory per timestep, prompt-aware `M_init`, initial noise, seed, CFG, 50-step schedule, resolution, pair order, and sample-global CFG normalization.
- Variable: `keep` versus `mask_nonmemory` prompt-KV visibility in the conditional image-generation branch. Mask applies to **every** generation layer and round, not merely Write. Only memory queries may attend cached prompt keys when masked. Prompt cache is built identically and never mutated.
- Four Write arms run under each policy: `correct_M`, paired-sample shuffled Read memory, prompt-aware `m0`, and zero memory. CFG unconditioned branch is the same fixed `correct` path for all eight cells.
- Primary observations: per-prompt keep-vs-mask pixel MAE for the same arm, and per-policy arm-vs-correct MAE; gallery and raw memory probes. Pixel MAE establishes sensitivity, not semantic quality. Optional later semantic scoring must use the same image maps and metric definition.
- Confounder: masking only GEN queries would leak prompt information through current boundary queries. Therefore the mask covers every non-memory query, including image boundary queries, in prefix/body/suffix.

## 3. Code translation

| Path | Current role | Planned change | Guard |
|---|---|---|---|
| `qwen2_navit.py` | packed attention and same-depth body | block per-sample cached prompt key columns for all non-memory query rows; merge with strict Read memory mask | preserve memory-query prompt access; fail closed outside MoT same-depth loop |
| `bagel.py` | generation branch routing | pass mask only to conditional branch across all layers | keep unconditional CFG identical |
| `bagel_write_sensitivity_t2i.py` | four-arm paired evaluation | add keep/mask policy axis, 8-column gallery, v3 manifest and per-prompt deltas | reuse one prompt cache/noise bundle for both policies |
| `tests/` | routing checks | assert per-sample mask geometry, strict Read composition, policy forwarding and legacy default | no silent masking in trained path |

## 4. Execution design

- Minimal pilot: 2 prompts × 2 policies × 4 arms × 2 schedule points, 16 PNGs in a new output directory.
- Full run: 16 prompts × 2 policies × 4 arms × 50 schedule points, 128 PNGs, one NPU card, one process.
- Stop if: mask geometry is invalid, cache absent in conditional branch, results incomplete, memory probes nonfinite, policy noise hashes differ, or pilot fails.
- Do not silently switch to batch size 1: within-pair shuffle would cease to be valid.
- Strongest alternative explanation: with prompt KV removed from non-memory rows, a frozen Write route may be too weak to carry prompt semantics despite functional routing.

## 5. Runtime and recovery

- Commands and new output directories are in `RUN.md`; no NPU launch by this agent.
- Use a 2-prompt pilot before the full run. Keep output directories distinct from prior v1/v2 outputs.
- If batch-2 OOMs, stop and report rather than changing resolution within the same comparison.
- User monitors log completion and verifies `run_manifest.json` has `complete=true`, 16 prompts, 2 policies and 4 arms.

## 6. Revision log

| Date | Decision | Impact |
|---|---|---|
| 2026-09-23 | Mask cached prompt keys to all non-memory generation queries, not just VAE GEN rows | closes the boundary-token relay confound |
| 2026-09-23 | Keep prompt cache and memory-query access intact | isolates direct prompt-to-nonmemory routing; does not test removal of prompt information from memory |
| 2026-09-24 | Use one global RMS for centered slot offsets | per-slot RMS after centering broke the `mean_k M_init(P)=A_s(P)` invariant; fix is code-only, no NPU results existed yet |
