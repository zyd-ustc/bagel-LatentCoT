# Fixed reader injection diagnosis

## 1. Objective

Parent: Phase1A.0 step-5000 from `phase1a0_reader8_20474_20260930_115000_resume750`.
Question: does its translation adapter improve generation versus the same frozen
reader without the trained translation, at a fixed injection coefficient?
User requests coefficient 1.0 (editable), 16 hard prompts and 8 independent GPUs.
This is a diagnostic intervention, not gate-trained OPD. No paper outline applies.

## 2. Boundary and comparability

Three arms: native; untrained_reader; step5000_reader. The latter two share the
same fixed coefficient in every body layer and timestep. Untrained readout uses
B=0, exactly the initial translation function regardless of A. Everything else
is frozen, including native Q/O and writer. Prompt KV remains visible; no Write,
mask, memory persistence or R ablation is introduced. Each arm evolves its own
trajectory, starting from an identical per-prompt CPU-generated noise tensor.
Model, geometry, schedule, CFG=1, K/body/rank/alpha come from checkpoint config.
Gate=0 must reproduce native velocity before image generation. This intervention
does not preserve Warm-up's no-injection generation contract when gate is nonzero.

## 3. Slice plan

| Slice | Type | Question | Comparator | Priority |
|---|---|---|---|---|
| fixed-g1-hard16 | ablation | Does trained translation add semantic value? | native and B=0 reader | first |

## 4. Hypothesis and interpretation

Trained versus untrained is the translation test; either reader versus native
also changes the generation architecture. Changed images, final latent distance
or lower reconstruction loss are not semantic gains. Preserve null/negative results.

## 5. Assets

Existing checkpoint, model and original prompt data remain on H200 shared storage.
Use the existing independent semantic hard64 list; select four prompts each from
atomicity 7/8/9/10, preserving order/VQA metadata, and audit normalized train overlap.
No new data source, teacher cache or learned gate checkpoint is needed.

## 6. Execution

Only CPU/tiny-model tests run here. User runs H200 GPU smoke/main when memory
is available. One frozen model per GPU, prompts sharded by global index. Require
enough visible cards and configurable minimum free GiB before launching workers;
do not wait for cards, kill other jobs, or start training. Full-model success
remains unverified until a real run completes. Fresh output directories only.
Worker manifests bind checkpoint/config/benchmark/seed/scale; merge requires
exact prompt coverage, intact PNGs, same noise/schedule per arm and passed parity.

## 7. Reporting

Output HTML, benchmark JSONL, manifests, per-prompt noise hashes/parity and image
maps. Later automated GenEval2 scoring must determine semantic improvement.
Readout-only parent metrics cannot substitute for that scoring.
Dedicated campaign/bash/artifact tools unavailable: project files and normal
execution logs provide the fallback. Parent code and saved run are not modified.

## 8. Checklist

See CHECKLIST.md. Next: implement and validate CPU wiring; GPU run is user-owned.

## 9. Revision log

2026-10-03: user explicitly authorizes fixed coefficient 1.0, 8 GPUs, hard16.
