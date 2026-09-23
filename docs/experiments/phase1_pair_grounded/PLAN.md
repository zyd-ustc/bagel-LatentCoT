# Phase 1 Pair-Grounded Implementation Plan

## Objective

Replace the structured-reflection mainline with paired visual supervision while
retaining the old delta-velocity trainer only as an ablation.

## Locked architecture

- K=8, R=2, same-depth body `[12,20)`, strict Read then Write.
- Phase 1.1 runs `prefix -> strict Read -> STOP` and trains UND-Q LoRA only.
- Phase 1.2A loads/freeze the Phase 1.1 UND-Q adapter and trains GEN-Q LoRA only.
- `m0`, BAGEL, VAE, ViT, projectors, K/V, FFN, norms, and GEN-O stay frozen.

## Training sequence

1. Validate paired data without requiring reflection or teacher scores.
2. Train pair-memory delta from frozen source/target visual references.
3. Gate on memory cosine, relative error, no-op RMS, rank, and retrieval.
4. Train conditional target-flow SFT with the ground-truth velocity.
5. Add joint relaxation, swap controls, and persistence only after 1.2A passes.

## Baseline and provenance

- Code baseline: branch `codex/phase1-structured-reflection`, HEAD `5b1b637`.
- Old `bagel_loop_delta_v_distill.py` remains the reflection ablation.
- Primary specification: `docs/BAGEL_LatentCoT_Phase1_Pair_Grounded_Memory_Plan.md`.
