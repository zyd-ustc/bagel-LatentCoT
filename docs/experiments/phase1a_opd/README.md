# Phase 1A T0 Self-CoT OPD

This is the **T0 implementation**, not a completed H200 run or evidence of
semantic gain. T1 Draft-Verify, Q_mem LoRA, recurrent Write/Deep Supervision and
SMA are explicitly deferred until the plan's gates pass. Existing v2 entrypoints
remain as historical baselines; OPD uses a separate checkpoint schema.

## Protocol

- Data: curated T2I semantic prompt JSONL only, with `id`/`prompt_id`, `prompt`,
  and `category` in `count`, `spatial_relation`, `attribute_binding`,
  `multi_object_composition`, `action_relation`, `rare_concept`, or
  `reasoning_heavy_t2i`. No source/target images, style, text rendering,
  portrait identity or edit pairs. The existing 115,883-row dedup export is
  **not automatically** a valid Phase 1A semantic subset because it lacks
  these categories; curate train and held-out splits first.
- Offline frozen BAGEL generates the seven-section Text-CoT once. Every row
  must have nonempty sections, 80–160 BAGEL tokens, no refusal, and no explicit
  count contradiction. These checks are conservative, not a semantic judge.
  The cache is immutable during training; teacher velocity remains online.
- M0 samples 8 actual content-token hidden rows at the start of layer 12;
  short prompts repeat slots with deterministic 1e-5 jitter. Strict Read
  computes M on `(P,x_t,t)` and detaches it. The native frozen GEN/UND Q/K/V
  and backbone are never trained. An independent low-rank O_mem branch is
  installed only in layers `[12,20)`; its B matrix starts at zero. No M is
  concatenated into native K/V; no prompt mask or extra Write loop is used.
- Student rolls out its current policy at CFG=1 with BAGEL `num_steps=50`
  (49 velocity updates) and selects 2 states.
  Both frozen teacher `[P;R]` and student `P+M` predict at each identical
  detached `(x_t^S,t)`. The *only* optimized loss is mean velocity MSE.
  Backward is per selected state, then one clipped AdamW update; the full
  trajectory is never kept for autograd.
- Shuffle/zero controls appear only in diagnostics/evaluation, never in the
  training loss. Same-state field errors are reported for native/correct/
  shuffled/zero versus teacher. Optional five-arm images use the same initial
  noise, CFG and timestep schedule, but each trajectory evolves independently. No semantic
  or image-quality score is fabricated; external GenEval/CoRe and quality
  scoring is still needed for the plan's final go gate.

## Execution order

1. Prepare **separate curated train and held-out** JSONL files. Give each
   record a stable id, prompt and allowed semantic category.
2. Build the offline T0 cache. The builder stops at the first invalid row and
   leaves a `.partial` file for inspection; it never silently accepts a bad
   teacher explanation.

   ```bash
   PYTHONPATH="$PWD" python scripts/data/build_bagel_cot_teacher.py \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/semantic_train.jsonl \
     --output /path/to/semantic_train_t0.jsonl
   ```

3. Run the **pre-training teacher baseline** on at least 8 semantic prompts
   and two seeds. The JSON is a field-difference gate only, not proof that
   teacher images improve semantics. Optionally generate images for human or
   external scorer comparison against native.

   ```bash
   PYTHONPATH="$PWD" python scripts/evaluate/bagel_memory_opd_eval.py \
     --baseline-only --config configs/training/memory_opd_t0.yaml \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/semantic_train.jsonl \
     --teacher-cot-data /path/to/semantic_train_t0.jsonl \
     --output-dir /path/to/teacher_baseline
   ```

4. Train only after inspecting the baseline. Pass its
   `teacher_baseline.json`; the runner rejects a missing, mismatched or
   nonpositive field/semantic gate. A preflight with `--validate-only` checks
   paths, categories and cache schema without loading model weights.

   ```bash
   PYTHONPATH="$PWD" python scripts/train/bagel_memory_opd.py \
     --config configs/training/memory_opd_t0.yaml \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/semantic_train.jsonl \
     --teacher-cot-data /path/to/semantic_train_t0.jsonl \
     --teacher-baseline-json /path/to/teacher_baseline/teacher_baseline.json \
     --output-dir /path/to/opd_train
   ```

5. Generate a **separate held-out CoT cache**, then evaluate a T0 checkpoint.
   The evaluator refuses the training split unless explicitly marked as debug.

   ```bash
   PYTHONPATH="$PWD" python scripts/evaluate/bagel_memory_opd_eval.py \
     --config configs/training/memory_opd_t0.yaml \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/semantic_heldout.jsonl \
     --teacher-cot-data /path/to/semantic_heldout_t0.jsonl \
     --adapter-path /path/to/opd_train/reader_step_0000250.safetensors \
     --output-dir /path/to/opd_heldout_eval --generate-images
   ```

The train output has `metrics.jsonl`, `train_diagnostics.jsonl`, adapter
safetensors + metadata, optimizer snapshots and `status.json`. Training-split
diagnostics are **not held-out results**. No exact optimizer/data-cursor resume
is implemented. Distilled checkpoints must not be loaded by legacy v2 entrypoints.

## What still requires evidence

Initial zero-effect native parity and isolated reader gradients are covered by
tiny-BAGEL CPU tests. These do **not** validate full H200 memory, numerical
stability, teacher quality, semantic gains or visual quality. Before promoting
Phase 1A, verify on held-out semantic subsets:

1. `E_correct < E_native` on the teacher field.
2. `E_correct < E_shuffled` without training on shuffled memory.
3. Student end-to-end semantic score exceeds native without quality collapse.

Only after these gates should T1 Draft-Verify, Q_mem LoRA or loop supervision
be implemented and evaluated. See [plan](PLAN.md) and [checklist](CHECKLIST.md).
