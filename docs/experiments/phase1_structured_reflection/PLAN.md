# Phase 1 Structured Reflection Experiment Plan

## 1. Objective

- run id: `phase1_structured_reflection_fresh_v1`
- selected idea: Distill the velocity correction produced by native BAGEL editing
  with a short structured text reflection into the frozen model's strict latent
  Read→Write loop. Only loop-gated UND-Q and GEN-Q LoRA are trainable.
- user's core requirements: implement the supplied Phase 1 design while matching
  BAGEL's native context, CFG, schedule, cache, and flow-velocity contracts.
- non-negotiable constraints: Base/Teacher use per-call K=0; Student uses K=8;
  no temporary config mutation; no full-trajectory autograd; frozen m0/base model;
  strict Read then Write in `[12,20)`; Q-only first.
- research question: can the latent loop reproduce native structured-reflection
  velocity corrections without relearning BAGEL's image prior?
- null hypothesis: the student correction does not align with the teacher and
  no-op drift remains high.
- alternative hypothesis: cosine alignment rises and normalized correction error
  falls while no-op correction stays small.

## 2. Baseline And Comparability

- baseline id: native BAGEL editing, K=0, source image + instruction.
- teacher variant: native BAGEL editing, K=0, source image + instruction +
  structured reflection.
- student variant: same source/instruction and exact `(x_t,t)`, K=8, strict
  `1R+1W`, early body `[12,20)`, fresh memory.
- dataset / split: Phase 1 structured-edit JSONL grouped by source scene; initial
  code smoke uses fixtures only, then 100–500 train samples for overfit.
- primary metric: `cos(delta_v_student, delta_v_teacher)`.
- required metric keys: normalized Smooth-L1, direction loss, overshoot loss,
  no-op loss, cosine, relative correction error, student/teacher correction RMS.
- comparability risks: differing CFG cache layouts, timestep schedules, latent
  states, or loop-token layouts invalidate the comparison.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `qwen_latent_cot/bagel/inferencer.py` | Native BAGEL orchestration | Per-call K and reusable velocity bundle/kwargs | Explicit K=0/8 contract | CFG layout mismatch |
| `qwen_latent_cot/bagel/modeling/bagel/bagel.py` | Native flow rollout | Capture selected Euler states without SDE | Replay without full graph | Altering existing GRPO trajectory |
| `qwen_latent_cot/bagel/loop_distill.py` | New | Velocity replay, timestep sampling, Δv losses | Isolate Phase 1 math | Accidental base gradients |
| `scripts/train/bagel_loop_delta_v_distill.py` | New | Single-device CUDA/NPU trainer | Minimal Phase 1.1 execution | Large-model memory |
| `configs/training/loop_delta_v_early_*.yaml` | New | Fresh/Persist matched configs | Auditable defaults | Path portability |
| `tests/test_loop_distill.py` | New | Contract/loss/replay tests | Fail closed before compute | Mock/source drift |

`qwen2_navit.py` and the loop architecture remain unchanged.

## 4. Execution Design

- minimal experiment: unit-test per-call K=0/8, state capture, loss gradients,
  trainable allowlist, and dataset/reflection validation.
- smoke / pilot plan: one 512×512 source, one non-noop plus one no-op record,
  one optimizer update, 2–4 replay states.
- full run plan: overfit 100–500 examples with fresh `[12,20)` Q-only adapter;
  stop before scale-up unless cosine rises and relative error falls.
- expected outputs: v8 safetensors adapter + metadata, resolved config, JSONL
  metrics, and exact source/teacher/student contracts.
- stop condition: non-finite loss/gradient, empty LoRA gradient, mismatched packed
  K, or teacher/base/student `(x_t,t)` mismatch.
- abandonment condition: 500-example overfit cannot lower correction error or
  increase direction cosine.
- strongest alternative hypothesis: the native text reflection changes velocity
  in a way Q-only latent memory cannot express; GEN-O is considered only after
  Q-only underfit is demonstrated.

## 5. Runtime Strategy

- smoke command: `python scripts/train/bagel_loop_delta_v_distill.py --config configs/training/loop_delta_v_early_fresh.yaml --max-steps 1`
- main command: same entry with validated data/output overrides.
- expected runtime / budget: code/unit validation under 10 minutes; model smoke
  depends on one available CUDA/NPU and checkpoint I/O.
- log / artifact locations: configured output directory with
  `metrics.jsonl`, `resolved_config.json`, and v8 adapter pairs.
- safe efficiency levers: selected-state replay, bf16 base, fp32 LoRA, no VAE
  decode during rollout, frozen teacher/base, and 2–4 states per source.
- existing tooling: reuse native BAGEL cache creation, shifted schedule,
  loop-gated LoRA, accelerator abstraction, and exact replay conventions.

Monitoring: check after model load, first rollout, first replay, and first
optimizer step. Kill on OOM, non-finite values, K/layout mismatch, or missing
trainable gradients.

## 6. Fallbacks And Recovery

- endpoint/download failure: use the existing local BAGEL checkpoint only.
- tighter memory: reduce image size and replay states, then enable checkpointing;
  do not change the objective.
- wrong code path after smoke: isolate K/layout and one-state replay before retry.
- non-comparable run: discard it rather than reinterpret metrics.

## 7. Checklist Link

- checklist: `docs/experiments/phase1_structured_reflection/CHECKLIST.md`
- next unchecked item: run one optimizer-update smoke on the training machine.

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-09-21 | Initial Step 1–2 implementation contract | User supplied new Phase 1 plan | No metric change; code-only start |
| 2026-09-21 | Implemented K contract, selected-state replay, loss, trainer, and matched configs | Complete Phase 1.1 code path before accelerator smoke | 128 tests pass; model smoke pending |
