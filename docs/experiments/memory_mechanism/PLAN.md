# Phase 0.5 memory mechanism vs content — implementation contract

## Objective / constraints

Implement the supplied 2026-09-26 plan in `bagel-LatentCoT-v1` only. Frozen
BAGEL, K=8, R=2, body [12,20), strict Read then one Write, fresh each timestep,
native SOI/EOI initialization, prompt KV visible. No adapter, dynamic-prompt
residual, prompt-aware initialization, prompt mask or single-pass intervention.
This task implements and tests code; it does not launch an NPU experiment.

## Baseline and comparability

Six modes: native, static_null, zero_dynamic, normal, shuffled_dynamic,
frozen_correct. Native has K=0 and one native decoder pass. All other modes
share the same prefix/Read/reset/Write/suffix call schedule. Non-memory rows
cannot read memory in prefix/Read, including boundary rows (no relay).
All modes keep prompt KV visible and the same image timestep schedule/CFG.
Native has fewer transformer calls by definition: equal denoising NFE is NOT
equal transformer FLOPs. Do not claim compute-matched A–B comparison.

Clarifications required to make the supplied protocol executable:

- C starts the Write at zero; its Read is computed and discarded, as in the
  plan's C-vs-D comparison. It is not a persist/zero-every-layer intervention.
- B clamps hidden and post-normalization/post-RoPE Q/K/V to zero throughout
  prefix, Read, Write AND suffix, so suffix cannot reintroduce content.
- F freezes hidden to M_read through Write AND suffix. Each layer still uses
  its own projections; K/V are not copied between different layers.
- E swaps whole [K,D] states between two distinct prompts in the same packed
  pair. Pair identity is stable across seeds/shards and recorded. Each CFG
  branch performs the same intervention using its own Read, never conditional
  memory in the unconditional branch.
- Hidden sensitivity is measured in the conditional branch. Velocity is the
  final guided velocity. Attention masses at body layers 12 and 19 are exact means, computed in chunks
  without saving full attention maps; disabled for end-to-end image runs.
- Every probe arm receives the SAME native x_t. Only native advances the probe
  trajectory. Independent end-to-end trajectories run separately afterward.

## Code map

- `memory_mechanism.py`: named controls, decoder execution, strict-null clamp,
  chunked attention diagnostics and guards.
- `qwen2_navit.py`: explicit opt-in inference hooks. Preserve training/legacy
  internals; the new runner cannot combine them with old controls.
- `mechanism_inference.py`: packed-pair preparation, frozen velocity engine,
  guided/native parity, counterfactual metrics and independent trajectories.
- `scripts/evaluate/bagel_memory_mechanism.py`: unified six-arm protocol,
  stable pair sharding, manifests, traces, curves and gallery.
- Old T2I/write-sensitivity/single-pass CLI entry points fail with migration
  guidance. They no longer silently launch a different experimental protocol.

## Verification / outputs / recovery

CPU tests: layer/round traces; null QKV despite biases; fixed F hidden;
same-state probe; reset/non-memory invariance; whole-sample shuffle; native
parity; CFG and cache isolation; schema/sharding/merge validation.
Syntax checks and a dry-run precede any user-launched NPU smoke (2 prompts).
Main run: hard16, one seed; larger/three-seed runs only after smoke validation.
Required outputs: run manifest, Base states, per-step metrics/attention JSONL,
six images/prompt, timestep curves, gallery. Pixel MAE is behavioral distance,
not semantic quality. No unmeasured quality claim is allowed.

Source backup: `/tmp/bagel-mechanism-backup.yDAIvs/inference-before.tar.gz`.
The copied `.git` points to a missing remote worktree; do not rewrite it or
claim a Git commit. Local system PyTorch is broken; use an isolated test env.
On failure stop scaling, record the issue, test the smallest discriminating
case. `CHECKLIST.md` records implementation/test status; NPU remains unrun.

## H200 adaptation (2026-09-26)

Only this `bagel-LatentCoT-v1` copy is modified. The launcher now defaults to
CUDA/current Python/repository-local weights, with explicit NPU opt-in. Pair
shards are bounded by visible allocated GPUs and prompt pairs, and each worker
receives one visibility token. Strict runtime checks run before loading weights;
hardware metadata is saved per shard. No scientific controls or schedules change.
CPU inference cannot dispatch to CUDA FlashAttention just because it is installed.
Local CPU regression: 204 tests pass; actual H200 smoke remains user-run.
