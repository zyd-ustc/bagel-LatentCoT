# Phase 1A T0 Self-CoT OPD implementation

## Objective and limits

Implement the user-supplied Phase 1A v3 T0 code plan in this v1 checkout only.
Research question: can a frozen BAGEL writer plus a memory-only generation
residual learn the privileged [prompt; BAGEL Text-CoT] teacher velocity field?
Null: it cannot improve correct-memory teacher error or content specificity.
Alternative: correct memory improves both without a shuffle/dependency loss.
This is an implementation pass, not a main training run or claim of improvement.
T1 Draft-Verify and Phase 2 loop supervision are conditional future stages in
the plan; do not run or silently substitute them for T0.

## Baseline and comparability

Existing memory-grounding v2 remains the last-known-good code and its tests must
still pass. OPD is a separate checkpoint schema/entrypoint. Teacher uses frozen
BAGEL K=0 and [P;R]; student sees P and frozen strict Read memory only. Teacher
and student score the same detached student-policy state, timestep, CFG=1 and
NFE. Only plain velocity MSE is optimized. Shuffle/zero are evaluation-only.
Before GPU training, operator must furnish a teacher-vs-native semantic baseline
gate on fixed prompts/seeds; field difference alone is not semantic improvement.

## Code translation

| Component | Change | Safety gate |
|---|---|---|
| prompt hidden | capture content token layer-12 hidden; uniform K slots | no special tokens/grad; deterministic short-prompt jitter |
| reader | independent GEN residual with native frozen Q + UND K/V and zero-effect low-rank O | no native KV concatenation or GEN-Q training |
| model | optional reader branch in layers [12,20); frozen strict Read API | O=0 matches native K=0 exactly |
| teacher cache | offline BAGEL Text-CoT, schema/length/contradiction checks | cache immutable during OPD |
| trainer | detached student rollout, online same-state teacher, MSE only | frozen writer/backbone and explicit trainable-name check |
| evaluator | fixed-state native/correct/shuffled/zero/teacher | no control-arm training leakage |

## Execution and recovery

Tier: auxiliary/dev code implementation. Minimal experiment: tiny-BAGEL CPU
gradient/parity tests. Full run: deferred to operator after teacher baseline
semantic gate. No remote upload, external model load or H200 run requested.
Stop if parity, isolation, or existing regression tests fail. Local pre-change
archive: `/Users/zyd/Documents/LCoT-codex/bagel-phase1a-backup.aRDlRK/before-phase1a.tar.gz`.
Tooling note: experiment-skill `bash_exec`, quest memory and artifact APIs are not
available here; use local terminal, this plan/checklist and test logs. This
checkout's `.git` points to an unavailable Linux worktree; do not repair it.

## Revision log

- 2026-09-28: Initial T0 implementation contract, no training authorized.
