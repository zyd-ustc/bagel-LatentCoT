# 20474 eight-H200 continuation plan

## 1. Objective

User explicitly authorized eight-card continuation on20474. Resume Phase1A.0
Reader Warm-up from saved step750 to total5000, with optimizer restored.
This is the same interrupted training line, not a new algorithm or OPD stage.
Question: does heldout reconstruction continue improving under unchanged setup?
Null: further updates do not improve useful reconstruction versus controls.

## 2. Baseline and comparability

Parent run: `phase1a0_reader8_20474_20260930_115000`, interrupted at755.
Checkpoint: parent `main/reader_warmup_step_0000750.safetensors` plus original
metadata/AdamW/source manifests.751–755 were not saved and are recomputed.
Original train24547/heldout64, evaluated prefix8, seed42, B1/rank/global8.
K8/body[12,20), rank8/alpha16,512x512,50 schedule points,shift3,CFG1,
two native states/rollout. AdamW1e-4,betas.9/.95,wd0,clip1; eval/save250.
Same frozen writer/native model, train FP32 reader A/B, no generation injection.
Original operational gate unchanged; correct-vs-zero remains a separate diagnostic.
Legacy old checkpoint has no full RNG: `legacy_seeded_rollout`, not bitwise H200
resume. Code/model/data hashes checked; parent files never overwritten.

## 3. Code translation

Reuse validated prepared deployment:
`/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-resume-20260930`.
Originally deployed from58cb0a6; no `.git` checkout. Training module, CLI and YAML
SHA-256 identical to current GitHub/local084182d. New scoring/offline-only changes
do not alter the training mechanism. Add launcher/run documentation only.
Active dedicated local run branch remains `codex/run/phase1a0-h2008-20260930`.

## 4. Execution design

Root: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000_resume750`.
Pilot: separate `smoke/`, two resumed updates751/752 with original evaluation8;
require complete status752, world8, finite losses/gradients, restored750 baseline,
native parity<=1e-6, new resume marker/RNG sidecars and valid optimizer752.
Main: restart from original750 into fresh `main/`, final5000. Do not promote pilot
as extra optimization budget or silently load its752 checkpoint.
Stop on invalid artifact, NaN/parity mismatch, pilot failure or resource conflict;
do not kill unrelated jobs or auto-start OPD. Preserve every failed attempt.

## 5. Runtime strategy

Only20474 CUDA0–7 assigned. Recheck eight cards idle immediately before launch
and after pilot exit; do not consume20344/20470. Persistent named tmux session
`phase1a0_resume750_20474`; stdout/logs survive SSH disconnect.
Capture GPU/process/environment/data/code snapshots, exact commands and source
hashes. Two-step pilot expected2–5min including model load/evaluation. Remaining
4250updates expected approximately4–6h from prior throughput; not guaranteed.
Bounded launch checks every30–60s; verify actual main optimizer updates before
handoff. This request does not create a recurring automation or completion claim.

## 6. Recovery

If prepared source checks fail, stop to reconcile rather than change the model.
If GPUs become occupied, stop launch before allocations; touch only our scoped
run on concrete failure. Old run remains intact. New checkpoints preserve
per-rank RNG and completion markers. SIGTERM can leave status.json stale; use
processes and durable logs, not status alone.
Skill-specific bash/artifact/memory tools unavailable: ordinary execution plus
project PLAN/CHECKLIST/RUN and captured logs provide the fallback record.

## 7. Checklist

See CHECKLIST.md and RUN.md. CPU preflight and two-update pilot passed;
formal continuation completed at step 5000, verified on 2026-10-03. Semantic
image-generation evaluation remains pending. No runtime Python implementation edits.

## 8. Revision log

- 2026-10-02: user authorizes20474, eight cards. Snapshot14:37UTC: eight idle
  H200,139.8GiB free/card. Previous refusal to start is superseded by this request.
- 2026-10-03: completion verified; final heldout readout MSE favors correct over
  shuffled and zero in the evaluated eight prompts. This is not image-quality evidence.
