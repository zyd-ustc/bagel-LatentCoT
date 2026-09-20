# Phase 0.5 T2I 128-Prompt Experiment Plan

## 1. Objective

- run id: `phase05_t2i_128_body_persistence`
- selected idea: compare strict `1R+1W` at early, mid, and late body windows, with
  and without cross-timestep memory persistence.
- user requirements: Z0 baseline, matched body-window/persistence ablations, 128
  prompts, and GenEval2 Soft-TIFA scoring.
- research question: does body location or persistent memory improve structural
  composition over frozen BAGEL T2I?
- null hypothesis: none of the six loop arms improves GenEval2 AM/GM over Z0.
- alternative hypothesis: at least one matched loop arm improves AM/GM without
  unacceptable visual degradation.

## 2. Baseline And Comparability

- baseline: Z0 frozen BAGEL T2I.
- dataset: official GenEval2 commit
  `a6e82d2289e8d418f27f0adee77908b07060eea3`; atomicity 7–10, 32 prompts each,
  selected by the existing split script with seed 42.
- primary metrics: Soft-TIFA AM and GM; secondary metrics: per-skill AM,
  per-atomicity GM, pixel MAE, `ΔM`, `ΔG`, and `Δv`.
- comparability: prompt, initial noise, image geometry, CFG, NFE, timestep
  schedule, `K=8`, and `R=2` are shared across all loop arms.

## 3. Code Translation Plan

| Path | Planned change | Risk |
|---|---|---|
| `bagel_loop_t2i_zeroshot.py` | seven-arm matched matrix | arm mismatch |
| `run_bagel_loop_t2i_zeroshot.sh` | 128 defaults and optional integrated scoring | missing VLM |
| `geneval2_hard_128.*` | deterministic balanced evaluation split | split drift |
| tests/docs | lock pair equivalence and dataset balance | stale commands |

## 4. Execution Design

- smoke: unit tests plus `MAX_PROMPTS=1`, `ARMS=Z0,Z2,Z5`, `SCORE=0` on NPU.
- full run: 128 prompts × 7 arms on 16 NPUs, followed by Soft-TIFA scoring.
- expected outputs: gallery, run manifest, mechanism summary, image maps, per-arm
  score lists, AM/GM CSV/JSON/Markdown, skill CSV, and atomicity CSV.
- stop condition: any missing arm/image, non-matching prompt set, or invalid score.
- abandonment condition: persistent OOM or inability to run the pinned evaluator.
- strongest alternative: output changes come from recurrent compute but do not
  improve compositional semantics.

## 5. Runtime Strategy

- smoke command: `MAX_PROMPTS=1 ARMS=Z0,Z2,Z5 SCORE=0 bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /data/outputs/t2i_128_smoke`
- full command: `SCORE=1 bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /data/outputs/bagel_loop_t2i_phase05_128`
- output: `/data/outputs/bagel_loop_t2i_phase05_128`.
- monitoring: inspect all shard logs after 60 seconds, then every 5 minutes; stop
  if a worker exits, images are missing, or NPU OOM repeats.
- fallback: set `SCORE=0` for generation, then score later against a live server.

## 6. Checklist Link

- checklist: `docs/experiments/phase05_t2i_128/CHECKLIST.md`
- next unchecked item: run the one-prompt NPU smoke on the training machine.

## 7. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-09-20 | Expanded 16→128 and paired every body window with persistence | user request | stronger evaluation; 896 images, 6.22× the previous 16×9 matrix |
