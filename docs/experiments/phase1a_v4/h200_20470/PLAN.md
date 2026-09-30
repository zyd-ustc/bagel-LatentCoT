# Phase 1A.0 H200 20470 execution contract

## Objective and user constraints

User authorized GitHub push, SSH code upload and direct 8-card training on
`ssh -p 20470 root@vr.turbo-ai.com`. Run only Reader Warm-up, not automatic OPD.
Hypothesis: the translation adapter improves heldout prompt-bank reconstruction
without changing native generation. Null: correct readout is not improved or
not prompt-specific. This is an auxiliary/dev training run, not image-quality evidence.

## Baseline and data comparability

Baseline: zero-B initial reader and unchanged native generation, same seeds,
CFG=1, NFE=50, 512x512, shift=3, K=8, body [12,20), 2 states/rollout.
Source: user-specified `stage12_stage2_train.jsonl` under CORT adapted 20260816.
Independent source split: neighboring `cort36k_val.jsonl`, not train rewrites.
Both lack categories. Export only explicitly matched count/spatial prompts;
tags are conservative regex metadata, not human-certified semantic labels.
Exact/normalized duplicate prompts are deduplicated; every official val prompt
is excluded from training. SHA256, source lines, matched phrases, skip counts
and train/heldout counts are recorded. Evaluate first 8 of 64 exported heldout.
This narrower two-category subset is not a seven-category benchmark.

Primary keys: correct/initial/shuffled/zero heldout MSE, native parity max abs,
per-layer effective slot count and max slot mass. Acceptance is the existing
four-check warmup gate; never bypass it. Shuffled/zero are evaluation-only.

## Implementation and runtime

Add explicit synchronous adapter gradient averaging to custom forward-method
training (ordinary DDP would not wrap these calls). Each rank has B=1 and a
frozen full BAGEL; trainable A/B alone are broadcast and gradients averaged
before clipping. Rank 0 alone writes/evaluates; collectives synchronize steps.
Train global batch=8, LR=1e-4, AdamW betas .9/.95, wd=0, clip=1.
5000 optimizer steps = 40000 prompt rollouts, unlike single-card 5000 steps.
No writer/backbone/Q/gate updates; no images/CoT consumed.

## Smoke, main run, artifacts and monitoring

1. CPU suite including two-rank gloo synchronization/global-gradient check.
2. Fresh code directory, preserving existing remote checkout and GPU jobs.
3. Export data and validate-only, run remote CPU tests.
4. 8-card torchrun smoke: 2 optimizer steps, 2 heldout prompts; same main
   resolution and NFE. Require finite losses/gradients, native parity and artifacts.
5. Only after smoke succeeds, fresh 8-card 5000-step main training. Save/eval
   every 250 steps, final evaluation. Separate immutable smoke/main outputs.

Launch detached with durable PID/log/status files (tmux absent on host).
Log code commit, source data export and all resolved config. Inspect startup
and first actual optimizer update; subsequent training completion is not claimed
by a successful launch. Do not start Phase 1A.1 until warmup and teacher gates pass.

Stop on invalid data, nonfinite gradient/loss, native parity failure, missing
outputs or failed worker. Do not loosen guards. Do not kill unrelated processes.
If model/dependencies fail, diagnose the concrete error before one revised smoke.
No automatic installation or replacement of shared model weights.

Checklist: [CHECKLIST.md](CHECKLIST.md). Runtime status: [RUN.md](RUN.md).

## Revision log

2026-09-30: User authorized execution. 20470 has 8 H200; shared existing
GPU processes have ~4.2 GiB/card and 0% utilization at preflight, preserved.
Train manifest: 204154 rows, 115883 exact prompts, no categories. Official val:
3662 rows, 3660 exact prompts, 21 exact overlaps with train; exporter removes
official heldout overlap from train, does not silently mix the splits.
