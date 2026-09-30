# Phase 1A.0 fresh restart on H200 20474

## 1. Objective

Run ID: `phase1a0_reader8_20474_20260930_115000`.
User explicitly requests no resume implementation: stop our 20470 job and
restart from zero on 20474. Preserve every old artifact and unrelated GPU job.
Train the frozen-writer side-head reader; do not start OPD or inject into GEN.
Null: heldout correct reconstruction does not improve over initial reader.
Alternative: reconstruction improves with prompt-specific memory and native parity.
This is auxiliary/dev training, not an image-quality or semantic utility claim.

## 2. Baseline and comparability

Identical model, code `4148a70bfd318d506a2b261ca9aaa56b8de31075`, seed 42,
train/heldout exports and optimizer protocol as the successful 20470 smoke.
Reuse shared exports in `phase1a0_reader8_20260930_052322_retry2/data`:
24547 train prompts, 64 independent heldout prompts; evaluate first 8 heldout.
Train SHA256: `68d976705045ada5425ef62dd197ddf7f4112d61632ec1809069b73cccfd5df9`.
Heldout SHA256: `c99d2dc3430b394d499d03bece85c4e211d679ca23f88dc3e152010842bd0f0e`.
Baseline is zero-B initialization and unchanged native generation.
Required diagnostics: correct/initial/shuffled/zero MSE, native parity max abs,
per-layer effective slot count/max slot mass and existing four-check gate.
Prior smoke zero-MSE beat correct-MSE: smoke establishes executability only.

## 3. Code translation plan

No model or training code edits. Dedicated existing run branch:
`codex/run/phase1a0-h2008-20260930`. Only execution notes and launcher added.
Use existing shared code and weights; no upload, export or split change.

## 4. Execution design

Prior 8-card 2-step smoke passed, native parity max abs = 0; reuse this evidence
because code/model/data/runtime are unchanged. Validate target paths/config.
Fresh main: 8 H200, B=1/rank, global B=8, 5000 updates, eval/save every 250.
K=8, body [12,20), 512x512, NFE50, shift3, CFG1, two states/rollout.
FP32 A/B only; frozen writer/backbone; LR1e-4, betas .9/.95, wd0, clip1.
Expected outputs: resolved config, metrics.jsonl, status.json, adapter safetensors,
heldout reports and warmup_gate.json. No checkpoint is loaded from the old run.
Stop on worker failure, nonfinite metrics, native parity failure or invalid data.
Do not relax guards. Strongest alternative: reconstruction exploits low-energy
teacher statistics rather than semantic memory, assessed by zero/shuffled controls.

## 5. Runtime strategy

Exact command: `launch.sh` in this directory, copied to the fresh remote root.
Remote root: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000`.
Main artifacts: `main/`; log: `train.log`; launcher PID: `train.pid`.
Python: `/private/software/conda/envs/lcot/bin/python` (torch2.5.1+cu124).
Preflight: 20474 has 8 idle H200, each 143167 MiB free; shared data hashes match.
Budget: existing 5000-update contract, 40000 prompt rollouts. Initial loading
estimate 5–8 minutes; step timing and total ETA measured after startup.
Inspect at 60-second intervals during launch, then hand off durable log commands.
Preserve config/global batch/precision; only host contention changes.
Specialized experiment tools unavailable; normal SSH execution with durable files.

## 6. Fallbacks and recovery

If target resources change, do not consume unrelated cards or auto-migrate.
If launch fails, retain new outputs and diagnose the concrete error before retry.
If old process is not fully stopped, do not launch duplicate training.
Do not implement resume or silently restart without user direction.

## 7. Checklist

See [CHECKLIST.md](CHECKLIST.md). Next: scoped stop, direct fresh launch.

## 8. Revision log

2026-09-30 11:50 UTC: user requests fresh restart on idle 20474 instead of
resume on contended 20470. Algorithm, data and budget unchanged.
