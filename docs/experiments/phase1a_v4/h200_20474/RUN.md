# Phase 1A.0 fresh restart: 20474

## Research question and objective

Same frozen-writer side-head reader warm-up as 20470, now on idle H200 cards.
Auxiliary/dev training; no image-quality or semantic claim from launch alone.
No resume implementation or checkpoint load; seed 42 and fresh A/B initialization.

## Setup and execution

- SSH: `ssh -p 20474 root@vr.turbo-ai.com`.
- Host: `dedicated-developjob-js-public-huvlx`; eight H200, CUDA 0–7.
- Code: `/private/yida_workspace/bagel-LatentCoT-phase1a-v4-reader-8gpu-20260930-r2`.
- Immutable deployed code: `4148a70bfd318d506a2b261ca9aaa56b8de31075`.
- Root: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000`.
- Main: `main/`; log: `train.log`; command: `command.txt`; launcher: `launch.sh`.
- Python: `/private/software/conda/envs/lcot/bin/python`, torch2.5.1+cu124.
- Existing shared exported data reused; hashes in PLAN.md and remote input_sha256.txt.
- 5000 steps, B=1/rank/global8, eval/save250, unchanged native generation.
- K8/body[12,20), 512x512, NFE50, shift3, CFG1, two states per rollout.

## Restart event

2026-09-30 11:52:40 UTC: old 20470 run stopped by explicit user request.
Scoped launcher/torchrun/eight worker command lines verified before signals.
All GPU-owning workers gone; unrelated tasks still occupy ~83 GiB/card.
Old last completed step: 208; loss_reader_mse=4.834789142012596,
grad_norm(pre-clipping)=2.331719398498535. No new checkpoint was saved.
Original metrics/logs/status preserved; interruption marker:
`/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/interrupted_for_20474.json`.
The original status.json still says running/208: marker and process state are
authoritative for the interruption; original status was not rewritten.

2026-09-30 ~11:53 UTC: direct fresh 8-card main launched on 20474.
Torchrun PID: `2641732`. Validation-only succeeds; initial loading in progress.
Prior identical-code eight-card smoke establishes executability, not semantic merit.

## Results, analysis and conclusion

2026-09-30 11:57:56 UTC: 10 actual optimizer updates recorded, status running/10.
All recorded losses/gradient norms finite, eight scoped workers alive.
Latest loss=6.038202449679375; pre-clipping gradient norm=0.15764881670475006.
Initial 8-heldout native parity max abs=0, correct MSE=7.139190256595612,
shuffled MSE=7.515929102897644, zero MSE=2.7046388387680054.
Zero target diagnostic remains better than correct at initialization; no
semantic usefulness claim is made. First two losses reproduce prior identical
seed/config smoke: 6.685308516025543 and 6.424177944660187.
All eight cards use about 33 GB and show 82–84% utilization at 11:57:32 UTC.
Detailed durable evidence: remote `startup_verified.json`, `main/run_manifest.json`,
`main/metrics.jsonl`, `main/heldout_diagnostics.jsonl`.

Launch is verified, not full training completion or gate passage.
Next scheduled checkpoint: training's step-250 eval/save; full 5000-step validation
remains pending. Do not auto-start OPD.
