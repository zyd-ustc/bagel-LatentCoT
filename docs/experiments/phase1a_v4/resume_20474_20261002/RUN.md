# 20474 Phase1A.0 continuation launch record

## Identity and evidence boundary

User authorized eight-card continuation on 2026-10-02. This is Reader Warm-up,
not Self-CoT OPD. Final target is total step 5000, starting from saved step 750.
The old run and its checkpoint are preserved. Legacy checkpoint lacks complete
RNG state: optimizer is restored, but bitwise-identical continuation is not claimed.

- Host: `dedicated-developjob-js-public-huvlx`, SSH port `20474`.
- Resources: H200 CUDA `0,1,2,3,4,5,6,7`; effective global batch 8.
- Session: `phase1a0_resume750_20474`.
- Root: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000_resume750`.
- Parent: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000/main/reader_warmup_step_0000750.safetensors`.

## Timeline (UTC)

- 2026-10-02 14:37:44: all eight GPUs idle (1 MiB used each), no GPU compute jobs.
- CPU preflight: passed, restored step 750/world 8/AdamW/config/data contract.
- 2026-10-02 14:44:54: persistent tmux launcher created.
- 2026-10-02 14:45:00: two-update pilot process launched; model loading began.
- 2026-10-02 14:47:14: eight worker processes alive; model loading still active,
  about 52 GiB CPU RSS per worker. No optimizer-update success claim yet.
- 2026-10-02 14:49:31: pilot complete at step 752. Eight-rank global losses
  751=2.4411004409193993, 752=2.606482081115246; native parity max abs 0.
  Original step-zero heldout baseline retained exactly; AdamW step counters 752,
  complete checkpoint and RNG resume sidecars verified. GPUs released after pilot.
- 2026-10-02 14:49:41: formal eight-rank continuation launched in the same tmux
  session, using original step 750 and final total 5000; model loading began.
- 2026-10-02 14:53:02: formal step 751 completed, loss 2.4411004409193993.
  All eight GPUs computing, approximately 32.5 GiB used/card.
- Handoff verification: formal status running/773, 23 actual updates 751–773;
  latest global loss 2.321913793683052, gradient norm 2.506894826889038.
  All 23 losses/gradient norms finite, eight prompt IDs/update, eight workers alive.
  Manifest confirms start 750, world/global batch 8, mean gradients before
  clipping; resolved config confirms final total 5000.

## Artifacts

Remote root contains `launch.sh`, `PLAN.md`, `CHECKLIST.md`, `launcher.log`,
`preflight.json`, `gpu_preflight.txt`, `package_freeze.txt`, `input_sha256.txt`,
and exact pilot command `smoke_command.txt`. Pilot stdout is `smoke.log`.
The launcher checks completion, finite eight-rank updates, native parity,
optimizer counters and new RNG sidecars before formal launch. Formal training
restarted from the original step 750, not the pilot's step 752. Main stdout is
`train.log`; `main/status.json`, `main/metrics.jsonl` and `main/run_manifest.json`
establish actual continuation progress rather than process-start-only evidence.

## Current state

Completed. Inspected on 2026-10-03 at 12:37 Shanghai time: status `complete`,
step 5000; launcher reports successful exit, no warm-up workers remain.
`status.json` was finalized at 03:37:09 Shanghai time; launcher exited at 03:37:19.
There are 4250 contiguous resumed updates (751–5000); all recorded losses and
gradient norms are finite. Final global loss is 1.5403026044368744; gradient norm
is 1.072780728340149. No automatic OPD launch. Semantic generation quality is
not established by this training result.

## Final heldout reconstruction

Eight prompts evaluated, unchanged from the source evaluation contract. At step 5000:

| Bank condition | Readout MSE |
|---|---:|
| correct | 2.582562506198883 |
| shuffled | 3.0956303775310516 |
| zero | 2.7046388387680054 |

Native velocity parity max abs is 0; all original operational gate checks passed.
Correct MSE is about 4.5% below zero in this evaluated prefix. No new statistical
significance analysis or semantic image evaluation was performed for this record.

Final checkpoint:
`/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000_resume750/main/reader_warmup_step_0005000.safetensors`.
Its `.json`, `.optimizer.pt`, `.resume.pt` and `.resume.json` files are present.
Artifacts remain on the H200 shared filesystem; weights and datasets are not in Git.
