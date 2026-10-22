# Guidance-safe 64-NPU restart — 2026-10-22

## Fixed run

- Host: `modelarts-64`; four nodes × sixteen NPUs; fresh native initialization, not old step5000 continuation.
- Launch: 2026-10-22 around 13:12 Asia/Shanghai.
- Executed revision: `8d8004d9125f32ec314510af20204e2450f3b076`, pushed to `loop-layer-NPU`.
- Remote code: `/data/zyd_workspace/bagel-LatentCoT/code/bagel-guidance-safe-20261022`.
- Remote output: `/data/zyd_workspace/bagel-LatentCoT/outputs/und_guidance_safe_64npu_20261022_5000`.
- Launch logs/status: same output path plus `_launch`; `node0.log` through `node3.log`, `status.py`, `run.json`, `launch.json`.
- Python: `/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python`.
- Config: `configs/correction_guidance_safe.json`; 5000 steps, batch256, 512px, lr1e-4, warmup100, seed17, CFG4, native global renormalization, first25/50 updates active.
- Corpus: `/data/zyd_workspace/bagel-LatentCoT/datasets/correction_photos_v2_20261022`; same accepted photo pairs and image-disjoint split.
- Preserve UND architecture; do not implement GEN velocity output.

## Objective and interpretation

See [PLAN.md](PLAN.md) for the fixed formula. Train raw UND prediction against the conditional residual with its negative guidance component removed. Apply the same removal at sampling, after native CFG/renormalization. Global uses the same projected target and deployment guard.

`loss` and `zero_correction_mse` share the new target. `base_mse` and `applied_base_mse` remain physical guided-flow errors; do not compare their scale to `loss` as a percentage improvement. `raw_negative_fraction` may be positive during learning. `applied_negative_fraction` must be zero within the logged coefficient tolerance. Target removal changes the regression task; success still requires actual image quality and semantic checks.

## Acceptance evidence

- All21 CPU tests passed, including projection geometry, zero guidance, target gradients, raw prediction gradient, native parity, old-config semantics, exact new-mode resume, and three-arm validation. Log: local `artifacts/guidance_safe_20261022/cpu_tests.log`.
- Full native NPU pilot: four updates including checkpoint3 restore and update4 continuation; head then UND gradients, frozen native, optimizer/sampler/RNG restore and native/global/corrected sampling passed.
- Pilot actual images inspected: all retain recognizable photographic rendering. Native has two dogs; corrected has three; global has more dogs and limb artifacts. Four-step pilot is execution evidence, not semantic improvement evidence.
- Pilot applied negative fraction0 at all four steps; sampled applied coefficients passed >=−1e−5. Pilot peak48.02GiB.
- Pilot files downloaded to local `artifacts/guidance_safe_20261022/pilot`; remote `outputs/und_guidance_safe_pilot_20261022`.
- Four-node launch returned success; launch PIDs node0=964508, node1=635368, node2=634075, node3=635194. Audit found no obsolete or active correction jobs before launch.

## Follow-up and completion

Read the pinned `status.py` snapshot, four node logs, finite values, saved checkpoints and validation records. Download newly completed semantic100/1000/3000/5000 sets to local `artifacts/guidance_safe_20261022`, compare native/global/corrected actual images, and record quality/semantic observations here. Do not equate MSE with semantic success.

Required completion: step5000, `step_005000.pt`, `validation_005000/records.json`, and `semantic_005000/records.json`. Inspect all32 final semantic cases and report quality separately from content accuracy. Preserve old artifacts. Do not change data/objective/budget or implement GEN output during monitoring.

For a genuine interruption, retain logs and checkpoints. CLI `train --resume <checkpoint>` supports exact optimizer/sampler/per-rank RNG/calibration restore with the same config and64-rank layout. The fresh-run launcher rejects an existing output path; do not delete that path to restart. Resolve the specific failure before arranging resumed workers and separate logs.

## Launch health

At the launch acceptance snapshot, step18 was running. All four nodes logged16 loaded ranks (64 total), with no error tails. All logged metrics were finite. Step18 loss=0.422767 versus matched zero_correction_mse=0.454614; this is training-fit evidence only. Applied negative fraction was0 at every logged step; raw negative fraction at step18 was0.589844. UND gradient=0.019488, so the trainable UND adapters receive gradients. Peak48.16GiB/NPU; recent steps about2.5–2.7seconds excluding initial startup and checkpoint/validation work.

Checkpoint10 loaded successfully on CPU: correct new objective,64 per-rank RNG states,25 calibration nodes,2560 accumulated training observations. The prior run/checkpoints remain intact.

Heartbeat `guidance-safe-64-5000` is active every30minutes and stays quiet during healthy unchanged state. It will inspect milestone images, record findings and report completion/failure. Initial compute estimate is about4hours including scheduled validation, roughly17:15 Asia/Shanghai; this is an estimate, not a completion claim.


## 13:43 follow-up — Step650, Step100 images inspected

All64 ranks remain loaded, no four-node error tails, all logged values finite. Checkpoints10/100/500 exist. Step650 loss0.425869 vs matched zero0.462656; applied negative fraction remained0 across all logged updates. Raw negative fraction0.59375 shows deployment projection still removes learned negative components. Recent20 updates average2.647s.

Downloaded semantic100 (32 prompts × three arms, including cached native images), validation100 records, training records and four node logs. Local gallery: `artifacts/guidance_safe_20261022/run/semantic_000100/index.html`; four contact sheets and per-case `visual_notes.json` accompany it. All32 cases inspected visually. This is an early qualitative snapshot, not a benchmark score.

Observed: blowdryer08 and keyboard/microwave09 remain recognizable; face19 remains intact; train24 and desert28 retain their scene types instead of the prior run's destructive changes. However corrected08/09/10/14 still introduce vivid colored lighting/texture, horse15 remains on the wrong side and becomes toy-like, pose29 changes toward upright with extra background people, and difficult counts/precise text remain unreliable. Some apparent fixes (two sheep00, single treadmill23) also occur in global. The new objective has not yet demonstrated UND-specific semantic gains or eliminated style drift. Comparisons to oldstep5000 are descriptive across different training stages, not a matched-step causal ablation.

Validation100 across32 paired states: native physical guided MSE0.792692, global0.809923, corrected0.717277; objective MSE0.414917 is a different target. Latest100 downloaded training rows average loss0.418616 vs matched zero0.451965. None of these values measures semantic success. Continue the fixed run; no change to data, objective, budget or GEN architecture.


## 14:39 follow-up — Step1876, Step1000 images inspected

Pinned status: all64 ranks loaded, four logs without error tails, finite training values. Step1876 loss0.413773 vs matched zero0.476331, applied negative fraction0 across the run, raw negative fraction0.710938; checkpoints10/100/500/1000 exist. Recent20 steps average2.616s. No budget, data, loss, architecture or process changes.

Downloaded semantic1000 (32×three-arm records and generated images; native cache already local), validation1000, train.jsonl (snapshot through1890) and four node logs. Readable HTML and four contact sheets at `artifacts/guidance_safe_20261022/run/semantic_001000`; all32 cases visually inspected, with per-case notes in visual_notes.json. Full-resolution inspection of books10 and treadmill23 supplements contact sheets.

Objects and faces remain recognizable in08/09/19/21, and24/28 retain train/desert scene types. Style drift persists:08 has strong red/blue neon,15 turns the horse toy-like while leaving it on the wrong side, and13 loses the purple-pizza attribute. At23, corrected has two treadmills again (step100 had one; global1000 has one), so that apparent improvement is not stable. Books10 has three main book bodies but ambiguous thin edges/layers; it is closer to the requested count than native, not a formal success score. Hard multi-object counts, precise text17/30 and pose27/29 remain unreliable. No stable overall UND-specific semantic improvement is established at1000.

Validation1000 physical guided MSE: native0.792692, global0.814135, corrected0.733595 (corrected100 was0.717277). New-objective MSE0.425636 vs matched zero0.494856. Latest100 downloaded training rows: loss0.398927 vs zero0.450404; applied negative fraction0. Sample traces cover800 active states per arm: raw negative coefficient88.875% for corrected and77.75% for global; minimum applied coefficient−1.92e−8/−1.98e−8, respectively, within numerical projection tolerance. Removing the negative direction works geometrically, but has not removed all visual/style problems. Continue the fixed5000-step run.
