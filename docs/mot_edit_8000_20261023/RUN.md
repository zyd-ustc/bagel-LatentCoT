# MoT edit8000 run

2026-10-23: user authorized new implementation and64-NPU8000-step run. Branch and fixed recipe in PLAN.md. Remote source JSONL exists,606433 rows,2,517,077,158 bytes. It has uid/image/prompt/benchmark/source_file/source_line. Existing27B server from completed base benchmark still occupies logical NPU0/1; its recovery configuration exists. No new training has launched yet.

## Implementation and first checks

Added role-local shared-base Q/K/V/O LoRA, clean-proposal GEN observation, goal-free UND observation, goal-aware UND feedback and private GEN editor with native llm2vae. No new output head. CPU real tiny-MoT checks passed: role gradients, frozen native parameters, projected fixed-anchor round closure, K0 native isolation, checkpointed forward, stage switch and exact optimizer/sampler/RNG resume.

Initial NPU pilot loaded the full native model, then stopped before training because the concurrently building corpus index had not finalized manifest.json. No optimizer updates occurred. Preserve `outputs/mot_edit_pilot_20261023.log`; rerun waits for the manifest dependency before loading the model. Index construction continues normally.

Monitoring uses existing tool/runtime paths and durable run files. No bash_exec/artifact/memory runtime is available in this harness.

## Full512 NPU pilot passed

Real native model, NPU15,1 caption update +2 two-round editing updates. Caption loss4.900319, observation gradient0.372362. Edit update2: loss1.506909, feedback gradient0.034617, GEN gradient0.190310; update3: loss1.342860, feedback0.077492, GEN0.436807. All finite; applied negative fraction0. Peak28.97GiB. Stage switch, trainable-only checkpoints and checkpoint reloading passed. Single-card edit updates2.12–2.20s; this excludes64-rank stragglers/communication and is not a main-run ETA.

Downloaded and actually inspected native/two-round pilot images. Both show a red cube left and a blue ball right on a wooden table, with intact photographic style. Prompt asks two cubes; both images have one. Two edit updates have not fixed that count. Artifacts: `artifacts/mot_edit_8000_20261023/pilot/`. Initial pilot failure and retry logs retained separately.

Corpus finalized:606433 indexed pairs; train594454,val6058,test5921; zero duplicate/empty exclusions. Identity `a226556c-7479-43e9-8173-45b4721dd4bc`. Data path coverage/128-image decode audit runs read-only before main launch.

## Data and resource preflight

Full606433 image-path coverage passed:0 missing files.128 deterministic sampled images fully decoded with0 errors. Audit JSON: remote `outputs/mot_edit_data_audit_20261023.json`. All rows retained in the index.

Dedicated base-benchmark judge namespace confirmed idle (`inflight=0`), six scoring cells ended and no running/waiting cells. Saved current service argv/cwd/allowed runtime environment, then released only proxy1086311 and backend1084918 process groups. Recovery information: `outputs/mot_edit_judge_release_20261023.json`. No unrelated process was stopped.

## Main launch and scheduled checks

2026-10-23 02:52 Asia/Shanghai: four-node launcher succeeded, all node resource audits empty, each16 ranks requested. Run revision `b5f4241` (implementation `05d81d8`). Detached supervisor PIDs:node0=1164727,node1=718756,node2=717283,node3=718402. Pinned run and code paths in PLAN.md; launch logs at run path plus `_launch`. Initial launch snapshot is in local artifacts/status_initial.json; worker initialization still in progress at that snapshot.

Heartbeat `mot-64-8000` created every20minutes, quiet on healthy unchanged state. It follows observation1000 and semantic1100/2000/4000/8000, downloads actual images and records, and reports meaningful changes/failure/completion. It preserves8000 total steps and the selected data/objective. Local gallery entry generator: `scripts/mot_edit_gallery.py`.
