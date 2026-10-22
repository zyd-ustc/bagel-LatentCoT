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
