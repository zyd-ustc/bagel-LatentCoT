# Cross-time MoT v3 run

Implementation authorized2026-10-23. See PLAN.md for exact training/trajectory contract. New code /data/zyd_workspace/bagel-LatentCoT/code/bagel-mot-cross-time-8000-20261023; new run /data/zyd_workspace/bagel-LatentCoT/outputs/bagel_mot_cross_time_64npu_20261023_8000. Same Python/model/corpus as v2. Old monitor deleted before controlled stop. No training launched before full NPU pilot acceptance.

CPU seven v2 regression checks passed, five new cross-time checks passed: shared-reader last-layerQ/O gradients, prompt/native conditional entry parity without feedback, persistent replay/count/rollback,K0 isolation,three role gradients,exact resumed optimizer/sampler/RNG and retained caption metrics. Logs artifacts/mot_cross_time_8000_20261023/{cpu_tests.log,v2_regression.log}.

## Previous run stopped; pilot pending

Four nodes audited with16workers each, launcher identity and parent ownership verified. Torchrun gives each worker its own process group; the first group-equality audit sent no signals. After verifying the actual worker→torchrun→supervisor chain, sent SIGTERM only to each owned torchrun launcher. All four report zero remaining workers; old logs/checkpoints preserved. Stop evidence artifacts/mot_cross_time_8000_20261023/old_stop_verified.json. Previous monitor removed. NPU pilot launch uses isolated output bagel_mot_cross_time_20261023_pilot,NPU15; acceptance pending. Main source initially pinned8d19307.

## Full512 NPU pilot accepted,13:12 CST

Pinned source8d19307. Isolated NPU15 pilot completed1caption+2edit updates,exit0,checkpoint3 reloaded. Peak29.262GiB. Edit updates26.767/26.432seconds,loss0.597492/0.572223 vs same-state zero0.594253/0.571359. Last role gradients observation1.609146,feedback0.000712,editor0.034058 (preclip; total1.609507,clip1). Caption CE retained4.899872/5.059622. Applied negative fraction0. One-step caption output repeats punctuation, not an aligned reader; no pilot adapters/optimizer reused.

Downloaded and actually viewed both native andtwo-round512 PNGs: red cube andblue ball intact on wooden table,no colored grid/background breakup. Both show onecube although goal asks two; no semantic correction established after twoedit updates. Trace42persistent feedback steps,2exact rollback restores(anchor_error_rms0),91Euler total. delta/baseRMS0.00715–0.00928,minimum guidance coefficient−1.685e−8. Complete artifacts in local pilot/ andpilot.log. New64rank training now authorized to launch on fixed configuration. Pilot suggests ~27seconds/edit update before64-rank synchronization;7000edit updates alone roughly52hours,actual distributed throughput to follow.

## Four-node launch requested

Four-node preflight passed with no occupied NPU processes or active old MoT/corrector/judge workers. Launch returned0 on all four nodes. Supervisor identities:node0=1247358,node1=788138,node2=786665,node3=787786. Pinned deployed source8d19307,master29673; config explicitlyconfigs/mot_cross_time_8000.json throughMOT_CONFIG. Mainrun uses fresh native initialization,no pilot checkpoint/optimizer reuse. Finite-step/checkpoint acceptance pending first status snapshot.

Created20-minute quiet-on-healthy heartbeat named MoT跨时间反馈64卡8000步. App reused IDmot-64-8000 after the old automation was deleted; prompt now pins only v3 paths, retained caption loss,91-step two-round sampling and detached sampled-write training. It checks actual milestone images and records,not just loss.

##64-rank launch accepted

First finite snapshot Step14 caption stage:64workers loaded/live(16/node),four logs without errors/exits/probe failures. Caption loss5.047731,observation gradient1.201051(preclip),feedback/editor0 as expected. Globalbatch64,total8000,firstcheckpoint10present;peak29.294GiB. Current update1.090s,recent14mean1.674s including startup. Snapshot artifacts/mot_cross_time_8000_20261023/status_first.json. This accepts runtime launch; no newrun milestone semantic result yet.

Checkpoint10 opened on remoteCPU:kindbagel_mot_cross_time_v3,total8000,batch64,rollback12/read33,captionweight1,64per-rankRNG entries,224optimizer state entries(caption-stage observation),672role adapter tensors. Same corpus identitya226556c-7479-43e9-8173-45b4721dd4bc,pinned source8d193073b44394878eb499f3e82cb2b48ab284fd. Metadata saved as checkpoint10_metadata.json. Later documentation/status compatibility commits do not change the active source or its DEPLOYED_COMMIT.
