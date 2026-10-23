# Cross-time MoT v3 run

Implementation authorized2026-10-23. See PLAN.md for exact training/trajectory contract. New code /data/zyd_workspace/bagel-LatentCoT/code/bagel-mot-cross-time-8000-20261023; new run /data/zyd_workspace/bagel-LatentCoT/outputs/bagel_mot_cross_time_64npu_20261023_8000. Same Python/model/corpus as v2. Old monitor deleted before controlled stop. No training launched before full NPU pilot acceptance.

CPU seven v2 regression checks passed, five new cross-time checks passed: shared-reader last-layerQ/O gradients, prompt/native conditional entry parity without feedback, persistent replay/count/rollback,K0 isolation,three role gradients,exact resumed optimizer/sampler/RNG and retained caption metrics. Logs artifacts/mot_cross_time_8000_20261023/{cpu_tests.log,v2_regression.log}.

## Previous run stopped; pilot pending

Four nodes audited with16workers each, launcher identity and parent ownership verified. Torchrun gives each worker its own process group; the first group-equality audit sent no signals. After verifying the actual worker→torchrun→supervisor chain, sent SIGTERM only to each owned torchrun launcher. All four report zero remaining workers; old logs/checkpoints preserved. Stop evidence artifacts/mot_cross_time_8000_20261023/old_stop_verified.json. Previous monitor removed. NPU pilot launch uses isolated output bagel_mot_cross_time_20261023_pilot,NPU15; acceptance pending. Main source initially pinned8d19307.

## Full512 NPU pilot accepted,13:12 CST

Pinned source8d19307. Isolated NPU15 pilot completed1caption+2edit updates,exit0,checkpoint3 reloaded. Peak29.262GiB. Edit updates26.767/26.432seconds,loss0.597492/0.572223 vs same-state zero0.594253/0.571359. Last role gradients observation1.609146,feedback0.000712,editor0.034058 (preclip; total1.609507,clip1). Caption CE retained4.899872/5.059622. Applied negative fraction0. One-step caption output repeats punctuation, not an aligned reader; no pilot adapters/optimizer reused.

Downloaded and actually viewed both native andtwo-round512 PNGs: red cube andblue ball intact on wooden table,no colored grid/background breakup. Both show onecube although goal asks two; no semantic correction established after twoedit updates. Trace42persistent feedback steps,2exact rollback restores(anchor_error_rms0),91Euler total. delta/baseRMS0.00715–0.00928,minimum guidance coefficient−1.685e−8. Complete artifacts in local pilot/ andpilot.log. New64rank training now authorized to launch on fixed configuration. Pilot suggests ~27seconds/edit update before64-rank synchronization;7000edit updates alone roughly52hours,actual distributed throughput to follow.
