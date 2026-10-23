# Cross-time MoT v3 run

Implementation authorized2026-10-23. See PLAN.md for exact training/trajectory contract. New code /data/zyd_workspace/bagel-LatentCoT/code/bagel-mot-cross-time-8000-20261023; new run /data/zyd_workspace/bagel-LatentCoT/outputs/bagel_mot_cross_time_64npu_20261023_8000. Same Python/model/corpus as v2. Old monitor deleted before controlled stop. No training launched before full NPU pilot acceptance.

CPU seven v2 regression checks passed, five new cross-time checks passed: shared-reader last-layerQ/O gradients, prompt/native conditional entry parity without feedback, persistent replay/count/rollback,K0 isolation,three role gradients,exact resumed optimizer/sampler/RNG and retained caption metrics. Logs artifacts/mot_cross_time_8000_20261023/{cpu_tests.log,v2_regression.log}.
