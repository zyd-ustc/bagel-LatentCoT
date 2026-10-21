# 八组消融：运行与结果

固定 Stage A step4000、seed17。GenEval2 最大 atom_count 档、GenEval、DPG、DSG 各8条，
共32条原始 Prompt、8组、256张图。对照包括原版、Parent 无循环、正常一次/两次反馈、
Memory屏蔽、回灌关闭，以及 CFG1 的 Parent/一次反馈。

- [对照定义及解释边界](PLAN.md)
- [本次结果与图像抽查](RESULTS.md)
- [完成情况](CHECKLIST.md)

## 在 modelarts-job 的仓库根目录运行

需要空闲的16张NPU、BAGEL模型、Stage A checkpoint，以及已有 `/root/npu-eval-tools` 的原始题目。
脚本不停止其他服务。使用新的输出目录，不覆盖已经完成的实验。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python
RUN=/data/zyd_workspace/bagel-LatentCoT/outputs/loop_ablation_step4000_repeat
"$PYTHON" scripts/sample_loop_ablation.py prepare --output "$RUN" \
  --checkpoint /data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_64npu_20261020_20000/step_004000.pt
bash scripts/loop_ablation_16.sh "$RUN"
```

`prepare` 固定 checkpoint 和选样，并保留全部原始标注。
GenEval 覆盖六类，并对 counting、position 各多取一条；其他题库固定 seed17抽样。
每张图保存控制项、逐步遍历记录及CFG前的条件速度差异幅度。全部成功后写出离线 `index.html` 和 `summary.json`。

只重新生成页面，不重新生图：

```bash
"$PYTHON" scripts/sample_loop_ablation.py render --output "$RUN"
```

本次结果位于 `/data/zyd_workspace/bagel-LatentCoT/outputs/loop_ablation_step4000_20261021`。
这是小规模诊断，不包含官方 benchmark 分数。图片和 checkpoint 不随代码提交。
