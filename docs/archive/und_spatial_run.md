# UND spatial correction：64 卡启动记录

2026-10-21，modelarts-64。训练正在运行；以下为启动验收时的快照。

## 已部署

- 分支：`loop-layer-NPU`。
- 训练代码：`cd56200cd7d204b3c8014eec45eeb80dffb9f394`。
- 模型：原生 BAGEL GEN 冻结；原生预训练 UND 独立副本，Q/K/V/O rank-64 LoRA，空间输入投影和零初始化输出 head。
- 唯一损失：`MSE(delta, epsilon-x0-v_native)`。
- 删除旧 diagnostics/interaction、Memory、consumer、回灌、semantic、partial Prompt、Teacher 及其测试和脚本。历史模型、数据和实验图片保留。
- 训练不从 C2 或旧 Stage A 续训。

## 运行位置

- 代码：`/data/zyd_workspace/bagel-LatentCoT/code/bagel-und-spatial-main-20261021`。
- 训练：`/data/zyd_workspace/bagel-LatentCoT/outputs/und_spatial_64npu_20261021_1000`。
- 启动日志和各节点 PID：训练路径加 `_launch`。
- 语料：`/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus`，仅使用 generation views，保留已有 family 划分和完整 caption。
- 4 节点 × 16 NPU；world=64，全局 batch 256，逐 rank microbatch 1，累计 4 次。
- 总预算 1,000 步；512px；lr 1e-4；warmup 100；clip 1；纯 MSE；activation checkpointing 关闭。
- 可训练参数：53,444,672。

## 已取得的证据

10 个真实小型 MoT CPU 测试通过，包括精确训练续跑。完整模型 NPU pilot 通过 4 次更新、原生冻结、零输出初始化、UND 梯度、检查点恢复、优化器/采样器/RNG 续训，以及两组完整生图。人工检查了 pilot 图片，两组均生成了可辨认场景；pilot 用于执行验收，不作为质量提升结论。

4 个节点的审查均未发现残留旧训练。64 个 rank 已全部载入并参与全局梯度同步。Step 10 检查点已保存。

启动快照：Step 84，loss 0.415764，同批原生 MSE 0.449157，delta RMS 0.138581，UND 梯度范数 0.007293。最近 20 步平均 2.329 秒/步，峰值显存 49.08 GiB。数值均有限。

计划在 Step 100、500、1000 保存并运行固定验证样本的原生/纠错生图。当前状态为“重构并启动训练已完成”，不是“1,000 步训练已完成”。

本地详细证据：`artifacts/und_spatial_refactor_20261021/`；CPU 测试日志：`artifacts/und_spatial_cpu_tests.log`。
