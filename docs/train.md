# 目标驱动纠偏训练指南

本文档介绍两阶段目标驱动纠偏的训练流程。详细参数和数据格式见 [目标驱动纠偏实现规范](target_correction.md)。

## 训练阶段设计

训练分为两阶段，分别解决局部编辑速度拟合和完整轨迹端到端对齐问题。

### Stage 1：残差速度预训练

Stage 1 训练模型学习在给定草图和修正指令下的局部修改速度场。

1. **输入与起始点**：使用双点线性桥在历史回退点与目标图像之间插值生成状态。起始时间在深回退时间 0.07 与浅回退时间 0.15 之间交替选取。
2. **监督目标**：模型拟合从起始状态到目标图像的恒定解析速度。窗口衰减函数在此阶段不生效。
3. **数据比例**：使用 80% 的外部高质量编辑对（Tier A）、15% 的真实失败轨迹对（Tier B）和 5% 的零残差保持样本（Zero）。
4. **超参数**：全局批大小为 256，学习率为 0.0001，训练 6000 步。

该阶段检查点仅作为 Stage 2 的权重初始化来源，不能直接用于推理部署。

### Stage 2：真实轨迹端到端监督

Stage 2 解决交接累积误差问题，确保模型在真实回退状态下积分后，经由原生尾段能生成满足编辑目标的完整图像。

1. **确定性历史回退**：使用 `rho = 0.0`，直接从真实前向保存的历史状态恢复生成，避免噪声扰动破坏已有结构。
2. **混合网格展开**：
   - 前向过程：沿部署使用的 Heun 细网格积分。在纠偏窗口从回退点积分至时间 0.50，随后由冻结的原生生成器继续积分至成图时间 1.0。
   - 反向过程：纠偏窗口采用 5 步 Euler 积分回传梯度，原生尾段采用 3 步 Euler 积分提供代理雅可比。
3. **监督目标**：最终成图与目标图像之间的复合结构损失，包含冻结理解模块提取的多层图像特征距离、4 倍下采样图像损失以及非编辑区域保持损失。
4. **数据比例**：使用 20% 的 Tier A 样本、70% 的 Tier B 样本和 10% 的 Zero 样本。
5. **超参数**：全局批大小为 128，学习率为 0.00003，训练 4000 步。从 Stage 1 检查点载入权重后重置优化器状态。

## 执行命令

### 启动 Stage 1

```bash
python -m qwen_latent_cot.u15_target_correction.cli train \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_proposal_correction/target_stage1.json \
  --corpus "$CORPUS" --output "$STAGE1_OUT" --device npu:0
```

### 启动 Stage 2

```bash
python -m qwen_latent_cot.u15_target_correction.cli train \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_proposal_correction/target_stage2.json \
  --init-checkpoint "$STAGE1_OUT/step_006000" \
  --corpus "$CORPUS" --output "$STAGE2_OUT" --device npu:0
```

多卡分布式训练在命令末尾添加 `--distributed`。
