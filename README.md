# Bagel Latent CoT: 等时生成态编辑

在 Ascend NPU 上使用冻结的 SenseNova-U1.5。原生文生图从噪声积分到读取点后，状态编辑算子只在 `s: 0→1` 上改变状态，原生生成时间 `t` 固定；随后由未适配的原生 T2I 在目标条件下接续到 `t=1`。

训练只适配 GEN 全层 rank-64 LoRA 和 `edit_s_embedder`。主损失是固定原生时间上的单步条件流匹配，标签来自 L/I/R 三类状态配对。没有终图反传、长程代理梯度、旧编辑 CFG 或固定回退时间。

## 安装

在配置好 PyTorch 与 torch_npu 的环境中安装：

```bash
pip install -e .
# 可选：缓存质量/评测需要 lpips，外部评审需要 openai
pip install -e ".[metrics,judge]"
```

缓存质量记录和评测需要 `lpips`，外部评审需要 `openai`。

命令行入口是 `u15-target-correction`，子命令为 `train`、`sample`、`prepare-state-pairs`、`native` 和 `judge`。`native` 显式运行冻结的原生 T2I 或 IT2I 基线，不经过状态编辑 LoRA。

## 当前合同

- 方案标识：`u15_isochronic_state_flow_v1`。
- 原生采样：512×512，50 步基础网格，shift=3，Euler，CFG=4，`t_eps=0.05`。
- 支持状态时间：0.15、0.25、0.36、0.50；部署读取点 0.36。
- 状态编辑：固定 `t`，`s` 上 8 步 Euler；训练 `s~U(0,1)`。
- 数据来源：`linear_pair`、`inverted_pair`、`native_parent`，另有约 10% 显式保持样本。
- 检查点只保存 LoRA 与 `edit_s_embedder`，区分同阶段 resume 和跨阶段权重加载。

## 文档

- [实现合同](docs/target_correction.md)
- [训练](docs/train.md)
- [推理](docs/inference.md)
- [等时生成态编辑方案](docs/isochronic_state_flow_training_plan.md)
- [历史方案：选择性语义修复](docs/selective_semantic_correction_plan.md)
- [历史结果：E1 记录](docs/e1_20261105_results_and_training.md)
