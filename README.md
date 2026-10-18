# BAGEL-LatentCoT

通过连续 memory 连接 BAGEL 的理解与生成，让模型在图像生成过程中读取状态、形成反馈并更新生成预测。

生成专家（GEN）先计算当前状态的内部特征。理解专家（UND）结合目标文字写入连续 memory，第二遍 GEN 消费反馈，重新预测同一状态的速度。外层采样器据此推进图像。

训练路线为 **Observe 初始化 → 交互监督微调（SFT）→ Flow-GRPO 强化学习（RL）**。SFT 联合训练 UND writer 与 GEN consumer，RL 根据终图奖励更新 consumer。

[方法设计](docs/design.md) · [实验与运行](docs/experiments.md) · [训练数据](docs/data.md) · [Benchmark](docs/benchmark.md)

## 当前进展

| 阶段 | 结果 |
|---|---|
| Observe | 原生 UND 全层注意力训练完成 20,000 步；高噪声课程与多机训练入口已实现 |
| 交互 SFT | 双 rank-16 LoRA 已接入；真实 BAGEL 单卡 2 步、16 卡 20 步验证通过 |
| Flow-GRPO | 14 张策略卡与 2 张评分卡完成 32 张候选生成、评分、一次参数更新和重载生成 |
| 统一推理 | 支持原版 BAGEL、parent、正常反馈、屏蔽反馈和交换反馈 |

2026-10-17 的 16 卡 SFT 记录为平均 1.569 秒/步、峰值显存 38.406 GiB。运行目录、训练参数与阶段结果见[实验文档](docs/experiments.md#阶段结果)。下一轮训练预算为 500 步 SFT 与 100 批 RL。

## 开始使用

准备 Python 3.10+、与设备匹配的 PyTorch 和 BAGEL 基础模型。在仓库根目录安装：

```bash
python -m pip install -e '.[dev]'
python -m qwen_latent_cot.interaction.cli --help
```

交互训练读取 caption-image JSONL：

```json
{"caption":"Three red apples on a white plate.","image":"/absolute/path/to/image.png"}
```

数据准备见[训练数据](docs/data.md)，Observe、SFT 与 RL 启动命令见[实验与运行](docs/experiments.md)。

使用交互 checkpoint 生成图像：

```bash
export BAGEL_MODEL=/absolute/path/to/Bagel-7B-MoT
export INTERACTION_CHECKPOINT=/absolute/path/to/step_000500.pt
python -m qwen_latent_cot.interaction.cli sample \
  --model "$BAGEL_MODEL" --checkpoint "$INTERACTION_CHECKPOINT" \
  --caption 'Three red apples on a white plate.' \
  --device npu:0 --seed 0 --output sample.png
```

输出为 `sample.png` 与采样节点记录 `sample.png.json`。现有训练脚本面向 Ascend NPU，模型与环境路径通过环境变量设置。

## 许可

基于 [BAGEL](https://github.com/ByteDance-Seed/Bagel) 的理解、生成与流匹配接口。代码采用 [Apache-2.0](LICENSE)，模型与数据遵循各自许可证。
