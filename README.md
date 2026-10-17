# BAGEL-LatentCoT

用连续 memory 连接 BAGEL 的理解与生成，在图像生成过程中读取状态、形成反馈并重新预测速度。

理解专家 UND 从目标文字和当前 GEN 表征写入 16 槽 memory，生成专家 GEN 消费反馈，外层采样器用最终速度推进图像。当前主线为 Observe 初始化 → 连续交互 SFT → Flow-GRPO。memory 的生成作用直接接受速度监督和终图奖励。

[Design · 原理](docs/design.md) · [Experiment · 实验与运行](docs/experiment.md) · [Inference Tests · 推理测试](docs/inference-tests.md)

[交互 SFT / RL 方案](docs/generation-loop-sft-rl-plan.md) · [新训练入口与验证](docs/experiment.md#7-交互生成训练)

## 当前进展

已完成显式文字 memory 推理、C1 文本接口与原生 UND Observe / Feedback 训练。交互生成使用独立的 SFT / RL 入口。显式与浅层生成的人工审查已观察到语义编辑；生成训练评估修改正确性、正确内容保持和跨 seed 稳定性。

Observe / Feedback 使用流式记录分片与任务索引。交互 SFT / RL 使用独立 caption-image JSONL 与两组 rank-16 attention LoRA。SFT 同时训练 UND writer 和 GEN consumer；RL 冻结 writer，仅训练 consumer。prompt 编码、来源 GEN 和无条件 CFG 保持冻结。

原生 UND 全层注意力 Observe 已完成 20000 步；C2 Feedback 的文字反馈训练入口已准备。当前必要 smoke/日志检查共 8 项通过；结果与运行命令见实验文档。数据构建先读取带显式划分的来源以固定 family 划分，再生成记录与任务索引，保留重复记录；覆盖 21 个新标注文件与旧 CoRT 数据。这批新增标注的来源统计总量为 37,913,066 条。旧版 `annotation_index_v1` 索引需要重新构建。

## 开始使用

在与设备匹配的 PyTorch 环境中安装项目：

```bash
python -m pip install -e '.[dev]'
python -m qwen_latent_cot.diagnostics.feedback_cli --help
python -m qwen_latent_cot.diagnostics.visual_cli --help
python -m qwen_latent_cot.interaction.cli --help
```

- 运行文字反馈与 16 槽 memory 对照：[显式推理](docs/experiment.md#61-显式固定-memory-重生成)。
- 运行 Observe / Feedback：[文字训练入口](docs/experiment.md#62-observe--c2-feedback)，[W&B 与日志](docs/experiment.md#64-wb-与训练日志)。
- 运行交互 SFT / RL：[生成训练入口](docs/experiment.md#7-交互生成训练)。

## 基础模型与许可

基于 [BAGEL](https://github.com/ByteDance-Seed/Bagel) 的原生理解、生成与流匹配接口。代码采用 [Apache-2.0](LICENSE)，模型与数据使用各自许可证。
