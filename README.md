# BAGEL-LatentCoT

用连续 memory 连接 BAGEL 的理解与生成，在图像生成过程中读取状态、形成反馈并重新预测速度。

理解专家 UND 从目标文字和当前 GEN 表征写入 16 槽 memory，生成专家 GEN 消费反馈，外层采样器用最终速度推进图像。项目围绕两个任务展开：Observe 获取生成信息，Feedback 将目标偏差转化为可执行修正。

[Design · 原理](docs/design.md) · [Experiment · 实验与运行](docs/experiment.md)

## 当前进展

已完成显式文字 memory 推理、C1 文本压缩与响应训练、旧版视觉读取训练，以及统一的 Observe / Feedback / C3 训练框架。显式与浅层生成的人工审查已观察到语义编辑；下一轮实验聚焦修改正确性、正确内容保持和跨 seed 稳定性。

主训练使用流式记录分片与任务索引，按来源、图像角色和噪声范围选择监督。C1 权重作为可选初始化或独立文字教师，生成损失通过冻结 GEN 回传到 memory。

本地工程验证包含 118 项测试。下一步是整合 21 个新标注文件与旧 CoRT 数据，启动新版 NPU 训练。这批新增标注的来源统计总量为 37,913,066 条。

## 开始使用

在与设备匹配的 PyTorch 环境中安装项目：

```bash
python -m pip install -e '.[dev]'
python -m qwen_latent_cot.diagnostics.feedback_cli --help
python -m qwen_latent_cot.diagnostics.visual_cli --help
```

- 运行文字反馈与 16 槽 memory 对照：[显式推理](docs/experiment.md#61-显式固定-memory-重生成)。
- 运行 Observe、Feedback 与 C3：[训练入口](docs/experiment.md#62-observe--feedback--c3)。

## 基础模型与许可

基于 [BAGEL](https://github.com/ByteDance-Seed/Bagel) 的原生理解、生成与流匹配接口。代码采用 [Apache-2.0](LICENSE)，模型与数据使用各自许可证。
