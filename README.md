# Bagel Latent CoT：目标驱动纠偏

在 Ascend NPU 上，用冻结的 SenseNova-U1.5 实现文生图轨迹内纠偏。

模型从纯文本开始生成。冻结的理解模块检查中途前瞻草图；通过则继续原轨迹，发现缺陷则回到确定性的历史态。纠偏器在短窗口内修正速度场，之后交由原生模型完成尾段。

训练分两阶段：先用目标图监督速度残差，再从真实历史态展开，通过冻结尾段监督最终图像。第二阶段使用部署细网格做前向，粗网格提供代理梯度。

## 使用

在已有 PyTorch / torch_npu 环境中安装：

```bash
pip install -e .
```

入口为 `u15-target-correction`，包含 `trace`、`train`、`sample` 和 `judge`。外部裁判另需安装 `openai`。

- [数据格式、两阶段训练与采样命令](docs/target_correction.md)
- [训练概览](docs/train.md)
- [推理概览](docs/inference.md)
