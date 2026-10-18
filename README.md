# BAGEL-LatentCoT

当前实验线是 **Anchored Dual-State Loop**。冻结原生 BAGEL。在同一个 flow 时间步里，GEN body 以原生入口加一个有界残差重入。旁路读者是独立冻结的 Observation 读出：中性种子、全部层、full-query GEN KV，权重来自 Observation checkpoint 的拷贝，不写回生成模型，也不把 memory token 写进 GEN attention。输入模式由 checkpoint 的训练任务解析为 `gen_full_query`。`prompt_and_gen` 保留为名字，当前前向不会拼接 Prompt KV。第二轮读出是 Anchored Observation：复用原生 Pre KV，Body 与 Post 用本轮重算结果。图像 token 用 Cross-Attention 按内容读取各个 Memory slot，再经残差门控写回 Body 入口。写入量是 `Bound(gamma * delta)`，上界作用在 Gate 之后的残差上。训练和推理都读取可学习 gate。gate 精确为 0 时速度与原生前向一致。`force_native_bypass` 与 `force_full_zero_path` 是独立的实验开关。Observer 的 query 是中性种子嵌入，并经过全部 UND 层；Body 区间只决定 GEN 重入位置。`gen_only` 对照重复 GEN body，残差只依赖当前 hidden 和原生入口，不携带 Memory，也不读取 UND。两轮修正返回完整的 \(v^{(0)}, v^{(1)}, v^{(2)}\)，以及每一轮的写入残差和观测。默认损失只监督最后一轮速度。`round_loss_weights` 可以给每一轮单独的非负权重。`round_velocity_report` 记录每一轮相对目标的误差是下降还是上升，以及相对原生速度的漂移。

```bash
python -m qwen_latent_cot.anchored.cli self-check
```

配置见 `configs/anchored_dual_state_loop.json`，合同见 [Anchored Dual-State Loop](docs/anchored_dual_state_loop.md)。

旧的 Observation → Memory writer → 生成 SFT 路线保留为诊断记录，训练入口默认锁住。复现历史训练时才设置 `BAGEL_ALLOW_LEGACY_MEMORY_SFT=1`。

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
