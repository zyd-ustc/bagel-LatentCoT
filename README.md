# BAGEL · 冻结 GEN + 空间 UND 速度纠错

冻结原生 BAGEL 的生成专家（GEN），把同样冻结的理解专家（UND）改造成**逐图像 patch 的速度纠错头**：UND 读取当前采样状态和原生 GEN 同层特征，预测一个速度修正 `r`，`v = v_native + r` 再走原生 Euler。训练只用天然图文对和 flow-matching 残差 MSE，不修改 GEN 的注意力、hidden state 或输出头。

**当前状态（2026-10-22 14:18 CST）**：guidance-safe 条件残差实验在 4 节点 × 16 NPU 上运行中，已到 step 1411 / 5000，数值全部有限，checkpoint 10/100/500/1000 与 step 100/1000 的固定 32 条三路图已生成。上一轮 5000 步 post-CFG 实验已完成：残差 MSE 明显下降，但 32 条三路生图未通过画风与语义验收，作为负面结果保留。两轮实验都还没有产出"可用的生成改进"。

## 方法

```text
完整 Prompt + noisy latent
        ↓
冻结 GEN + 原生 CFG/renorm → v_native、x̂、同层 GEN KV
        ↓
空间 UND 纠错器（每 patch 一个 query）→ 速度修正 r
        ↓
v = v_native + r → Euler（修正不再经过 CFG）
```

- **纠错器**：原生预训练 UND 的独立副本（28 层），Q/K/V/O 加 rank-64 LoRA。输入投影从恒等矩阵初始化，FP32 输出 head 从零初始化，因此零修正时与原生采样逐像素等价。可训练参数 53.4M。
- **上下文**：完整 prompt KV 与同层 GEN KV 只作为冻结上下文；没有额外的 Memory bank，也不回灌或修改 GEN。
- **采样**：50 个节点、shift 3，前 25 次更新启用纠错；修正以 FP32 增量加一次，原生 BF16 Euler 乘积不变。
- **训练目标（当前主线）**：令 `g = v_guided − v_conditional`，`H_g(z) = z − min(⟨z,g⟩/‖g‖², 0)·g`，回归 `H_g(ε − x0 − v_conditional)`；推理时用同一个 `H_g` 投影实际施加的修正（含全局基线）。早期版本直接回归 `ε − x0 − v_guided`，诊断表明该目标包含系统性抵消 guidance 的分量。

方法细节、两种目标的推导与度量定义见 [docs/method.md](docs/method.md)。

## 关键证据

5000 步 post-CFG 实验，固定 8 条验证样本、相同输入与噪声下的速度 MSE：

| t | 原生 guided | 全局常量场 | 完整 UND |
|---:|---:|---:|---:|
| 1.000 | 1.665219 | 1.419663 | 0.524145 |
| 0.939 | 0.652931 | 0.626488 | 0.391629 |
| 0.861 | 0.450791 | 0.441983 | 0.343504 |
| 0.758 | 0.401827 | 0.401421 | 0.359151 |

完整 UND 明显优于时间相关的全局常量场，说明它学到了该基线之外的真实信号。但同一个 checkpoint 的 32 条三路生图显示：对象消失、重复背景纹理、彩色光照漂移，计数、空间关系和文字仍不可靠。**残差误差下降没有转化为生成质量提升**，这是本轮最重要的结论。

三条可复用的发现：

1. **训练指标不等于图像质量。** 最后 100 步平均 loss 0.400167，相对同批原生 guided MSE 0.700796 降低 42.9%，但固定 prompt 的画质和语义没有改善。论文/汇报中不能把这类数字当作成功率。
2. **纠错在系统性抵消 CFG。** 冻结 step5000 直接测量：200 个状态 100% 满足 `a < 0`（合并 `a ≈ −1.08`，闭环 `≈ −1.14`），删除反向分量能恢复吹风机、键盘、微波炉和人物脸部，但不能产生可靠语义纠错。
3. **因此先改目标，而不是堆步数或换数据。** 当前 run 只改训练目标与部署投影，架构、数据、预算和采样调度保持不变。

实验时间线、完整表格与逐条观察见 [docs/experiments.md](docs/experiments.md)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `qwen_latent_cot/correction/` | 纠错器、目标函数、训练/采样循环、分布式累积、检查点 |
| `qwen_latent_cot/bagel/` | 冻结 BAGEL 的加载、CFG/renorm、上下文捕获、加速器 |
| `scripts/` | 64 卡持久启动、NPU pilot、照片语料构建 |
| `configs/` | 训练配置；`correction_guidance_safe.json` 为当前主线 |
| `tests/` | 21 个 CPU 契约测试（真实小型 MoT） |
| `docs/` | 方法、实验与复现文档；`docs/archive/` 为工作记录 |

## 快速开始

安装本项目及设备对应的 PyTorch（Ascend 另需匹配的 `torch_npu`）：

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v        # 21 tests
```

训练与三路采样（完整命令、语料构建、64 卡启动、续训约束见 [docs/run.md](docs/run.md)）：

```bash
python -m qwen_latent_cot.correction.cli train \
  --model /absolute/path/Bagel-7B-MoT \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --config configs/correction_guidance_safe.json \
  --output /absolute/path/new-run --device npu:0
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/method.md](docs/method.md) | 架构、两种训练目标、度量定义、设计取舍 |
| [docs/experiments.md](docs/experiments.md) | 四阶段实验记录、数值表、负面结果与下一步 |
| [docs/run.md](docs/run.md) | 环境、语料、训练/采样/续训、评测流程、排障 |
| [docs/archive/README.md](docs/archive/README.md) | 按日期归档的计划、运行日志与旧版素材 |

## 限制

- 尚未产出通过验收的生成模型；两轮实验的语义与画风检查均未通过。
- 没有自动 benchmark 成功率。固定 32 条 prompt 的三路图只用于定性检查，不构成统计结论。
- 仓库不含模型权重、训练语料和生成图片，路径与检查点位置记录在实验文档中。
- 只在 Ascend NPU 上跑过全模型；CPU 测试用小型 MoT 验证训练契约，不代表全模型数值行为。
- 指标不可混用：`loss` / `zero_correction_mse` 是同一目标的拟合误差，`base_mse` 是物理 guided 速度误差，二者不能算出"改善百分比"。

## 许可与参考

代码采用 [Apache-2.0](LICENSE)。模型权重与外部数据使用各自许可证。

- BAGEL: <https://arxiv.org/abs/2505.14683>
- Correction Training: <https://arxiv.org/abs/2610.14136v2>
- CorrFlow: <https://arxiv.org/abs/2610.24750>（本实现只取 MSE，不含其 cosine 项）
