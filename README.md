# BAGEL · 冻结 GEN + 空间 UND 速度纠错

在冻结的 BAGEL 生成骨干上训练一个空间速度纠错头：UND 副本逐图像 patch 读取当前采样状态与原生 GEN 同层特征，输出速度修正 `r`，采样按 `v = v_native + r` 推进原生 Euler。GEN 的全部参数、注意力、hidden state 与输出头保持冻结，可训练部分集中在纠错器内部（53.4M 参数）。

## 原理

![Method overview](assets/method_und.png)

*图：A 为原生 BAGEL MoT 结构，B 为训练目标，C 为推理循环。*

```text
完整 Prompt + noisy latent
        ↓
冻结 GEN + 原生 CFG/renorm → v_native、x̂、同层 GEN KV
        ↓
空间 UND 纠错器（每 patch 一个 query）→ 速度修正 r
        ↓
v = v_native + r → Euler
```

**纠错器**

- 主干为原生预训练 UND 的独立副本（28 层），Q/K/V/O 加 rank-64 LoRA。
- 输入为干净 proposal `x_t − t·v_native`，经原生 `vae2llm`、单位阵初始化的输入投影、原生时间嵌入与 2D 位置嵌入。
- 每层以原生 LayerNorm/RoPE 处理 patch query，并与完整 prompt KV、同层 GEN KV 拼接做注意力，随后走原生 MLP。
- 输出 head 从零初始化并在 FP32 下输出逐 patch 速度修正；零修正时采样轨迹与原生模型逐像素一致。

**训练目标**

```text
g      = v_guided − v_conditional
H_g(z) = z − min(⟨z, g⟩ / ‖g‖², 0)·g
target = H_g(ε − x0 − v_conditional)
loss   = MSE(r_raw, target)
推理   r_used = H_g(r_raw)
```

`H_g` 保留与 guidance 正交的修正分量，并投影掉沿 `−g` 的反向分量；`g = 0` 时为目标与部署的恒等映射。loss 直接作用于原始输出，反向分量在训练中获得梯度；部署时对实际施加的修正与全局基线应用同一投影。

**采样**

50 个时间节点、shift 3，前 25 次更新启用纠错；修正以 FP32 增量加入一次，原生 BF16 Euler 项保持原样。三条对照臂由同一采样器给出：`--native` 原生、`--global-only` 全局常量场、默认完整纠错。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `qwen_latent_cot/correction/` | 纠错器、目标函数、训练/采样循环、分布式累积、检查点 |
| `qwen_latent_cot/bagel/` | 冻结 BAGEL 的加载、CFG/renorm、上下文捕获、加速器 |
| `scripts/` | 64 卡持久启动、NPU pilot、照片语料构建 |
| `configs/` | 训练配置；`correction_guidance_safe.json` 为当前主线 |
| `tests/` | 21 项 CPU 契约测试（真实小型 MoT） |
| `assets/` | 方法图（PNG/PDF/SVG）与绘制脚本 |
| `docs/` | 方法原理文档 |

## 使用

安装本项目及设备对应的 PyTorch（Ascend 另需匹配的 `torch_npu`）：

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

构建照片语料并启动训练：

```bash
python scripts/prepare_photo_corpus.py \
  --source /absolute/path/observe_feedback_corpus \
  --output /absolute/path/correction_photos_v2_20261022

python -m qwen_latent_cot.correction.cli train \
  --model /absolute/path/Bagel-7B-MoT \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --config configs/correction_guidance_safe.json \
  --output /absolute/path/new-run --device npu:0
```

采样三条对照臂：

```bash
python -m qwen_latent_cot.correction.cli sample \
  --model /absolute/path/Bagel-7B-MoT \
  --checkpoint /absolute/path/new-run/step_005000.pt \
  --prompt 'a photo of three books' --seed 17 \
  --output /absolute/path/three-books.png --device npu:0
```

## 文档

[`docs/method.md`](docs/method.md)：生成通路、纠错器结构、目标函数推导与投影性质、采样集成、训练配置、指标定义。

## 许可与参考

代码采用 [Apache-2.0](LICENSE)。模型权重与外部数据使用各自许可证。

- BAGEL: <https://arxiv.org/abs/2505.14683>
- Correction Training: <https://arxiv.org/abs/2610.14136v2>
- CorrFlow: <https://arxiv.org/abs/2610.24750>
