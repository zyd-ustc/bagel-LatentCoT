# BAGEL · UND Spatial Correction

冻结原生 BAGEL GEN，让理解专家 UND 直接纠正当前图像预测。训练只使用天然图文对和速度修正 MSE。

```text
完整 Prompt + noisy latent → 冻结 GEN → 干净预测 xhat、同次 GEN 特征
                                         ↓
                         空间 UND corrector → 速度修正 r
                                         ↓
                         v = v_native + r → CFG → Euler
```

- UND 使用原生预训练权重的独立副本，28 层，Q/K/V/O rank-64 LoRA。
- 一个图像 patch 对应一个纠错 query，保留原生 2D 位置嵌入和当前时间。
- Prompt 和同层 GEN KV 是冻结上下文；没有额外的 16-slot Memory，也不修改 GEN 注意力或 hidden states。
- 输入投影从恒等矩阵初始化，FP32 空间输出 head 从零初始化。初始条件速度等于原生预测。
- 唯一目标：`MSE(r, epsilon - x0 - v_native)`。不使用 semantic CE、Teacher、partial Prompt、cosine、排序或 keep loss。
- 每个选中采样节点执行一次 UND correction，之后推进外层 Euler；不执行第二遍 GEN correction。
- 沿用 50 个采样节点、shift 3；在前 25 次更新启用纠错。CFG 的参考分支保持原生。训练损失和输出 head 使用 FP32；采样时在 CFG 入口转回原生速度 dtype，保持原生 CFG 运算顺序。

## 训练

安装本项目及设备对应 PyTorch。Ascend 还需匹配的 `torch_npu`。在仓库根目录运行：

```bash
python -m qwen_latent_cot.correction.cli train \
  --model /absolute/path/Bagel-7B-MoT \
  --corpus /absolute/path/observe_feedback_corpus \
  --config configs/correction.json \
  --output /absolute/path/new-correction-run --device npu:0
```

已有 corpus 的 generation 二进制索引、原始完整 caption、目标图像及 train/val family 划分直接复用。采用源大小平方根加权采样，逐图计算并累计到全局 batch。VAE posterior 对同一样本使用固定种子；噪声和时间每次独立抽样。

64 卡（4 节点 × 16 NPU）的持久启动入口是 `scripts/launch_correction_64.py`。在 ModelArts 节点零执行：

```bash
/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python \
  scripts/launch_correction_64.py \
  --code /absolute/path/deployed-code \
  --output /absolute/path/new-correction-run \
  --corpus /absolute/path/observe_feedback_corpus
```

默认 1,000 步、全局 batch 256、AdamW lr 1e-4、warmup 100、梯度裁剪 1。配置见 `configs/correction.json`。默认关闭 activation checkpointing（单卡实测峰值约 47.9 GiB）；显存受限时可在配置中启用。每个节点有独立日志和 PID，主进程写入 `train.jsonl`。启动器审查旧项目训练并停止，不终止无关作业。

## 采样和续训

```bash
python -m qwen_latent_cot.correction.cli sample \
  --model /absolute/path/Bagel-7B-MoT \
  --checkpoint /absolute/path/new-correction-run/step_001000.pt \
  --prompt 'a photo of three books' --seed 17 \
  --output /absolute/path/three-books.png --device npu:0
```

加 `--native` 获得相同种子的原生对照。采样轨迹保存为图片路径加 `.json`。

训练时加 `--resume CHECKPOINT` 续训到 `--steps` 指定的总步数。新检查点仅保存可训练参数、优化器、数据采样器和各 rank RNG；冻结权重从原生 BAGEL 重建。精确续训要求相同原生模型、corpus、优化配置及 world size。不要替换原生模型目录的内容。

旧 C2、SFT、Stage A 检查点不能续训此架构。历史检查点/图片保留作对照；本分支不保留旧执行入口。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用真实小型 BAGEL MoT，检查零输出初始化、空间布局、梯度进入 UND、冻结隔离、checkpointing、原生 CFG、采样及检查点恢复。完整 NPU pilot 使用 `scripts/pilot_correction.py`，包含多次更新、保存/恢复及原生/纠错生图。

原始依据：[Correction Training](https://arxiv.org/abs/2610.14136v2)、[CorrFlow](https://arxiv.org/abs/2610.24750)、[BAGEL](https://arxiv.org/abs/2505.14683)。本实现选择纯 MSE，不采用 CorrFlow 的 cosine 项。
