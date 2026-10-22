# BAGEL · 原生 T2I SFT 与 UND Spatial Correction

## 原生文生图后训练

纯T2I入口为 `python -m qwen_latent_cot.t2i.cli`。直接训练原生GEN专家，使用Table3的SFT优化配方：15K steps、lr2.5e-5、warmup500、EMA0.995、shift4。冻结文本UND和视觉编码器，不接纠错器。数据准备、FSDP训练、精确续训、EMA导出及原生生图见[使用指南](docs/t2i_sft.md)。当前通过小型原生模型与两进程CPU分片验证；真实7B NPU训练与全量数据准备尚未执行。

## UND 空间纠错实验

冻结原生 BAGEL GEN，让理解专家 UND 直接纠正当前图像预测。训练只使用天然图文对和速度修正 MSE。

新训练方案与运行记录见[guidance-safe重启](docs/guidance_safe_20261022/PLAN.md)，GEN输出速度方案暂不实现。

**历史实验状态：** [5000步照片实验](docs/overnight_20261022/RESULTS.md)已完成，残差MSE下降，但生图仍有画风和语义崩坏，尚未通过生成质量验收。

```text
完整 Prompt + noisy latent → 冻结 GEN + 原生 CFG/renorm → v_native、xhat、GEN 特征
                                         ↓
                         空间 UND corrector → 速度修正 r
                                         ↓
                         v = v_native + r → Euler（纠错不再经过 CFG）
```

- UND 使用原生预训练权重的独立副本，28 层，Q/K/V/O rank-64 LoRA。
- 一个图像 patch 对应一个纠错 query，保留原生 2D 位置嵌入和当前时间。
- Prompt 和同层 GEN KV 是冻结上下文；没有额外的 16-slot Memory，也不修改 GEN 注意力或 hidden states。
- 输入投影从恒等矩阵初始化，FP32 空间输出 head 从零初始化。零纠错时保留原生 CFG 与 Euler 运算结果。
- 新实验的目标为 `H_g(epsilon - x0 - v_conditional)`，损失为原始 UND 输出与该目标的 MSE。`g=v_guided-v_conditional`，`H_g(z)=z-min(<z,g>/||g||²,0)g`；g为0时不修改。推理也用H_g删除反向分量，不修改GEN架构。
- 训练不投影原始预测再算loss，避免负向预测落入零梯度区。该目标有意排除了反向guidance残差，不再等价于完整guided-flow回归。不使用 semantic CE、Teacher、partial Prompt、cosine、排序或 keep loss。
- `loss`与`zero_correction_mse`使用同一新目标；`base_mse`与`applied_base_mse`分别记录原生/实际纠错速度对原始flow目标的误差。不能将loss/base_mse解释成匹配的误差改善比例。
- 每个选中采样节点执行一次 UND correction，之后推进外层 Euler；不执行第二遍 GEN correction。
- 沿用 50 个采样节点、shift 3；在前 25 次更新启用纠错。新配置中 `v_native` 指原生 CFG/renorm 后的速度，训练和推理使用同一个 proposal。纠错通过 FP32 Euler 增量加一次；无条件分支和原生 CFG 不变。
- 同批训练目标同时估计每个时间节点的全局常量场，作为独立基线。它不参与 UND 的损失或优化器；新模式的global输出也经过相同H_g，所以实际施加的基线还依赖当前guidance方向。验证并列输出 native/global/corrected。

## 训练

安装本项目及设备对应 PyTorch。Ascend 还需匹配的 `torch_npu`。在仓库根目录运行：

```bash
python -m qwen_latent_cot.correction.cli train \
  --model /absolute/path/Bagel-7B-MoT \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --config configs/correction_guidance_safe.json \
  --output /absolute/path/new-correction-run --device npu:0
```

先构建独立照片子集（不修改原始记录）：

```bash
python scripts/prepare_photo_corpus.py \
  --source /absolute/path/observe_feedback_corpus \
  --output /absolute/path/correction_photos_v2_20261022
```

选取 COCO/Flickr30k 人工描述照片，检查图片解码、独立 caption、重复图文对与跨 split 冲突，保留原有 family 划分，按有效图文对数采样。此筛选不等于逐条验证所有对象和数量。VAE posterior 对同一样本使用固定种子；噪声和时间每次独立抽样。

64 卡（4 节点 × 16 NPU）的持久启动入口是 `scripts/launch_correction_64.py`。在 ModelArts 节点零执行：

```bash
/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python \
  scripts/launch_correction_64.py \
  --code /absolute/path/deployed-code \
  --output /absolute/path/new-correction-run \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --config /absolute/path/deployed-code/configs/correction_guidance_safe.json
```

新配置为 5,000 步、全局 batch 256、AdamW lr 1e-4、warmup 100、梯度裁剪 1。新实验使用 `configs/correction_guidance_safe.json`；旧实验配置为 `configs/correction_photo.json`。默认关闭 activation checkpointing（单卡实测峰值约 47.9 GiB）；显存受限时可在配置中启用。每个节点有独立日志和 PID，主进程写入 `train.jsonl`。启动器审查旧项目训练并停止，不终止无关作业。

## 采样和续训

```bash
python -m qwen_latent_cot.correction.cli sample \
  --model /absolute/path/Bagel-7B-MoT \
  --checkpoint /absolute/path/new-correction-run/step_001000.pt \
  --prompt 'a photo of three books' --seed 17 \
  --output /absolute/path/three-books.png --device npu:0
```

加 `--native` 获得相同种子的原生对照；新配置检查点加 `--global-only` 获得训练估计的全局校准对照。采样轨迹保存为图片路径加 `.json`。

训练时加 `--resume CHECKPOINT` 续训到 `--steps` 指定的总步数。新检查点保存可训练参数、全局基线统计量、优化器、数据采样器和各 rank RNG；冻结权重从原生 BAGEL 重建。精确续训要求相同原生模型、corpus、优化配置及 world size。不要替换原生模型目录的内容。

`configs/correction.json` 保留旧条件残差配置；已有 1000-step spatial 检查点仍按旧的 CFG 前纠错方式采样，不能直接续训为新配置。旧 C2、SFT、Stage A 检查点不能续训此架构。历史检查点/图片保留作对照；本分支不保留旧执行入口。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用真实小型 BAGEL MoT，检查零输出初始化、空间布局、梯度进入 UND、冻结隔离、checkpointing、原生 CFG、采样及检查点恢复。完整 NPU pilot 使用 `scripts/pilot_correction.py`，包含多次更新、保存/恢复及原生/纠错生图。

Step100/1000/3000/5000 自动执行固定32条语义 prompt 的三路生图，保留 benchmark annotations 和问题。它们不是自动生成的成功率。

原始依据：[Correction Training](https://arxiv.org/abs/2610.14136v2)、[CorrFlow](https://arxiv.org/abs/2610.24750)、[BAGEL](https://arxiv.org/abs/2505.14683)。本实现选择纯 MSE，不采用 CorrFlow 的 cosine 项。
