# BAGEL Full-Layer Dual-State Loop

在 `loop-layer-NPU` 上重构。每个启用循环的 flow 步执行一次原生 GEN，加两次全层反馈遍历。
上一轮 GEN 的末层图像 hidden 回灌到下一轮入口；UND Memory KV 在全部 block 联合参与 attention。

- 前 `ceil(实际 Euler 更新数 / 2)` 次更新启用循环，后半程原生采样。
- 全部 28 个 block 共享权重；16 个 Memory slot 全强度注入。
- 保留 Prompt KV、GEN 的 +16 RoPE 偏移及原生 CFG；偏移不逐轮累计。
- 训练和采样共用 `interaction.loop.full_loop`，中间反馈输出直接接受 flow 监督。
- 新 checkpoint 为 `bagel_full_loop_v3`。旧交互 LoRA/RL checkpoint 不兼容；format-4 C2 checkpoint 用于初始化 Reader。

## 回灌与训练目标

回灌不做逐 token RMSNorm；投影后用全图标量 RMS 匹配入口，初始强度为 0.15。
consumer 训练 Q/V/O LoRA。训练目标使用均值归一为 1 的 Hard-Patch 误差权重，
以及 `keep_weight=0.02` 的低误差区域速度保真项。默认 `hard_patch_alpha=1.0`。
原始 flow MSE、加权 flow MSE 和保真项分别记录，详见架构文档。

## 训练

安装项目依赖后，在仓库根目录执行：

```bash
python -m qwen_latent_cot.interaction.cli sft \
  --model /absolute/path/Bagel-7B-MoT \
  --parent /absolute/path/observe/best.pt \
  --train /absolute/path/train.jsonl --val /absolute/path/val.jsonl \
  --output /absolute/path/full-loop-run --device npu:0
```

每行输入为 `{"caption":"Three red apples on a plate.","image":"/absolute/path/image.jpg"}`。
默认 10,000 步、全局 batch 256；步数可通过 `--steps` 修改。续训使用 `--resume` 替代 `--parent`。

多卡入口为 `scripts/train_interaction_npu.sh OUTPUT --parent ... --train ... --val ...`。
多实例时，在每个实例设置相同 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`，并设置各自的 `NODE_RANK`；每机默认 16 卡。

## 采样

```bash
python -m qwen_latent_cot.interaction.cli sample \
  --model /absolute/path/Bagel-7B-MoT \
  --checkpoint /absolute/path/full-loop-run/step_001000.pt \
  --caption 'Three red apples on a plate.' --seed 17 \
  --output /absolute/path/sample.png --device npu:0
```

输出图片及同名 `.png.json` 更新记录。`--native` 禁用循环和适配器，并使用原始 BAGEL UND 权重生成基准。

[架构与验收](docs/design.md) · [全部改动文件 URL](docs/changed_files_urls.md)

不额外训练时，采样命令将 `--checkpoint ...` 替换为 `--parent /absolute/path/observe/best.pt`。
这会初始化同一个新循环：Write/GEN LoRA 的增量为零，Memory 全强度注入，回灌强度为 0.15，使用全图 RMS 匹配（不做逐 token RMSNorm）。
