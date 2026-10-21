# BAGEL Stage A v2 · Predictive Canvas Loop

Stage A 使用预测画布 `x_t - t*v` 重入，以及独立 Softmax 的 Memory 注意力。
原生 GEN 的 Prompt / Self Image 注意力保持冻结，不再增加 16 个 RoPE 位置。

- 训练一次纠错；推理可执行 1–3 轮，每轮更新画布和 UND Memory。
- consumer 只训练 Memory 分支的 Query LoRA，rank 32。
- 入口门、逐层 Memory 门和速度门全部零初始化。
- full Prompt 原生 Teacher 与 student 共享 x/t；采用残差 MSE、余弦方向损失和 Memory 文本监督。
- 未缩放速度的辅助损失负责启动零门训练，不改变初始采样结果。
- 新检查点格式为 `bagel_canvas_stage_a_v2`，不兼容旧 Stage A v4，需从 C2 parent 重新训练。

[当前架构、训练命令、迁移与验收](docs/stage_a_v2_canvas_reentry.md)。
CPU 小型 MoT 已验证零初始化逐像素一致；完整模型的 NPU 画风、语义和性能验收待执行。

## 原 SFT 路径

下方 SFT 命令保持原合同：末层 hidden 重入、两轮反馈、联合 Memory 注意力、+16 RoPE 和 Hard-Patch/Keep Loss。
SFT checkpoint 仍为 `bagel_full_loop_v3`，不使用 Stage A v2 结构。

## SFT 训练

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

## Stage A 训练

使用 `stage_a` 入口：GEN 读取删减 Prompt，UND 读取完整 Prompt，原生 full Prompt Teacher 提供残差目标。
[Stage A v2 训练和迁移](docs/stage_a_v2_canvas_reentry.md)。
[旧 Stage A 的 Prompt 统计及实验记录](docs/stage_a.md)。
[Stage A 生图测试与离线对比页](docs/stage_a_sampling.md)。
[GenEval2 高复杂度 Prompt：Base / 1 / 2 / 3 / 4 次反馈对比](docs/feedback_sweep.md)。
[GenEval benchmark 接入、现成资源与全量运行命令](docs/benchmark_stage_a.md)。
[四个 benchmark 的八组消融：运行与结果](docs/loop_ablation/README.md)。

## SFT 采样

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
默认使用 SFT 配置，因此会初始化旧 SFT 循环：Write/GEN LoRA 的增量为零，Memory 全强度注入，回灌强度为 0.15，使用全图 RMS 匹配（不做逐 token RMSNorm）。
