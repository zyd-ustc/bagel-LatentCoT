# Stage A 生图测试

测试使用训练 checkpoint 的原始参数，不放大 gate，不改变 Loop 的时间窗口或全层注入方式。
采样和训练共用同一条生成路径。脚本固定 checkpoint、Prompt 和初始噪声种子，16 个独立 worker 各占一张卡。

## 测什么

`configs/stage_a_sample_prompts_32.jsonl` 包含 16 条人工编写、带明确检查项的长 Prompt，
8 条来自 v1 validation split 的可切分长 Prompt，以及 8 条既有短 Prompt 对照。
人工编写的长 Prompt 从 GEN 输入中删除中间完整句子；v1 文本使用 Stage A 的安全分句规则。

完整组：原生 BAGEL 完整 Prompt vs 训练后的 Loop 完整 Prompt。
信息差组：原生 BAGEL 删减 Prompt vs Loop（GEN 删减 / UND 完整）vs Loop（GEN 删减 / UND 错配）。
短 Prompt 仅执行完整组。合计 32 条、136 张图片，默认 seed 17。

错配组把 UND 的文本换成另一条完整 Prompt，GEN 保持当前删减文本；
这不是“用另一张图生成的 Memory 做交换”。观察生成图是否随 UND 内容变化，
并按每条检查项核验数量、属性绑定和空间关系，不把 RGB 差异当成语义改善。

## 在 modelarts-job 运行

在本次测试代码目录 `/data/zyd_workspace/bagel-LatentCoT/code/bagel-stage-a-sampling-20261021` 执行：

```bash
export SAMPLE_CHECKPOINT=/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_64npu_20261020_20000
export SAMPLE_PROMPTS=$PWD/configs/stage_a_sample_prompts_32.jsonl
export SAMPLE_OUTPUT=/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_sample_compare_20261021
bash scripts/stage_a_sample_compare_16.sh
```

必须使用新输出目录。若 checkpoint 参数为训练目录，启动器仅在开始时解析一次最新完整 checkpoint，
再把固定文件路径传给全部 worker。训练后续保存的新 checkpoint 不会混入这次比较。
运行前确认设备空闲；此脚本不会停止其他服务。

每张图旁保存实际使用的 GEN/UND 文本、种子、checkpoint、更新次数、时间和显存。
每个 worker 保存状态与独立日志。全部 worker 成功后，自动检查图片完整性并生成
`summary.json` 和离线 `index.html`，可切换完整组 / 信息差组、按样本来源筛选。

若只需重新生成页面（不重新采样）：

```bash
python scripts/render_stage_a_compare.py --output "$SAMPLE_OUTPUT"
```

Prompt 清单可通过 `scripts/prepare_stage_a_sample_prompts.py --corpus ... --output ...` 重新构建。
本次固定清单已保存到代码仓库，后续比较 checkpoint 时复用它，不重新抽取。

## 2026-10-21 首次运行

- modelarts-job 上的 16 个独立 worker 全部成功，固定使用 `step_003000.pt`、seed 17。
- 32 条 Prompt、136 张 512×512 图片；50 个采样节点、49 次 Euler 更新，前 25 次更新在全部 28 个 block 循环。
- 最大峰值分配显存 30.86 GiB；本次没有修改正在运行的 64 卡训练代码或参数。
- 结果目录：`/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_sample_compare_20261021`。
- 本地结果：`artifacts/stage_a_sample_compare_20261021/index.html`。

CPU 检查通过 9 项，覆盖分组的 GEN/UND 文本可见性、Prompt 清单、输出完整性和离线页面生成，
以及已有 Stage A 路径。页面提供完整组和信息差组切换；图片按每条预先写好的语义检查项人工判读。

首次人工抽查发现计数退化：四只鸟样例中，原生生成四只，Loop 生成七只；
三朵向日葵样例中，原生为三朵，Loop 为四朵。此结论仅针对这些样例和 seed 17，
完整记录见本地 `artifacts/stage_a_sample_compare_20261021/visual_inspection.md`。
