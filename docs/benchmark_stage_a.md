# Stage A 的现成 benchmark

> 历史 Stage A v4 记录：以下配置、检查点、运行状态和测试结果属于重构前版本。当前实现请使用 [Stage A v2](stage_a_v2_canvas_reentry.md)，旧检查点不能加载。


首选 **GenEval**：原始 553 条标注、六个类别，配套 Mask2Former + OpenCLIP 评分器。
可直接检查数量、颜色、位置与颜色绑定。官方仓库：<https://github.com/djghosh13/geneval>。

## 本机已有的评测资源

| Benchmark | modelarts-job 上已有 | 当前 Stage A 状态 |
|---|---|---|
| GenEval | 原始 553 条、检测器与 CLIP 权重、CPU 适配及独立环境 | 本次接入生成、官方目录导出、评分与汇总 |
| DPG / DSG | 原始题目与依赖图、mPLUG-large、既有评分环境 | 可复用；本次未接入新 Loop，也未重新跑评分 |
| GenEval2 | 800 条数据、Qwen3-VL-8B 权重、CPU / NPU 环境 | 尚未在本次接口上验证 NPU 单图评分 |
| GenEval++ / UniGenBench++ | 数据与部分脚本 | 原部署记录存在 API / 专用裁判依赖，本次未确认可直接运行 |

“资源在盘上”不等于当前链路 ready。GenEval 以本次真实生图和评分为验收，其他项明确保留未验证状态。

## 先检查整条链路

在 `/data/zyd_workspace/bagel-LatentCoT/code/bagel-stage-a-benchmark-20261021` 运行：

```bash
PYTHON=/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python
RUN=/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_geneval_step4000_pilot_20261021
"$PYTHON" scripts/benchmark_stage_a.py prepare --run "$RUN" \
  --checkpoint /data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_64npu_20261020_20000/step_004000.pt \
  --pilot
SAMPLE_WORKERS=6 bash scripts/benchmark_stage_a.sh "$RUN"
```

必须用新输出目录。pilot 使用六类各第一条原始 Prompt、seed 0，两组共 12 张图片。
生成沿用 Stage A 的窗口、28 层、feedback 次数、CFG 和分辨率，不恢复旧的 early20 / R2 实现。

## 全量入口

正式评测保留原始 prompt、标签和顺序；默认 553 条 × 4 seeds × 2 组，共 4,424 张图。
原生 BAGEL 与 Loop 使用相同 Prompt、seed、采样参数；不切割或改写正式评测 Prompt。

```bash
RUN=/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_geneval_step4000_full_20261021
"$PYTHON" scripts/benchmark_stage_a.py prepare --run "$RUN" \
  --checkpoint /data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_64npu_20261020_20000/step_004000.pt
SAMPLE_WORKERS=16 bash scripts/benchmark_stage_a.sh "$RUN"
```

仅在测试卡空闲时运行。全量入口已准备；pilot 不会自动启动全量。
评分环境为 `/root/venvs/geneval`，不修改共享环境，不调用付费 API。

## 输出和指标

- `plan.json`：固定 checkpoint、原始数据路径、类别数量、seed 与预期覆盖率。
- `samples/seed_0000/`：成图、实际参数、逐步记录、两组 HTML 对比。
- `official/{native_full,loop_full}/{原始索引}/`：原样 `metadata.jsonl` 及 `samples/{seed}.png`。
- `scores/{native_full,loop_full}.jsonl`：官方 `correct`、`reason`、检测结果及来源。
- `benchmark_summary.json`：每类准确率、覆盖率和类别宏平均。

只有全 553 条覆盖时才写 `full_553_prompt_macro_accuracy`；
只有默认四 seed 且全覆盖时才写 `geneval_macro_accuracy`。
pilot 仅写 `scored_subset_macro_accuracy`，不能当作正式 GenEval 分数。

现有 CPU 适配允许无 CUDA，CLIP 使用 FP32、关闭 CPU BF16 autocast；
沿用已部署评分器及官方阈值。它不是用 VLM 临时问答替代官方 GenEval。

## 本次验收结果

2026-10-21 在 modelarts-job 上，step 4000、seed 0 的 12 张 pilot 图片已全部生成并完成评分。
两组各六条，六类均返回官方 `correct` 与 `reason`。原始检测结果、图片和计划一起保存。
验证的是 Stage A → 原始 GenEval 标签 → 官方 CPU 评分的整条链路；全量评测尚未启动。

验收目录：`/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_geneval_step4000_pilot_20261021`。
若评分中断，可单独重跑 `benchmark_stage_a.py score --run ...`，会跳过已评分的图片；
重新汇总用 `benchmark_stage_a.py summary --run ...`。这两个命令使用 GenEval 独立 Python 环境。
