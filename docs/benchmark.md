# Benchmark

统一入口通过相同 prompt、seed、采样参数和官方评分器比较 BAGEL、parent 与训练后交互模型。生成接口为 `InferenceEngine`，调度入口为 `scripts/benchmark_memory_controls.py`。

## 评测集合

| Benchmark | Prompt 数 | 评分器 | 指标 |
|---|---:|---|---|
| GenEval | 553 | Mask2Former＋OpenCLIP ViT-L/14 | 六类准确率与类别平均 |
| GenEval2 | 800 | Qwen3-VL-8B-Instruct；官方 Soft-TIFA | 每个 prompt 的原子概率几何平均，再跨 prompt 平均 |
| DPG-Bench | 1,065 | mPLUG-large | 官方依赖传播分数 |
| DSG-1k | 1,060 | mPLUG-large | 无依赖分与有效依赖子集分 |
| GenEval++ | 280 | 官方 GPT-4.1 脚本 | 官方汇总 |
| UniGenBench++ | 2,400 | 官方专用裁判或 Gemini | 中英文长短 prompt 分项 |

默认流程运行前四项，共 3,478 个 prompt。六项合计 6,158 个 prompt。UniGenBench++ 包含中英文长短 prompt 各 600 条。

使用官方原始 prompt、题目与顺序。每个计划项保留 benchmark、原始 item ID、prompt、seed、评分元数据和图片路径。

## 六组交互对照

对 `interaction_lora_v1` checkpoint，准备入口自动选择六组：

| arm | 权重与行为 |
|---|---|
| `bagel` | 原版 BAGEL，无 memory |
| `parent_no_memory` | parent UND 编码 prompt，无 memory |
| `parent_memory` | parent 的 R2 反馈，两组 LoRA 关闭 |
| `interaction_memory` | writer 与 consumer LoRA 开启，逐节点形成反馈 |
| `interaction_memory_masked` | 保留两遍 GEN、writer、consumer 和位置布局，屏蔽 memory KV |
| `interaction_memory_swap` | 消费同一 prompt / t、不同 seed 的独立轨迹 memory |

默认接收方 seed=0，供体 seed=1，`--donor-seed-offset` 调整差值。交换组记录供体状态、seed、状态哈希与额外计算量。

Format-4 C2 checkpoint 使用五组：`bagel`、`c2_no_memory`、`c2_memory`、`c2_memory_masked`、`c2_memory_swap`。两类 checkpoint 共用生成、评分、导出与汇总入口。

## 采样协议

| 项目 | 设置 |
|---|---|
| 图像分辨率 | 512×512 |
| 时间网格 | 50 节点、49 次 Euler 更新，shift=3 |
| CFG | 4；global renorm；renorm min=0 |
| 交互窗口 | 前 20 次更新 |
| 交互深度 | GEN / UND 全部 28 层 |
| 单节点计算 | R2：两遍 GEN，一次 UND 反馈 |
| Memory | 16 槽，每次反馈从中性种子重写 |
| 首轮 seed | 0；扩展比较使用 0、1、2、3 |

两遍 GEN 使用相同 x_t 和 t；后续采样节点消费最终引导速度。窗口外沿原生 GEN 采样。

准备阶段将具体 checkpoint 路径、SHA-256、seed 和采样参数写入计划。生成与评分按该计划执行。

## 准备与执行

在配置了 `/root/npu-eval-tools` 的 NPU 环境中，于仓库根目录执行：

```bash
export MODEL_PYTHON=/root/venvs/bagel-NPU/bin/python
export MODEL_PATH=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export CHECKPOINT=/absolute/path/to/interaction_checkpoint.pt
export RUN=/root/outputs/interaction_bench
export NPUS=0,2,4,6,8,10,12,14
"$MODEL_PYTHON" scripts/benchmark_memory_controls.py prepare \
  --run "$RUN" --model "$MODEL_PATH" --checkpoint "$CHECKPOINT" \
  --seeds 0 --early 20 --loop-layers 28
bash scripts/evaluate_memory_controls.sh "$RUN" --model "$MODEL_PATH"
```

全流程生成缺失图片，再调用 GenEval CPU、DPG / DSG CPU 和 GenEval2 NPU 评分，最后写入汇总。重新执行可接续已有图片与评分。

### 分开生成、评分和汇总

```bash
bash scripts/benchmark_memory_controls_npu.sh "$RUN" \
  --model "$MODEL_PATH" --benchmarks geneval2 \
  --arms interaction_memory interaction_memory_masked interaction_memory_swap

ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  /root/venvs/qwen3vl-npu/bin/python scripts/benchmark_memory_controls.py score \
  --run "$RUN" --scorer geneval2 --benchmarks geneval2 --device npu:0

"$MODEL_PYTHON" scripts/benchmark_memory_controls.py summary --run "$RUN"
```

部分图片评分使用 `--available-only`。`summary.json` 按组保存覆盖率，完整覆盖使用正式汇总字段，已评分子集使用 `scored_subset_*`。

### 多 seed

```bash
"$MODEL_PYTHON" scripts/benchmark_memory_controls.py prepare \
  --run /root/outputs/interaction_bench_seed0123 \
  --model "$MODEL_PATH" --checkpoint "$CHECKPOINT" \
  --seeds 0 1 2 3 --early 20 --loop-layers 28
```

不同 seed 协议使用独立输出目录。各组共享相同 prompt 与 seed，汇总同时报告分数差、跨 seed 变化和实际耗时。反馈贡献通过正常、屏蔽和交换三组比较；计算效率比较计入 GEN、UND、供体轨迹及实际生成时间。

## 官方评分细节

### GenEval

使用 `/root/npu-eval-tools/geneval/src/geneval/evaluation/evaluate_images.py`，沿用官方阈值，记录六类分数、检测框和失败原因。CPU 接入对 CLIP 分类使用 FP32 前向。

### GenEval2

使用 `/root/npu-eval-tools/GenEval2/evaluation.py` 的原始 `vqa_list` 和 `skills`。主指标为原子概率几何平均，同时保存算术平均与逐 atom 概率。

裁判为 `/data/zyd_workspace/models/Qwen3-VL-8B-Instruct`，NPU 环境为 `/root/venvs/qwen3vl-npu`。交互 RL 的训练奖励采用[训练问题缓存的算术平均](data.md#rl-奖励问题)，两种用途分别输出。

### DPG-Bench

完整读取 1,065 个 item、14,392 条原始题目，以每张 512px 图片作为 `pic_num=1` 输入。逐图调用官方 `compute_dpg_one_sample`，答案按官方 `yes` 规则匹配，依赖传播沿官方更新顺序。

### DSG-1k

读取 1,060 个 item、8,182 条题目。全部题目执行问答，汇总包含 1,060 题无依赖分与 1,051 题有效依赖子集分。有效依赖调用官方 `calc_vqa_score`。

8 个 item 的依赖为非数字：`posescript_69`、`posescript_41`、`posescript_55`、`stanford_paragraph_54`、`tifa160_67`、`stanford_paragraph_55`、`stanford_paragraph_72`、`posescript_19`。`tifa160_134` 缺少父题。

DPG 与 DSG 共用 `/root/npu-eval-tools/dpg/models/mplug-large`，接入 ModelScope 的 `MPlugForAllTasks` 与 `MPlugPreprocessor`。

### GenEval++

使用 [Echo-4o](https://github.com/yejy53/Echo-4o) 原始 280 行顺序。官方脚本读取 `Geneval++.jsonl` 与 `image/1.jpg` 至 `image/280.jpg`。

设置 `OPENAI_API_KEY` 后，从仓库根目录生成、导出并评分：

```bash
bash scripts/benchmark_memory_controls_npu.sh "$RUN" \
  --model "$MODEL_PATH" --benchmarks geneval-pp
"$MODEL_PYTHON" scripts/benchmark_memory_controls.py export \
  --run "$RUN" --benchmarks geneval-pp
cd "$RUN/interaction_memory/exports/geneval-pp/seed_0000"
env -u PYTHONPATH PYTHONNOUSERSITE=1 /root/venvs/geneval-pp/bin/python \
  /root/npu-eval-tools/geneval-pp/src/Eval-gpt-4.1-geneval++.py
```

六组的导出目录分别执行官方评分。

### UniGenBench++

使用 [UniGenBench](https://github.com/CodeGoat24/UniGenBench) 的四份 CSV。图片命名为 `{index}_{sample_index}.png`，专用裁判可使用 `CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1`。`en_long` index 509 的考点分隔符修正写入 `exports/format_repairs.json`。

设置专用裁判服务 `UNIGEN_API_URL` 后，从仓库根目录生成英文短 prompt 图像、导出并评分：

```bash
bash scripts/benchmark_memory_controls_npu.sh "$RUN" \
  --model "$MODEL_PATH" --benchmarks unigen-en
"$MODEL_PYTHON" scripts/benchmark_memory_controls.py export \
  --run "$RUN" --benchmarks unigen-en
env -u PYTHONPATH PYTHONNOUSERSITE=1 /root/venvs/unigen/bin/python \
  /root/npu-eval-tools/unigen/src/eval/src/offline_model_en_eval.py \
  --data_path "$RUN/interaction_memory/exports/unigen-en" \
  --csv_file "$RUN/interaction_memory/exports/unigen-en/prompts.csv" \
  --api_url "$UNIGEN_API_URL" --images_per_prompt 1 --image_suffix .png --resume
```

其他组、语言与长短 prompt 使用对应导出目录和官方入口。

## 输出与运行记录

```text
RUN/
├── controls.json
├── summary.json
├── generation/worker_*.log
└── <arm>/
    ├── plan.json
    ├── images/<benchmark>/seed_0000/<item>.png
    ├── images/<benchmark>/seed_0000/<item>.json
    ├── generation/worker_*.jsonl
    ├── scores/*.jsonl
    └── exports/
```

2026-10-16，`modelarts-job` 使用 C2 `step_002000.pt` 完成五组真实 NPU 生成和 GenEval2 评分链路。输入为 GenEval2 第 0 题、seed=0；交换供体 seed=1。五组纯生成耗时约 13.0 / 13.1 / 17.7 / 18.0 / 27.6 秒，计时对应原版、C2 无反馈、正常、屏蔽、交换。

当日五组计划保存在 `/root/outputs/c2_2000_memory_controls_bench`。2026-10-17，交互 checkpoint 完成正常、屏蔽、交换及 parent 的 Euler 验证和 RL 保存重载，运行结果见[实验文档](experiments.md#阶段结果)。完整榜单结果按各运行的 `summary.json` 汇总。
