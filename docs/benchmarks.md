# early20_L28_R2 生图评测

## 五组统一推理接口（2026-10-16）

统一 Python 接口为 `qwen_latent_cot.diagnostics.inference.InferenceEngine`。Benchmark 入口为 `scripts/benchmark_memory_controls.py`，复用本页既有的官方 prompt、评分器和导出规则，不抽取独立固定 prompt 集。

| `arm` | 权重与推理行为 |
| --- | --- |
| `bagel` | 原版 BAGEL 权重与原始精度；不注入 memory |
| `c2_no_memory` | 加载 C2 UND 权重，使用该权重编码 prompt；不注入 memory |
| `c2_memory` | C2，early20/L28/R2；每节点根据自己的状态重新写 memory |
| `c2_memory_masked` | 保留同样的两遍 GEN、UND 写入、16 槽位置移动；全部 memory KV 从注意力中排除 |
| `c2_memory_swap` | C2，在相同 prompt/t 下消费另一 seed 的 memory；供体独立沿正常 `c2_memory` 轨迹更新 |

默认接收方 seed=0，交换供体 seed=1，可用 `--donor-seed-offset` 修改差值。供体不消费接收方反馈。每个启用节点记录接收方/供体状态 hash、供体 seed、prompt、t，以及供体额外计算量。它控制的是不同采样状态的反馈，不保证供体一定具有不同语义错误。

`bagel` 不加载 C2 参数。其结果中的 `checkpoint_sha256/checkpoint_step/trained_stage` 为空；`control_checkpoint_sha256` 标识本轮其他四组使用的 C2 文件。进程复用一个模型，切换时恢复原始 UND 权重和 dtype，并清空文字缓存；中性种子保持原生预填充，与原部署一致。

### 准备与运行

代码部署目录为 `modelarts-job:/root/bagel-LatentCoT-controls_20261016`。默认 `--checkpoint` 指向 `/root/outputs/feedback_nativeund_attn_full_20k`；准备时选取目录中步数最大的 `step_*.pt`，将具体路径与 SHA-256 固定到计划。后续生成读取计划，不随训练继续而换权重。`last.pt/best.pt` 可通过显式文件参数选择，文件改变后须重新准备新目录。

本次核对最新已保存编号为 `step_002000.pt`，SHA-256 为 `cb8479c5671c01ea07dc5b88a5a7c78577aacd3aa3141974b462eb832f73750f`。五组计划已准备在 `/root/outputs/c2_2000_memory_controls_bench`。每组包含全部榜单，默认执行前四项；单 seed 共 3478×5=17390 张被评测图，交换组另有供体轨迹计算。

以下命令均在 `modelarts-job` 执行。先把 `NPUS` 设为实际空闲物理设备 ID。

```bash
cd /root/bagel-LatentCoT-controls_20261016
export RUN=/root/outputs/c2_2000_memory_controls_bench
export NPUS=2,4,6,8,10,12,14

# 准备缺失的计划并生成五组；已完成图片可续跑。
bash scripts/benchmark_memory_controls_npu.sh "$RUN"

# 或一次完成前四项的生成、评分和汇总。
bash scripts/evaluate_memory_controls.sh "$RUN"
```

新一轮评测可显式指定 checkpoint 和 seed；改变采样参数时使用新输出目录：

```bash
/root/venvs/bagel-NPU/bin/python scripts/benchmark_memory_controls.py prepare \
  --run /root/outputs/c2_latest_controls_seed0123 \
  --checkpoint /root/outputs/feedback_nativeund_attn_full_20k \
  --seeds 0 1 2 3 --early 20 --loop-layers 28
```

按需生成部分榜单/对照，不改变完整计划。`--limit` 仅供路径检查，按每组、每 worker 分片后限制条数，正式评测不传该参数。

```bash
bash scripts/benchmark_memory_controls_npu.sh "$RUN" \
  --benchmarks geneval2 --arms c2_memory c2_memory_masked c2_memory_swap

ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
  /root/venvs/qwen3vl-npu/bin/python scripts/benchmark_memory_controls.py score \
  --run "$RUN" --scorer geneval2 --benchmarks geneval2 --device npu:0

/root/venvs/bagel-NPU/bin/python scripts/benchmark_memory_controls.py summary --run "$RUN"
```

只有部分图片完成时，评分增加 `--available-only`。汇总按组记录覆盖率，未覆盖完整榜单时保留 `scored_subset_*` 标记。GenEval++ / UniGenBench++ 的 API 和专用裁判要求沿用下文，调用统一入口 `export` 后分别在各组导出目录评分。

输出布局：

```text
RUN/
├── controls.json             # 固定五组、checkpoint、seed 和采样参数
├── summary.json              # arm → benchmark 的汇总
├── generation/worker_*.log
└── <arm>/
    ├── plan.json
    ├── images/<benchmark>/seed_0000/<item>.png
    ├── images/<benchmark>/seed_0000/<item>.json
    ├── generation/worker_*.jsonl
    ├── scores/*.jsonl
    └── exports/
```

### 其他代码调用同一接口

```python
from qwen_latent_cot.diagnostics.inference import InferenceEngine

engine = InferenceEngine.load(
    model="/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT",
    checkpoint="/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt",
    device="npu:0",
)
image, nodes = engine.generate(
    prompt=benchmark_row["prompt"], seed=0,
    arm="c2_memory_swap", donor_seed=1,
    early=20, loop_layers=28,
)
```

接口返回 PIL 图片及逐节点记录。五组固定为 R1（无 memory）或 R2（一次反馈），后续生成侧适配可在此接口扩展参数加载和策略。

### 验证记录

本地新增对照与既有浅层/原生 UND 检查：13 项通过。覆盖原版/C2 权重与 dtype 切换、无 memory、正确 memory、同布局屏蔽、同目标独立供体轨迹、最新编号 checkpoint 选择，以及五组计划和评分隔离。

远端完整 BAGEL 权重的五组 NPU 出图已通过，使用 GenEval2 原始第 0 题、seed=0 和完整 49 次 Euler 更新。交换组 donor seed=1。五组纯出图耗时分别约 13.0/13.1/17.7/18.0/27.6 秒，顺序对应上表；不含加载与 checkpoint 校验。路径检查图片保留在正式计划目录，后续生成可复用。五组图片均已通过 GenEval2 官方函数与 Qwen3-VL-8B 的实际 NPU 评分，并完成统一汇总；此前文档中的“NPU 裁判前向待验证”已由本次调用补齐。单题结果仅验证接口，不能用作 benchmark 改善结论。

扩大到 `test_memory_topology.py` 后为 41 项通过、3 项失败。三项分别涉及旧 C1 writer 无可训练参数、旧 L 拓扑入口、已移除的 `adapter_type` 配置；这些模块本轮未修改。此记录不将整套历史测试记为通过。

下文保留此前单组部署与评分细节。

部署主机：`modelarts-job`。记录日期：2026-10-16。

本轮使用 `/root/npu-eval-tools` 中已有的官方数据、评分代码和权重。生成代码位于 `/root/bagel-LatentCoT-gen_c2_2000_20261016`。评测对象是 C2-feedback 的 `step_002000.pt`，首轮每题生成一张图，seed 为 0。

## 评测范围

| Benchmark | 原始 prompt 数 | 评分器 | 当前部署状态 |
| --- | ---: | --- | --- |
| GenEval | 553 | Mask2Former + OpenCLIP ViT-L/14 | CPU 检测和评分已实际调用 |
| GenEval2 | 800 | Qwen3-VL-8B-Instruct，官方 Soft-TIFA | CPU 已完成匹配图片的实际评分；独立 NPU 环境已通过导入检查 |
| DPG-Bench | 1065 | mPLUG-large，依赖传播 | 权重全部匹配，实际问答与官方逐图评分均已通过 |
| DSG-1k | 1060 | mPLUG-large，官方依赖评分函数 | 官方评分函数已实际调用；1051 题依赖有效，另 9 题保留无依赖分 |
| GenEval++ | 280 | 官方脚本调用 `gpt-4.1` | 数据和脚本已部署；当前未设置 `OPENAI_API_KEY` |
| UniGenBench++ | 2400 | 官方专用裁判或 Gemini | 中英文长短 prompt 各 600；当前没有专用裁判权重或 Gemini 接口 |

前四项共 3478 张首轮计划。六项合计 6158 张。GenEval++ 与 UniGenBench++ 已接入生成计划，默认全流程只运行前四项。

这些指标主要检查文本遵循，包括物体、属性、数量、空间和事实依赖。图像清晰度、自然度和伪影仍需结合人工审阅判断。

## 固定的生成路径

生成入口直接调用现有 `shallow_sampling.sample`。参数固定为 `passes=2`、`early=20`、`loop_layers=28`。

- 前 20 次 Euler 更新执行 R2：第一次 GEN 计算后，UND 读取 prompt KV 与全部 28 层的 GEN 当前 query KV，写出 16 槽 memory；第二次 GEN 计算消费 memory。
- R2 表示总共两次 GEN 计算，即一次反馈更新。两次 GEN 计算使用同一个 `x_t` 和时间步。
- 后续 Euler 更新沿用原生生成路径。UND 每次反馈均参与全部 28 层计算。
- 分辨率 512，`sampler_steps=50`，实际是 50 个时间节点、49 次 Euler 更新。`timestep_shift=3`，CFG 4，`cfg_renorm_type=global`，`cfg_renorm_min=0`。
- checkpoint：`/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt`。
- checkpoint SHA-256：`cb8479c5671c01ea07dc5b88a5a7c78577aacd3aa3141974b462eb832f73750f`。

已有结果中有 16 张 GenEval2 图片与上述 checkpoint、prompt、seed、采样配置及完整解码条件一致，计划通过符号链接直接复用。

## 数据与评分约定

使用官方原始 prompt 和题目，不使用 `/root/unimoe-benchmark-supplement` 的改写任务。每个计划项保留 benchmark、原始 item ID、prompt、seed、评分元数据和图片路径。

首轮使用 seed 0，完整覆盖每个榜单的 prompt。后续通过 `prepare --seeds 0 1 2 3` 建立独立输出目录，可扩展为每题四张。单 seed 结果需与多 seed 报告分别记录。

### GenEval

复用 `/root/npu-eval-tools/geneval/src/geneval/evaluation/evaluate_images.py` 中的模型和逐图评分。阈值沿用官方默认值。当前使用 CPU，原仓库的 CUDA 断言已允许 CPU 执行。`clip_benchmark` 默认开启 CPU BF16 autocast，会触发当前 ARM Torch 2.1 的矩阵乘法错误；接入在 CLIP 文本与图像分类时关闭 autocast，使用已加载的 FP32 权重，评分规则与阈值保持原值。

汇总记录六类准确率，以及类别平均分。原始逐图结果保留检测框和失败原因。

### GenEval2

复用 `/root/npu-eval-tools/GenEval2/evaluation.py` 中的官方函数，使用原始 `vqa_list` 和 `skills`。主指标是每个 prompt 内 atom 概率的几何平均，再在 prompt 间取平均；同时记录算术平均分与逐 atom 概率。

本地裁判权重：`/data/zyd_workspace/models/Qwen3-VL-8B-Instruct`。

CPU 试评使用 `/root/venvs/qwen3vl`，已实际运行。正式 NPU 环境是 `/root/venvs/qwen3vl-npu`：系统 Torch 2.7.1 + torch_npu，独立 Transformers 4.57.6 与 Tokenizers 0.22.2。NPU 模型前向尚待空闲卡验证。

CPU 试评分写在 `pilot_scores/`；正式评分写在 `scores/`。

### DPG-Bench

保留全部 1065 个 item、14392 条原始题目。官方 `prepare_dpg_data` 在 pandas 已读取表头后仍跳过第一行，会丢掉第一条真实标注。本轮读取全部 CSV 数据行，逐图计算仍直接调用官方 `compute_dpg_one_sample`。

每张 512 图片作为 `pic_num=1` 的完整输入。答案严格按官方规则匹配 `yes`。依赖传播沿用 DPG 的原地更新顺序。

### DSG-1k

保留全部 1060 个 item、8182 条题目，不删重复 prompt，也不使用其他项目筛出的 1046 题子集。

原始标注有 9 个 item 的依赖无法直接执行：

- 非数字依赖：`posescript_69`、`posescript_41`、`posescript_55`、`stanford_paragraph_54`、`tifa160_67`、`stanford_paragraph_55`、`stanford_paragraph_72`、`posescript_19`。
- 父题缺失：`tifa160_134`。

全部题目仍执行问答。汇总分别记录 1060 题的无依赖分，以及 1051 题依赖有效子集的分数。后者标记为子集结果。依赖有效的题目直接调用官方 `calc_vqa_score`，按原始父题答案进行过滤。

DPG 与 DSG 共用 `/root/npu-eval-tools/dpg/models/mplug-large`。接入直接运行 ModelScope 的 `MPlugForAllTasks` 和 `MPlugPreprocessor`，保留权重、图像变换、问题分词和答案解码，绕过未使用的 OFA 导入。实际权重加载返回 `All keys matched successfully`。

### GenEval++ 与 UniGenBench++

GenEval++ 使用 [Echo-4o](https://github.com/yejy53/Echo-4o) 的原始 280 行顺序。导出后，官方脚本在当前工作目录读取 `Geneval++.jsonl` 与 `image/1.jpg` 至 `image/280.jpg`。图片链接保留原 PNG 像素，由官方脚本完成一次 JPEG 编码。

UniGenBench++ 使用 [UniGenBench](https://github.com/CodeGoat24/UniGenBench) 的四份 CSV。导出图片命名为 `{index}_{sample_index}.png`。`en_long` 的 index 509 在两个考点之间缺少逗号，导出 CSV 只补上这个逗号，原文件保留，修正记录写入 `exports/format_repairs.json`。

UniGen 的离线官方裁判是 `CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1` 等专用模型。普通 Qwen3-VL-8B 的判断不计作该榜单官方分。

## 执行

以下命令在 `modelarts-job` 上执行。2026-10-16 部署时，16 个 NPU 均由另一组生成任务占用，全量生图尚未启动。启动前将 `NPUS` 设为空闲设备的物理 ID。

### 前四项：生成、评分、汇总

```bash
cd /root/bagel-LatentCoT-gen_c2_2000_20261016
export RUN=/root/outputs/c2_2000_early20_L28_R2_bench
export NPUS=0,2,4,6,8,10,12,14
nohup bash scripts/evaluate_memory_benchmarks.sh "$RUN" > "$RUN/launcher.log" 2>&1 &
```

脚本先用一张复用图片验证 NPU 裁判前向，再生成缺失图片，随后并行启动 GenEval CPU、DPG/DSG CPU 和 GenEval2 NPU 评分，最后写入 `summary.json`。再次执行会复用已完成图片和评分。

生成日志位于 `generation/worker_*.log`，评分日志位于 `scores/`。运行中也可以汇总覆盖率：

```bash
/root/venvs/bagel-NPU/bin/python scripts/memory_benchmark.py summary --run "$RUN"
```

`coverage_complete` 表示所选榜单的计划图片是否全部完成评分。未完成时，分数使用 `scored_subset_*` 字段，表示已评分子集。

### 扩展为四个 seed

```bash
export RUN=/root/outputs/c2_2000_early20_L28_R2_bench_seed0123
/root/venvs/bagel-NPU/bin/python scripts/memory_benchmark.py prepare \
  --run "$RUN" --seeds 0 1 2 3 --reuse /root/outputs/c2_2000_generation
nohup bash scripts/evaluate_memory_benchmarks.sh "$RUN" > "$RUN/launcher.log" 2>&1 &
```

### CPU 试评已有的 16 张 GenEval2 图片

```bash
export RUN=/root/outputs/c2_2000_early20_L28_R2_bench
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  /root/venvs/qwen3vl/bin/python scripts/memory_benchmark.py score \
  --run "$RUN" --scores "$RUN/pilot_scores" --scorer geneval2 \
  --benchmarks geneval2 --device cpu --available-only
/root/venvs/qwen3vl/bin/python scripts/memory_benchmark.py summary \
  --run "$RUN" --scores "$RUN/pilot_scores" --benchmarks geneval2
```

CPU 首张实测评分耗时约 467 秒。16 张已有图片的试评已在后台启动，进程号为 `437939`，日志为 `pilot_scores/continuation.log`。该入口用于复用现有图片；正式全量评分使用 NPU 环境。

### GenEval++：接通 API 后执行

先在环境中设置 `OPENAI_API_KEY`。

```bash
bash scripts/benchmark_memory_npu.sh "$RUN" --benchmarks geneval-pp
/root/venvs/bagel-NPU/bin/python scripts/memory_benchmark.py export --run "$RUN" --benchmarks geneval-pp
cd "$RUN/exports/geneval-pp/seed_0000"
env -u PYTHONPATH PYTHONNOUSERSITE=1 /root/venvs/geneval-pp/bin/python \
  /root/npu-eval-tools/geneval-pp/src/Eval-gpt-4.1-geneval++.py
```

### UniGenBench++：接通专用裁判服务后执行

`UNIGEN_API_URL` 填写专用裁判的 chat completions 接口地址。以下例子评测英文短 prompt，其余三项更换 benchmark 名和对应的 en/zh 官方入口。

```bash
cd /root/bagel-LatentCoT-gen_c2_2000_20261016
bash scripts/benchmark_memory_npu.sh "$RUN" --benchmarks unigen-en
/root/venvs/bagel-NPU/bin/python scripts/memory_benchmark.py export --run "$RUN" --benchmarks unigen-en
env -u PYTHONPATH PYTHONNOUSERSITE=1 /root/venvs/unigen/bin/python \
  /root/npu-eval-tools/unigen/src/eval/src/offline_model_en_eval.py \
  --data_path "$RUN/exports/unigen-en" --csv_file "$RUN/exports/unigen-en/prompts.csv" \
  --api_url "$UNIGEN_API_URL" --images_per_prompt 1 --image_suffix .png --resume
```
