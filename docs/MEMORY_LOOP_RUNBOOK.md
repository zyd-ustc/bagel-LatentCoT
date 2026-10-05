# H200：恢复旧 Memory loop 的 training-free 执行

代码目录：`/private/yida_workspace/bagel-LatentCoT-main-memory-loop-20261005`。
模型：`/private/yida_workspace/models/BAGEL-7B-MoT`。
模型 Python：`/private/software/conda/envs/lcot/bin/python`。
评分 Python：`/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python`。

所有正式生成、评分、问答和预算检查由用户启动。代理只执行 CPU 小模型测试。不要复用逐层 `1/N` 实验的输出目录。`SYNC_COMMIT` 和 `SYNC_SOURCE_SHA256` 记录此快照版本。

## 1. 选择空卡，执行真实权重一致性检查

在 H200 终端运行。先查看 GPU 占用；仅使用分配给本实验的空卡。以下 `GPUS` 示例为八卡，四卡可改为四个卡号。实际 worker 数由列表长度决定。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-memory-loop-20261005
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export GPUS=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"
export RUN=/private/yida_workspace/outputs/memory_loop_r1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
/private/software/conda/envs/lcot/bin/python scripts/evaluate/validate_native.py \
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT \
  --loop-rounds 1 --start-layer 0 --end-layer 8 --memory-slots 8 \
  --output "$RUN/native_parity.json" 2>&1 | tee "$RUN/native_parity.log"
```

通过条件：`passed: true`。三个 velocity 对比均 `equal: true`，prompt KV 不变。E0 固定同权重/同噪声并包括两个不同长宽比的样本。未通过时不要将图像结果用于机制结论；提供该日志用于修复。

## 2. 多卡配对生成与质量评分

仍在上述终端运行。默认 Base / Memory loop 各 144 prompt×2 seed，共 576 张图。hard128 和 easy16 只用于沿用现有问题，不设置 384×3 门槛。所有 arm 使用相同初始噪声。评分进程在生成模型退出后启动。

```bash
unset CUDA_VISIBLE_DEVICES
export LOOP_ROUNDS=1
bash scripts/evaluate/run_memory_loop_8gpu.sh "$RUN" \
  2>&1 | tee "$RUN/pipeline.log"
```

读取 `quality_report/summary.md`，并检查配对图片。报告包含语义、质量 proxy、invalid、Repair/Damage 和聚类区间；代理不能把这些指标等同于人工质量偏好。

R=2 使用新目录运行：

```bash
export LOOP_ROUNDS=2
export RUN=/private/yida_workspace/outputs/memory_loop_r2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" /private/software/conda/envs/lcot/bin/python scripts/evaluate/validate_native.py \
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT --loop-rounds 2 \
  --output "$RUN/native_parity.json" 2>&1 | tee "$RUN/native_parity.log"
bash scripts/evaluate/run_memory_loop_8gpu.sh "$RUN" \
  2>&1 | tee "$RUN/pipeline.log"
```

可设置 `PROMPTS=/absolute/path/data.jsonl`、`SEEDS=0,1`、`START_LAYER=0`、`END_LAYER=8`、`MEMORY_SLOTS=8`、`PROGRESS_START=0`、`PROGRESS_END=1`。不同设置必须使用不同 RUN。Data JSONL 包含 `prompt_id`、`prompt` 和评分问题；height/width 可指定不同分辨率。

续跑同一 RUN 会校验代码、权重、数据、配置和图像 hash。不要修改代码后续跑同一目录。日志位于 `generation/worker_*.log`、`quality/worker_*.log`。目录 glob 必须使用 `worker_*/`，避免匹配日志文件。

## 可选诊断：不作为默认评测

`ARMS=BASE,MEMORY_LOOP,MEMORY_NO_READ` 增加 no-read 实现诊断。其输出应等于 Base；该 arm 额外执行 native query 计算，不能用于预算结论。`DIAGNOSTICS=1` 记录 slot rank、cosine、std、sigma1 和逐层更新比率。

`PROBE_STEPS=8,16,24` 捕获最后一轮 body 中 GEN 读取的各层 Memory KV，以及第一轮 Read 的层输入 KV。M 在本版本跨层、跨 body 轮传递；SEED 是第一轮 Read 输入参考，不是旧逐层 prompt seed。捕获还解码 guided x0 proxy，增加开销。完整 QA 仍调用现有 `launch_offline_8gpu.sh memory-qa`、`memory-labels` 和 `merge_memory_probe.py`，不会默认启动。QA 存在读出校准限制，不足以单独证明 Memory 没有语义信息。

独立预算测量使用 `scripts/evaluate/benchmark_budget.py --loop-rounds 1`，关闭上述诊断，至少三次 warmup 和二十次测量。普通 generation 日志属于工程计时。
