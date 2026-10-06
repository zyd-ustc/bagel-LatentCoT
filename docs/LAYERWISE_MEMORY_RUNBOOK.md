# 分层 Memory KV：用户运行命令

日期：2026-10-06。代码路径：`/private/yida_workspace/bagel-LatentCoT-main-layerwise-kv-20261006`。原生模型：`/private/yida_workspace/models/BAGEL-7B-MoT`。本版本不加载任何 loop 训练权重。

正式 GPU 生成、E0、评分、QA 和预算测量均由用户启动。CPU 检查只验证接口与数值关系。运行前确认所选卡归当前任务使用；`GPUS` 可以是八张卡，也可以是四张卡。

## 1. 原生权重 E0

在远端 H200 终端执行。只占所选的一张卡，不生成正式评测图片。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-layerwise-kv-20261006
export MODEL_PYTHON=/private/software/conda/envs/lcot/bin/python
export MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
export GPUS=0,1,2,3,4,5,6,7
export RUN=/private/yida_workspace/outputs/layerwise_kv_r1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$MODEL_PYTHON" scripts/evaluate/validate_layerwise.py \
  --model-path "$MODEL_PATH" --output "$RUN/e0.json" \
  --start-layer 0 --end-layer 8 --memory-slots 8 --loop-rounds 1 \
  2>&1 | tee "$RUN/e0.log"
```

预期：`passed=true`。R0/native 和 no-read/native 的 max_abs 为 0，prompt cache 不变，Memory/native 的 conditional velocity 有非零且有限的差异。CFG velocity 也单独报告。非零差异只说明反馈能被读取，不能说明语义编辑成功。

可单独运行 CUDA packing 接口检查，包含不同长度和一个没有 Memory slot 的样本：

```bash
CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$MODEL_PYTHON" -m pytest -q \
  tests/test_layerwise_memory.py -k cuda_native_packed_writer_contract
```

旧路径 parity 的入口仍为 `scripts/evaluate/validate_native.py`。此前旧路径已经通过真实权重一致性检查；不将其结果当作新分层路径的图像验证。

## 2. 八卡配对图像评测

E0 通过后执行。八卡是独立 prompt/seed worker，不是模型并行。默认 144 prompt × 2 seed × 3 arms = 864 张图像。输出目录应为新目录，不复用旧结果。默认 R1、窗口 `[0,8)`、K≤8、全部采样步，512px、50 个时间点（49 次 denoiser）、shift=3、text CFG=4。

```bash
bash scripts/evaluate/run_layerwise_memory_8gpu.sh "$RUN" \
  2>&1 | tee "$RUN/driver.log"
```

脚本按顺序生成 `BASE / MEMORY_LOOP / LAYERWISE_MEMORY_KV`，再执行最终图像语义/质量评分与合并。生成期间日志在 `generation/worker_*.log`；评分期间日志在 `quality/worker_*.log`。主日志在两个阶段可能暂时没有新输出；查看 worker 日志及进程判断进度。

```bash
tail -n 5 "$RUN"/generation/worker_*.log
tail -n 5 "$RUN"/quality/worker_*.log
cat "$RUN/quality_report/summary.md"
```

报告包含新路径 vs Base 和新路径 vs 旧 Memory loop，语义 GM、quality proxy、invalid、Repair/Damage 及 prompt 聚类区间。旧对照恢复原有初始化、readout 与 null CFG 行为；新旧存在多个结构差异，不归因于 KV 格式这一项。quality proxy 不能替代人工观察。此处 generation timing 是工程日志，不能当作正式预算结论。

选四卡时，只在启动前修改：

```bash
export GPUS=0,1,2,3
```

## 3. 可选：原生 UND Memory QA

这是离线信息诊断，不进入生图流程，不是质量成功条件。先用独立目录导出 GEN 实际读到的 Memory KV 和 guided one-step x0 图像，再读取 Memory 答题。只使用窗口实际读取层 `[1,8)`。SEED 明确读取所选 prompt 内容的同层 KV，用于识别文本回声；DYNAMIC 不读取完整 prompt cache。

```bash
export QA_RUN=/private/yida_workspace/outputs/layerwise_kv_qa_$(date +%Y%m%d_%H%M%S)
mkdir -p "$QA_RUN"
PROMPTS=/private/yida_workspace/umm-anchored-eval-tools-d126833/data/hard16.jsonl \
ARMS=BASE,LAYERWISE_MEMORY_KV SEEDS=0 PROBE_STEPS=8,24 PROBE_ARM=LAYERWISE_MEMORY_KV \
  bash scripts/evaluate/run_layerwise_memory_8gpu.sh "$QA_RUN" \
  2>&1 | tee "$QA_RUN/driver.log"

```

导出完成后执行 QA、观察标注与合并：

```bash
export PROMPTS=/private/yida_workspace/umm-anchored-eval-tools-d126833/data/hard16.jsonl
export OUTDIR="$QA_RUN/qa"
bash scripts/evaluate/launch_offline_8gpu.sh memory-qa \
  --manifests "$QA_RUN"/generation/worker_*/manifest.jsonl --benchmark "$PROMPTS" \
  --model-path "$MODEL_PATH" --memory-arm LAYERWISE_MEMORY_KV
export OUTDIR="$QA_RUN/labels"
bash scripts/evaluate/launch_offline_8gpu.sh memory-labels \
  --qa-dirs "$QA_RUN"/qa/worker_*/ \
  --judge-model /private/yida_workspace/models/Qwen3-VL-8B-Instruct \
  --geneval2-source /private/yida_workspace/umm-anchored-eval-tools-d126833/GenEval2/evaluation.py
/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python \
  scripts/evaluate/merge_memory_probe.py \
  --qa-dirs "$QA_RUN"/qa/worker_*/ --label-dirs "$QA_RUN"/labels/worker_*/ \
  --output-dir "$QA_RUN/probe_report"
```

DYNAMIC/SEED/EMPTY/VIT_IMAGE 使用同一问题和答案协议。重点查看 prompt 与观察图像不一致时的准确率、prompt echo 与 unknown 覆盖。x0 仅是当前噪声状态的一步预测代理，不是最终图像。不同 step 或同 prompt 的不同 seed 只用于检验信息随生成状态变化；不默认做 shuffle。

## 4. 可选：部署预算

只在开发集上校准。包含 prompt prefill、denoising、decode，至少 3 次 warmup 和 20 次测量，不含 probe/诊断。

```bash
CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$MODEL_PYTHON" scripts/evaluate/benchmark_budget.py \
  --model-path "$MODEL_PATH" \
  --prompts /private/yida_workspace/umm-anchored-eval-tools-d126833/data/easy16.jsonl \
  --arms BASE,MEMORY_LOOP,LAYERWISE_MEMORY_KV --warmups 3 --repeats 20 \
  --output "$RUN/budget.json"
```

`LAYERWISE_KV_NO_READ` 是可选实现诊断。它保留 writer 计算，但 GEN 不读 Memory；不作为默认图像评测 arm。R2 或其他窗口用独立输出目录并设置 `LOOP_ROUNDS`、`START_LAYER`、`END_LAYER`。本轮不训练。
