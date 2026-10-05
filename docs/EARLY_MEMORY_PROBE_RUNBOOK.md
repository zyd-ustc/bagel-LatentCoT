# 早期 loop 与原生 UND Memory：H200 八卡评测

代码位于 H200：`/private/yida_workspace/bagel-LatentCoT-main-early-memory-20261005`。正式评测由用户运行；代理没有执行下列流程。

## 运行完整流程

前提：GPU 0–7 已分配给本实验。下面在 H200 终端执行，不要求 tmux。使用既有 BAGEL 与 scorer 环境，无需安装包。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-early-memory-20261005
export GPUS=0,1,2,3,4,5,6,7
unset PROMPTS
export MODEL_PYTHON=/private/software/conda/envs/lcot/bin/python
export SCORER_PYTHON=/private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python
export MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
export START_LAYER=0 END_LAYER=8
export SEEDS=0,1 EVALUATIONS=2 PROBE_STEPS=8,16,24 MAX_QUESTIONS=4
export RUN=/private/yida_workspace/outputs/early_und_memory_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
bash scripts/evaluate/run_early_memory_8gpu.sh "$RUN" 2>&1 | tee "$RUN/pipeline.log"
```

默认去重合并 `hard128.jsonl` 和 `easy16.jsonl`，保存为 `$RUN/prompts.jsonl`。这些是历史开发题，不称独立确认集。脚本输出实际题数。没有固定 384×3 任务。若要使用自己的带 `vqa_list/skills` 或 TIIF 问题的清单，在启动前设置 `export PROMPTS=/absolute/path/prompts.jsonl`。如 shell 已保留旧的 PROMPTS，先 `unset PROMPTS`，再用默认题目。

每个阶段各用 8 卡，阶段之间顺序执行并释放模型：

1. 五组配对生成：BASE、GEN_LAYERWISE、MEMORY_DYNAMIC、MEMORY_STATIC、MEMORY_NO_READ。
2. 最终图像评分：原 GenEval2 soft-TIFA、Repair/Damage、quality proxy、invalid，并合并报告。
3. 原生 UND 问答：DYNAMIC、SEED、EMPTY，以及离线 VIT_IMAGE 校准。
4. 固定 Qwen3-VL 标注各时刻 x0 图像估计的实际内容。judge 不见完整 prompt 或要求答案。
5. 合并问答与标签，检查完整配对覆盖。

`N=2` 表示每个选中层共执行两次 GEN 运算。窗口 `[0,8)` 表示零起算第 0–7 层。只在采样前半程循环。50 个时间点对应 49 次 denoiser 调用；probe step 8、16、24 的进度分别约为 0.167、0.333、0.5。捕获的 Memory 是最后一次 GEN 读取的 KV，而不是未被消费的额外 writer 输出。

## 查看进度与续跑

```bash
# generation / quality / memory_qa / memory_labels 四个阶段各有 8 份 worker 日志。
tail -n 12 "$RUN"/generation/worker_*.log
tail -n 12 "$RUN"/quality/worker_*.log
tail -n 12 "$RUN"/memory_qa/worker_*.log
tail -n 12 "$RUN"/memory_labels/worker_*.log
nvidia-smi
```

只读取当前已开始阶段的日志。大模型加载及权重哈希阶段可能暂时没有逐图输出。若进程失败，先看该阶段 worker 日志；错误不会变成零分或被静默丢弃。修复原因后，用相同 `$RUN` 和相同参数重跑完整命令，各 worker 会跳过已完成记录。不要改变源码、权重、数据或评分协议后复用同一输出目录。

## 只重跑离线阶段

生成 manifest 已完整时可分别运行。以下沿用上面的 `$RUN`、环境变量和同一代码目录；题目按生成时绑定的清单选择。

```bash
export PROMPTS="$RUN/prompts.jsonl"
export OUTDIR="$RUN/memory_qa"
bash scripts/evaluate/launch_offline_8gpu.sh memory-qa \
  --manifests "$RUN"/generation/worker_*/manifest.jsonl \
  --benchmark "$PROMPTS" --model-path "$MODEL_PATH" --max-questions 4 --max-count 12

export OUTDIR="$RUN/memory_labels"
bash scripts/evaluate/launch_offline_8gpu.sh memory-labels \
  --qa-dirs "$RUN"/memory_qa/worker_*/ \
  --judge-model /private/yida_workspace/models/Qwen3-VL-8B-Instruct \
  --geneval2-source /private/yida_workspace/umm-anchored-eval-tools-d126833/GenEval2/evaluation.py \
  --minimum-confidence .8

"$SCORER_PYTHON" scripts/evaluate/merge_memory_probe.py \
  --qa-dirs "$RUN"/memory_qa/worker_*/ --label-dirs "$RUN"/memory_labels/worker_*/ \
  --output-dir "$RUN/memory_probe_report"
```

## 结果怎么解释

最终图像结果在 `$RUN/quality_report/summary.md` 和 `summary.json`。匿名图像及人工核查表位于同目录的 `blind_review`；质量代理分数不能替代人工质量判断。

问答结果在 `$RUN/memory_probe_report/summary.md` 和 `summary.json`。先检查 `label_coverage` 的有效标签数、unknown 比例，以及各 step/skill 的 VIT_IMAGE 准确率。ViT 校准不可靠或大多数状态是 unknown 时，不用 Memory 问答失败否定 Memory 内容。

再检查 DYNAMIC 对 SEED/EMPTY 是否提高实际内容准确率。重点看 prompt 与实际图像不同的样本，以及同一 prompt 不同 seed 导致实际状态不同的样本。只答出 prompt 要求的对象数量，不支持“获取了当前生成状态”的结论。候选概率是 log-likelihood 的相对 softmax，不是校准置信度。

标签依据同一时刻 `x_t-t·v_guided` 的图像估计。这是当前状态的可视代理，不是 noisy GEN token 的直接真值，也不是最终图像。问答 probe 只证明可读出的信息；GEN 是否有效使用它，还要看动态/静态/no-read 的最终图像结果。

本流程有中间 VAE 解码和离线问答。生成日志的时间不能用于部署预算声明；独立预算评测须关闭 probe。没有自动启动训练。
