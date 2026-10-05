# BAGEL: denoiser 内部循环

`main` 仅保留冻结的原生 BAGEL、层内 GEN/Memory 循环、免训练生成和离线评测。旧 reader、OPD/GRPO、Anchored adapter 和训练入口已删除。历史结果留在 `docs/experiments`，不代表新架构的结果。

## 架构

固定一个 denoiser 输入 `x_t,t`。在 `[16,24)` 的每个 decoder layer 内执行 `N` 次 GEN 运算，残差步长为 `1/N`。`N=1` 直接调用原生实现。采样进度 `s=i/(S-1)` 从噪声走向图像；默认仅在 `s∈[0,0.5]` 启用循环。

Memory 从该层原生 prompt 输入状态中选取至多 16 个不同的内容 token。它保留源 token 的 RoPE，用原生 UND 的 norm、QKV、O projection 和 MLP 更新。第一次 GEN 运算不读 Memory；随后 GEN 读取更新后的 Memory KV。GEN 和 writer 在同一轮均读取更新前的状态。Memory 不跨 layer 或 timestep 传递；最后一次 writer 的 Q/attention/MLP 不执行。

prompt KV 只读。Memory KV 是临时 overlay，物理追加位置不会改变 RoPE。SOI/EOI 使用原生 UND expert，图像 token 使用 GEN expert。CFG 分支使用各自的 prompt/Memory。text-removed 分支没有 Memory，但保留同一 GEN 循环策略。packed 输入支持不同图像 token 数。每次 denoiser 只执行一次最终 suffix/readout。

五组免训练对照为 `BASE`、`GEN_LAYERWISE`、`MEMORY_DYNAMIC`、`MEMORY_STATIC`、`MEMORY_NO_READ`。静态 Memory 仍计算 writer，但丢弃更新，以匹配 writer 开销。no-read 必须与 GEN-only 数值一致。

原生来源和逐文件 SHA256 见 [docs/NATIVE_SOURCE.json](docs/NATIVE_SOURCE.json)。vendor 只改包名和 attention dispatch；CUDA 必须使用 FlashAttention，CPU 慢速实现仅用于测试。原始代码各文件的许可证声明保留。

## 在 H200 执行

环境需要 Python ≥3.10、PyTorch 2.5.1、Transformers 4.56–4.57 和 FlashAttention。质量 scorer 的 Qwen3-VL 需要 Transformers 4.57。原始 BAGEL 权重必须包含 `ema.safetensors` 或完整原生 shards，以及 `ae.safetensors`、配置和 tokenizer。禁止载入旧实验 checkpoint。

先执行原生一致性检查：

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-internal-20261005
CUDA_VISIBLE_DEVICES=0 /private/software/conda/envs/lcot/bin/python \
  scripts/evaluate/validate_native.py \
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT \
  --output outputs/native_parity.json
```

8 卡生成命令如下。`GPUS` 必须是已分配给本实验的卡；可设置为任意可用卡列表。每张卡独立加载冻结模型，按 prompt/seed 分片。相同 prompt/seed 在所有 arm 使用相同初始噪声。

```bash
export PYTHON=/private/software/conda/envs/lcot/bin/python
export MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
export PROMPTS="$PWD/data/engineering8.jsonl"
export OUTDIR=/private/yida_workspace/outputs/internal_loop_engineering
export GPUS=0,1,2,3,4,5,6,7
bash scripts/evaluate/launch_training_free_8gpu.sh \
  --stage engineering --seeds 0 --evaluations 2 \
  --arms BASE,GEN_LAYERWISE,MEMORY_DYNAMIC,MEMORY_STATIC,MEMORY_NO_READ
```

原生 `num_timesteps=50` 是 50 个时间点、49 次 denoiser 调用。baseline 与 loop 使用同样的原生 CFG、timestep shift 和采样设置。每条 prompt 可指定 `height,width`，默认 512×512。工程结果不能替代独立确认集。

单卡执行质量评测；BAGEL 生成完成后再加载 scorer：

```bash
CUDA_VISIBLE_DEVICES=0 /private/yida_workspace/umm-anchored-eval-tools-d126833/venv/bin/python \
  scripts/evaluate/quality_report.py \
  --manifests "$OUTDIR"/worker_*/manifest.jsonl \
  --benchmark "$PROMPTS" \
  --judge-model /private/yida_workspace/models/Qwen3-VL-8B-Instruct \
  --geneval2-source /private/yida_workspace/umm-anchored-eval-tools-d126833/GenEval2/evaluation.py \
  --output-dir "$OUTDIR/results"
```

评测报告包含 GenEval2 原始 soft-TIFA 的 GM、各 bucket、动态 Memory 对 GEN/static 的配对比较、Repair、Damage、净 Repair、quality proxy 和 invalid。TIIF 的 `yn_question_list/yn_answer_list` 使用明确标记的本地确定性 yes/no 协议。95% CI 按 prompt cluster 重采样，保留全部 seed 和 atom。scorer 失败不当作 0 分，不丢弃样本；修复后可继续评分。仅允许 `1e-6` 的概率边界舍入，原始 atom 分数保留。

`results/blind_review` 生成匿名 A/B 图像和 CSV；私有映射留在目录外。质量 proxy 不能替代人工盲评。生成日志中的计时只有工程用途；正式预算检查使用 3 次 warmup 与至少 20 次 batch=1 测量：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/evaluate/benchmark_budget.py \
  --model-path "$MODEL_PATH" --prompts "$PROMPTS" --output "$OUTDIR/budget.json"
```

`BASE_MATCHED_LATENCY` 需要先在 development 上校准 `--matched-base-timesteps`。确认集入口强制检查 384 条 prompt、固定 seeds、五组 arm 和排除集清单；训练准入仍需按研究方案核查 E0/E4/E5、质量与预算证据。当前没有训练入口。默认推理不计算诊断。需要检查 Memory 塌缩或读取时，可向生成命令添加 `--diagnostics`，记录每层、每轮的 effective rank、slot cosine/std、update ratio 和采样 query 的 attention read mass。这类运行不用于预算声明。

## 测试与方案

```bash
python -m pytest tests -q
```

本次 H200 工程结果见 [docs/experiments/internal_loop_20261005/REPORT.md](docs/experiments/internal_loop_20261005/REPORT.md)：两个预定窗口均未支持正向语义收益，当前不进入训练。

完整研究方案见 [docs/DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)，协议见 [docs/DENOISER_INTERNAL_MEMORY_PROTOCOL.yaml](docs/DENOISER_INTERNAL_MEMORY_PROTOCOL.yaml)。删除清单见 [docs/REFACTOR_REMOVALS.json](docs/REFACTOR_REMOVALS.json)。
