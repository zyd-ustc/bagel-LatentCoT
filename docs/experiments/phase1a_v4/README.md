# Phase 1A v4 — Reader Warm-up → Self-CoT OPD

当前入口先训练 **Phase 1A.0 Reader Warm-up**，通过 held-out 可读性检查后，
再进入 **Phase 1A.1a gate-only Self-CoT OPD**。以下是代码协议与运行方法，
不是 BAGEL-7B/H200 训练结果。

断点续训使用训练入口的 `--resume-checkpoint`；必须写入全新输出目录，
恢复 A/B、AdamW 和全局步数，`--max-steps` 为最终总步数。检查点、原始数据/
配置和 global batch 不匹配时拒绝恢复。旧 step-750 检查点兼容说明与只校验命令见
[续训说明](resume/README.md)。续训功能已准备；不代表正式续训已启动。

## 两个阶段的计算图

两阶段均采用 K=8、body `[12,20)`，从 layer-12 prompt **content** hidden 均匀取
M0；排除特殊/template/纯标点 token。短 prompt 使用 repeat + `1e-5` deterministic
jitter。Frozen strict Read 在当前 `(P,x_t,t)` 上只执行一遍 body，然后 STOP。
每层缓存 raw entry hidden，以及 attention 真正使用的 UND K/V：包括 input RMSNorm、
native projections、K norm、RoPE 和 native BF16 cast。缓存全部 detached，按层匹配；
不把最终 M20 投影后重复送给所有层。

独立 bank attention 使用当前层 native GEN Q（GEN RMSNorm、Q projection、Q norm、
RoPE 后的真实张量），GQA repeat 与 BAGEL 一致。memory-only 和 prompt-only readout
使用同一个 GEN Q，均经 frozen native GEN-O。它是独立 condition-bank readout；
native full attention 的 softmax 分母还包含当前 query/GEN，因此不是 native prompt
贡献的精确分解。

| 阶段 | 状态来源 | 优化目标 | 可训练参数 | GEN 注入 |
|---|---|---|---|---|
| Phase 1A.0 | native rollout，detached states | 各层 memory readout 对 detached prompt-bank target 的均匀 MSE | 低秩 translation A/B，B=0 初始化 | 无，velocity 必须与 native 一致 |
| Phase 1A.1a | 当前 student rollout，detached states | 同一 state 上拟合 frozen `[P;R]` teacher velocity 的 MSE | 每层一个 scalar injection gate | `alpha_l * readout_l`，alpha=0 初始化 |

Warm-up 的 readout 为 `GEN-O(Z_memory) + B(A(Z_memory))*alpha/rank`。
bank softmax 与 translation 残差累加采用 FP32；native Q/K/V 和 GEN-O 保留原生
BF16 数值，OPD 注入时转回主流 dtype，避免 warm-up 小更新被 BF16 残差相加量化掉。
OPD 加载并冻结这个 adapter；8 个 injection gate 是第一阶段全部可训练参数。
OPD 的 frozen Q 权重仍允许梯度穿过当前 hidden，传回更早的 gate。
没有 shuffle/ranking train loss、prompt mask、额外 Write round、Q LoRA 或 writer 更新。
reader bank 仅支持每卡 B=1。Warm-up 支持 `torchrun` CUDA 多卡同步训练：
每卡独立冻结模型，只广播/平均 A/B 参数与梯度，梯度先求均值再 clip。
rank 0 独占 heldout、日志和 checkpoint，其他 ranks 等待；OPD 仍是单卡入口。
8 卡时每 optimizer step 消耗 8 条 prompt（不是 8 个独立训练任务），
保持原 LR，global batch=8；不能把 5000 步与单卡 5000 条样本混为相同数据预算。

## 数据与配置

Warm-up 使用两个独立 JSONL：`reader_train.jsonl` 与 `reader_heldout.jsonl`。
最小字段为 string `prompt_id`、非空 `prompt`、允许的 `category`。例如：

```json
{"prompt_id":"train_0001","prompt":"Three red cubes surround one blue sphere.","category":"count","split":"train"}
```

允许类别：`count`、`spatial_relation`、`attribute_binding`、
`multi_object_composition`、`action_relation`、`rare_concept`、`reasoning_heavy_t2i`。
heldout 的 id 和 exact prompt string 都不能与 train 重合；heldout 应独立收集，
不能由 train 改写而来。代码检查 exact overlap；是否独立收集需要数据来源记录。
原有去重导出若缺少 category，需要先筛选/标注，不能直接作为该阶段的数据。

`scripts/data/prepare_reader_warmup_prompts.py` 可从用户指定 CORT train manifest
和独立官方 val manifest 导出 prompt-only 数据。仅显式计数/空间关系的正则匹配
样本保留，其他样本跳过；标签保存匹配证据，明确标记为 heuristic、非人工标注。
保留官方 heldout，不从 train 改写；训练中排除全部官方 val 的大小写/空白归一化
prompt 重叠。源清单/图片不修改，导出目录要求全新，统计和 SHA256 可复核。

`configs/training/memory_reader_warmup.yaml` 默认 512×512、NFE 参数 50、CFG=1、
shift=3、每次 native rollout 取 2 个 states、AdamW LR=1e-4、betas=(.9,.95)、
weight_decay=0。兼容方案里的 `num_memory_slots`、`memory_body_start/end` 别名；
内部统一为现有 `num_loop_tokens`、`memory_loop_start/end_layer`。
strict Read bank 每次在线重算，不导出离线 KV 训练数据。

## 执行顺序

在仓库根目录与已有 CUDA/PyTorch 环境执行。以下路径需要替换为实际文件；
各输出目录必须不存在。完整模型运行前，先执行第 1 步 preflight。

1. 检查 warm-up 配置和数据。

   ```bash
   PYTHONPATH="$PWD" python scripts/train/bagel_memory_reader_warmup.py \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/reader_train.jsonl \
     --heldout-prompt-data /path/to/reader_heldout.jsonl \
     --output-dir /path/to/reader_warmup --validate-only
   ```

2. 训练 reader side head。短程 pilot 可加 `--max-steps 10`。

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python scripts/train/bagel_memory_reader_warmup.py \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/reader_train.jsonl \
     --heldout-prompt-data /path/to/reader_heldout.jsonl \
     --output-dir /path/to/reader_warmup
   ```

3. 对指定 checkpoint 独立复核 heldout。训练期间也会自动保存最新的 `warmup_gate.json`；
   每份报告只对应其中记录的 checkpoint hash。评测时将 B 临时归零，在相同 native
   states 上重建 E_initial，再恢复 checkpoint 权重。此阶段不用看图。

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python scripts/evaluate/bagel_memory_reader_warmup_eval.py \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/reader_train.jsonl \
     --heldout-prompt-data /path/to/reader_heldout.jsonl \
     --checkpoint /path/to/reader_warmup/reader_warmup_step_0005000.safetensors \
     --output-dir /path/to/reader_warmup_eval
   ```

4. 准备 `bagel_t0_v2` 的独立 OPD train/heldout CoT cache，并取得 Teacher>Native 的
   heldout **semantic evidence**。builder 和 `--baseline-only` 使用 frozen base，
   不依赖 warm-up artifact。七段 schema、80–160 BAGEL tokens、拒答/显式数量检查
   保留；generation 为 deterministic。语义评测门槛与 JSON 字段见
   [既有 teacher baseline 协议](../phase1a_opd/README.md#execution-order)。

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python scripts/data/build_bagel_cot_teacher.py \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/opd_train.jsonl \
     --output /path/to/opd_train_cot.jsonl
   ```

5. 两个 gate 都通过后，加载 warm-up adapter，训练 zero-initialized injection gates。
   首先把同一命令加 `--validate-only` 检查 artifact/data provenance；然后运行训练。

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python scripts/train/bagel_memory_opd.py \
     --config configs/training/memory_opd_t0.yaml \
     --model-path /path/to/BAGEL-7B-MoT \
     --prompt-data /path/to/opd_train.jsonl \
     --teacher-cot-data /path/to/opd_train_cot.jsonl \
     --reader-warmup-checkpoint /path/to/reader_warmup/reader_warmup_step_0005000.safetensors \
     --reader-warmup-eval-json /path/to/reader_warmup_eval/warmup_gate.json \
     --teacher-baseline-json /path/to/teacher_baseline/teacher_baseline.json \
     --output-dir /path/to/opd_gate_only
   ```

## Readability gate 与评测控制

`warmup_gate.json` 记录 checkpoint/metadata/source SHA-256、固定 seed/schedule/
resolution、prompt ids、逐层指标和每个固定 state 的 correct/shuffled/zero MSE。
所有 arms 固定 recipient `(P,x_t,t)`，native 主路不注入，GEN Q 与 prompt target 一致。
shuffled bank 仅替换 donor prompt，也在 recipient 的同一个 `(x_t,t)` 上做 strict Read。

进入 OPD 必须同时满足：E_correct < E_initial、E_correct < E_shuffled、native 最大
绝对差 <=1e-6、slot 未完全塌缩。当前 slot operational guard 定义为所有层的 heldout
平均 `slot_mass_max < .99` 且 `slot_effective_count > 1.05`；报告明确记录阈值。
它是最小单-slot collapse 检查，不代替长期 utilization 曲线审查。
达不到条件时报告 `ready_for_opd=false`，OPD preflight 会拒绝该报告。
这两个 reader error 不等价于图像语义改善；OPD 的独立语义 teacher gate 继续保留。

Warm-up `metrics.jsonl` 包含逐层 MSE/cosine/relative error、target/readout/adapter RMS、
attention entropy/max、slot mass/effective count、memory hidden RMS/effective rank/slot cosine。
heldout 结果独立写入 `heldout_diagnostics.jsonl`。训练过程不把 shuffled/zero 输入 loss。
OPD 输出 `step0_parity.json`，检查加载 warmed adapter 后 gate=0 的 native parity。

## Artifact 与兼容性

Warm-up artifact schema 为 `bagel-memory-reader-warmup-v1`，只保存 `[12,20)` 的 A/B。
OPD schema 改为 `bagel-selfcot-opd-phase1a-v4`，只保存 gate tensors，并绑定 warm-up
checkpoint hash。OPD eval 需要同时传 `--reader-warmup-checkpoint`、
`--reader-warmup-eval-json`、`--adapter-path`（OPD gates）。模型路径、K/body、rank/alpha、
所有 tensor names/shapes 与源文件 provenance 都校验；原始数据文件要保留以便复核。

旧 raw-hidden、position-free、adapter-only OPD checkpoint 不兼容。旧 `bagel_t0_v1`
CoT cache 要重新生成，不能只改版本字符串。已有历史 loop/grounding 入口不参与 v4。
Warm-up 已实现 adapter/AdamW/global-step 续训，新 checkpoint 同时保存逐 rank RNG；
旧 step-750 没有完整 RNG，恢复模式明确标记为 `legacy_seeded_rollout`，不承诺
H200 位级一致。多卡采用显式参数/梯度同步，不是 DDP。OPD 仍无此续训实现；
Phase 1A.1b capacity relaxation、Q_mem LoRA、Draft-Verify 和 Phase 2 Write/loop
supervision 未实现。

CPU-only 结果诊断、生图评分测试与数据审计见 [offline/README.md](offline/README.md)。
原有 gate 不变；离线报告单独标出 correct-vs-zero 的证据边界。

本地机制与 runner 验证记录见 [VALIDATION.md](VALIDATION.md)。
