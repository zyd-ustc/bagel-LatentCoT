# 等时状态流训练

实现合同见 [等时生成态编辑](target_correction.md)，方案见 [训练方案](isochronic_state_flow_training_plan.md)。

每次更新只做一条单步 IFM：

1. 抽一个来源。Stage 1 只抽 `linear_pair`；Stage 2 按配置抽 L/I/R。约 10% 抽显式保持样本。
2. 抽一个支持时间 `t` 和编辑进度 `s~U(0,1)`。
3. L 数据用共享噪声构造 `a`、`b`，并用同一个前瞻函数生成 `P`；I/R 直接读取缓存的 `a,b,P`。
4. 由锚点 `a` 计算一次冻结 `b0`，用 joint prefix 计算 `uθ`。
5. 按有效区计算 `((uθ-(b-a))/t)^2`，反传、裁剪、更新。

不混入终图 L1、LPIPS 反传、旧 `L_early` 或 repulsion loss。

## 阶段

**Stage 1**：现有编辑对上的状态编辑学习。数据是 L，含 10% 保持样本。学习率 `5e-5` 线性 warmup 100 步，余弦衰减到 `1e-5`。预算 `min(6000, ceil(8 N_L / 64))`。

**Stage 2**：接续相容性适配。继续使用 Stage 1 权重并重置优化器。学习率 `1e-5` warmup 100 步，衰减到 `2e-6`。目标数据是 I 配对 1–2 万条、R 记录 2000–5000 条；缓存可以边生产边训练。有 R 时抽样 L:I:R=20:40:40；R 缺失时程序明确使用 L:I=20:80 并在日志记录。

## 命令

Stage 1：

```bash
u15-target-correction train \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_isochronic_state/stage1.json \
  --linear "$LINEAR_MANIFEST" \
  --output "$OUT" --device npu:0
```

Stage 2：

```bash
u15-target-correction train \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_isochronic_state/stage2.json \
  --linear "$LINEAR_MANIFEST" \
  --inverted "$INVERTED_CACHE" \
  --native "$NATIVE_CACHE" \
  --init-from "$STAGE1_CHECKPOINT" \
  --output "$OUT" --device npu:0
```

`--native` 可以省略。省略后抽样比例显式变成 L:I=20:80，检查点 `input_domains` 不会写 `native_history`。

## I/R 缓存生产

```bash
u15-target-correction prepare-state-pairs \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_isochronic_state/stage2.json \
  --mode inverted --linear "$LINEAR_MANIFEST" \
  --output "$INVERTED_CACHE" --device npu:0 \
  --cache-base-prediction

u15-target-correction prepare-state-pairs \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_isochronic_state/stage2.json \
  --mode native-parent --prompts "$PROMPTS" \
  --targets "$TARGETS" \
  --output "$NATIVE_CACHE" --device npu:0
```

`--targets` 只接受已经生成并通过质量审核的目标；没有可用目标生产接口时 R 不会伪造编辑记录。`--shard-index/--shard-count` 写 `manifest.shard_*.json`，训练入口指向同一目录即可合并。`--skip-existing` 用于续跑。

## 记录

`train.jsonl` 每步记录 `loss`、`gradient_norm`（裁剪前）、`source_kind`、`keep`、`time`、`progress`、调用数和实际 `input_domains`。检查点每 250 个更新及阶段末尾保存。`complete.json` 记录阶段、预算和是否使用 `--steps` 覆盖。
