# Stage A：用 Prompt 信息差训练 Memory → GEN 的响应

Stage A 在 `loop-layer-NPU` 上实现。数据使用原来的 `annotation_index_v1` 图文对，
不使用新改写文本，也不生成伪图像。这里只训练 Stage A；Stage B 尚未实现。

## Prompt 如何切割

- 以完整句子或分号为单位。引号、括号内不切，逗号不切，不截断单词。
- 保留第一句作为场景主干，随机删除后续完整句子，删除词数尽量接近全文的 40%。
- 少于 30 词、没有安全句界的文本保持完整。可切分长文本另有 20% 保持完整。
- 随机种子来自可恢复的全局 sampler；各 rank 先产生相同抽样，再按 rank 分工。
- GEN 的 prompt cache、第一次预测和所有反馈轮都只看到 `partial`。
  UND Reader 单独接收 `full` 和当前 GEN 状态。完整文本不进入 GEN 的 prompt cache。
- 删除句子是已知的文本监督，不表示已经确认模型在图中漏画了这些内容。

在 v1 上按既有 source coverage 权重抽查 4,096 条，得到：

| 切割结果 | 条数 |
|---|---:|
| 删除完整分句 | 1,239 |
| 少于 30 词 | 2,155 |
| 没有安全句界 | 387 |
| 可切分但本次保留完整 | 315 |

发生删除时，平均删除词数比例为 40.64%。这套保守规则实际使 30.25% 的样本产生信息差，
并非所有样本都删除 40%。`scripts/inspect_stage_a_prompts.py` 可导出真实切割样例和统计。

## 训练信号

每个时间点先执行无梯度的原生条件预测，再执行两次全层反馈。
保持全部 block 循环、前半程采样时间点训练、16 个 Memory slot 和原生 BAGEL 注入路径。

- 训练 Write K/V LoRA、GEN Q/K/V/O LoRA、ReentryBridge 和每轮速度缩放 `gamma`。
  C2 UND Reader、BAGEL 主干和 VAE 保持冻结。
- 每轮实际输出为 `v0.detach() + gamma[r] * (v_raw - v0.detach())`，`gamma` 初始为 1。
  训练监督这个实际输出。采样时在条件分支输出头合成，再执行原生 CFG 和归一化；空文本分支不变。
- Flow 目标为配对图像的 `epsilon - x0`。使用已有 Hard-Patch 加权和 `keep_weight=0.02`。
- 每轮 Memory 用冻结的原生 UND 解码器预测删除句子；保留完整 Prompt 时预测完整描述。
  这条路径保留输入梯度，并关闭 Write/GEN adapter scope。
- 另一 family 的 Memory 作为无梯度 shuffled 参照。固定接收方的 x、t、partial Prompt、
  反馈入口 hidden、consumer 和 gamma，只替换最后一轮 Memory。
  排序项为 `relu(correct_MSE - shuffled_MSE.detach() + 0.01)`，不会训练 donor 路径故意变差。
- 总损失：各轮加权 Flow/keep 之和 + `0.1 * semantic_CE` + `0.1 * memory_contrast`。
  Memory-off 和 shuffled 对照、各轮 Flow、实际速度改动及目标残差幅度分别记录。

`gamma=1` 时残差 Flow MSE 与普通 Flow MSE 等价。
信息差、Memory 文本监督和对照项负责建立额外训练信号；显式残差本身不保证语义编辑能力。

## 使用原来的全量 v1 图文索引

训练索引：`/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus`。
原生 generation views 中有 train 37,141,166 条、val 41,009 条、test 40,505 条索引记录。
这些是索引记录数量，不是重新验证的唯一图像数。训练只抽取 train 的 generation 图文对，
按既有 sqrt(view-size) source coverage 抽样，保留原 family split；不加载整库到内存。

在 modelarts-job 的新代码目录运行：

```bash
TRAIN_STAGE=stage_a NPROC_PER_NODE=16 bash scripts/train_interaction_npu.sh \
  /data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_20261020 \
  --parent /data/zyd_workspace/bagel-LatentCoT/checkpoints/observe_highnoise_full_20261017/best.pt \
  --corpus /data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus
```

默认 20,000 步、全局 batch 256。可用 `--steps`、`--global-batch`、`--val-images` 覆盖。
已有配对 JSONL 也可用 `--train ... --val ...` 替代 `--corpus`，格式与 SFT 相同；建议携带 `family_id`。

续训用 `--resume /absolute/path/step_XXXXXX.pt` 替代 `--parent`，其余数据/配置不变。
保存优化器、各 rank RNG、sampler 位置、Prompt 切割配置、速度门控和适配器。
Stage A 使用独立格式 `bagel_full_loop_stage_a_v4`。既有 SFT/v3 入口和 checkpoint 继续使用原合同。

## 判断训练是否有效

先检查梯度有限、checkpoint 能恢复，再比较 held-out 数据的 Correct / Memory-off / Shuffled。
速度改动大还必须降低目标误差。最终要看生成图中的语义要求是否真的更准确，
不能用训练 loss 下降或图像变化替代这个判断。

## 本次实现检查

- CPU 全套回归：89 项通过。补充验证路径后，Stage A 的 7 项测试全部通过，
  包括双 Prompt 隔离、CFG 前残差、冻结 Reader 下的梯度、checkpoint 续训和 held-out 对照。
- modelarts-job：单 NPU 与 16 NPU 均完成真实 v1 图文对的优化步和 checkpoint 保存。
- 16 NPU 检查使用全局 batch 16；其中 3 条样本发生 Prompt 删除。
  单步 9.97 秒，最大峰值分配显存 40.21 GiB。writer、consumer、bridge 的梯度均有限且非零。
- 首步 Correct MSE 为 0.452831，Memory-off 为 0.451002，Shuffled 为 0.452857。
  首步 Correct 尚未优于 Memory-off；这些检查证明训练链路运行，不证明语义编辑改善。
- 保存位置：`/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_16npu_smoke_20261020/step_000001.pt`。
  完整日志和切割样例保存在本地 `artifacts/stage_a_20261020/`。

## 当前 64 卡任务

2026-10-20 14:50（Asia/Shanghai）在 modelarts-64 的 4 个实例上启动，
每实例 16 卡，全局 batch 256，计划训练 20,000 步。
初始化使用 C2 Observe 的 `best.pt`，从 Stage A step 1 开始，不复用旧 anchored 交互参数。

- 代码：`/data/zyd_workspace/bagel-LatentCoT/code/bagel-stage-a-20261020`
- 输出：`/data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_64npu_20261020_20000`
- 启动日志：输出路径加 `_launch/node0.log` 至 `node3.log`。
- master 为 VC_WORKER_HOSTS 中第一个实例，端口 29630。
- 旧 anchored 任务已停止，保留 `anchored_highrank_hybrid_20261020_10000/dual_step_007000.pt`。

若新任务中断，在原 4 个实例上分别指定 `NNODES=4`、各自的 `NODE_RANK=0..3`、
相同的 `MASTER_ADDR` 与 `MASTER_PORT=29630`，用 `TRAIN_STAGE=stage_a` 的训练脚本恢复。
保持同一输出与 corpus 路径，把 `--parent` 替换为最新完整 Stage A `--resume` checkpoint；
不要在当前任务仍运行时重复启动。

启动验收：确认推进到 step 22；step 10 checkpoint 可读取，包含 64 个 rank 的 RNG 和优化器状态。
首次验证完成 4 条记录、8 张对照图；保存及验证后训练继续推进。4 个实例各有 16 个训练 worker，
峰值分配显存 42.52 GiB。验收快照见 `artifacts/stage_a_20261020/launch64_acceptance.json`。
