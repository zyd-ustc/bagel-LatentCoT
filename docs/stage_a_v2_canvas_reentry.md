# Stage A v2：预测画布重入与独立 Memory 注意力

本页描述当前实现。旧 Stage A 的训练结果保留在历史文档中，不能作为 v2 的质量或性能证据。

## 架构与兼容范围

保留 BAGEL 的 UND / GEN 全层交互、冻结的 C2 Reader、16 个 Memory slot、原生 CFG 和 Euler 采样。
Stage A 训练一次纠错，推理可以覆盖为 1–3 次；每轮重新写入 Memory、更新预测画布。
前半程 Euler 更新启用 Loop，后半程使用原版 BAGEL。原 SFT 的实现及 `bagel_full_loop_v3` 格式保持不变。

Stage A 使用新格式 `bagel_canvas_stage_a_v2`。旧 `bagel_full_loop_stage_a_v4` 的 adapter、bridge、优化器均不兼容。
用 format-4 C2 Observe/Feedback parent 重新初始化，不迁移旧 Stage A 的 consumer 权重。

## 预测画布代替末层 hidden

同一噪声状态 `x_t`、时间 `t` 下，第零轮用 partial Prompt 得到 `v0`。
每次反馈计算：

```text
canvas = stop_gradient(x_t - t * previous_velocity)
canvas_embed = vae2llm(canvas)
anchor = vae2llm(x_t) + time_embed(t) + latent_pos_embed
entrance = anchor + tanh(beta) * P(canvas_embed - anchor)
```

`P` 初始化为单位阵，`beta=0`。canvas 编码不重复添加时间或位置嵌入。
原生 GEN 会在入口之外继续使用原生 RoPE。末层 hidden 不再作为 Stage A 的重入输入。

## Memory 使用独立 Softmax

GEN 主分支直接调用原有 attention，实现和核函数均不替换。Prompt / Self Image 的 Q/K/V/O 不使用 consumer 增量，
也不对查询位置增加 16。原版 UND 权重负责 GEN 的文本缓存和边界 token；C2 UND 权重用于 Reader 和 Memory 文本监督。
临时切换使用参数绑定，保留反向传播保存的权重引用，并在激活重计算时重建相同作用域。

每层只给 Memory 分支的图像 Query 安装 rank-32 LoRA。该分支以归一化后的、未旋转的图像 Query
读取 UND 的未旋转 Memory K/V，使用内容注意力，不为 Memory 分配原生图文序列的位置。
图像入口已包含原生空间嵌入。独立 Softmax 不含 Prompt 或 Self Image token。

```text
memory_output = SDPA(image_query_with_consumer, memory_key, memory_value)
combined_image_attention = native_image_attention + tanh(alpha[layer]) * memory_output
output = frozen_o_proj_moe_gen(combined_image_attention)
```

每层 `alpha=0`，Memory 输出投影使用恒等映射。屏蔽 Memory 时直接执行原生 attention，跳过 consumer Query。
文本 token 和 CFG 参考分支不读取 Memory。`consumer_enabled=False` 只关闭 Query LoRA，仍可通过原生 Query 读取 Memory。
屏蔽 Memory 只消除该注意力分支；已训练的 canvas 重入仍可改变输出。

## Teacher、残差损失与零初始化启动

Teacher 使用冻结原版 BAGEL、完整 Prompt，以及与 student **同一个** `x_t` 和 `t`。
构造 Teacher 不再次读取图像、采样 VAE posterior 或抽噪声；full Prompt 的缓存不替换 student 的 partial 缓存。

```text
target = teacher_mix * stop_gradient(teacher_full_velocity)
       + (1 - teacher_mix) * (epsilon - x0)
velocity = stop_gradient(previous_velocity)
         + gamma * (raw_velocity - stop_gradient(previous_velocity))
```

默认 `teacher_mix=0.8`，可设为 1.0 做纯蒸馏。`gamma=0`，在条件速度头合成后才执行原生 CFG。
推理额外轮次复用单步训练的 gamma，合成锚点随轮次更新，不一直围绕 `v0`。

纠错损失比较 `velocity - stop_gradient(previous_velocity)` 与 `target - stop_gradient(previous_velocity)`：

```text
correction = residual_MSE + 0.1 * residual_cosine_loss
raw_correction = 同一损失，但用 raw_velocity 代替 velocity
total = correction + raw_correction_weight * raw_correction + 0.1 * semantic_CE
```

**对原方案的必要调整**：beta、alpha、gamma 同时为零时，raw_velocity 等于基线，
实际残差损失无法启动纠错分支。加入默认权重为 1 的 `raw_correction`，使 beta 和 alpha 先获得梯度；
随后 Query LoRA 和 gamma 开始学习。辅助损失不改变前向采样的零初始化结果。
首步 consumer 梯度为零符合该结构。BF16 也可能将训练早期很小的实际速度变化舍入为零。

每个 TrainingState 对应一张图。余弦方向按全图所有 token 和通道计算，数值稳定项为 `1e-4`；
目标残差严格为零时不计方向损失。Hard-Patch 权重和 Keep Loss 已从 Stage A 移除。
Memory contrast 默认关闭，验证仍保存 Memory-off / shuffled 对照。

日志中的 `correct_flow`、`round_flow`、`off_flow`、`shuffled_flow` 改为对混合 Teacher 目标的 MSE；
`base_flow` 仍是对数据集 `epsilon-x0` 的 MSE，不能直接与前四项相减。
新增 `cos_loss`、`raw_correction_loss`、`memory_gates`、`reentry_gate`。
`target_residual_rms` 使用混合 Teacher 目标；验证文件还记录 `round_cos` 和 `round_raw_correction`。

## 训练与采样

在部署主机的项目根目录执行。需要 BAGEL 权重、format-4 C2 parent、配对图文数据和可用 NPU。
必须使用新输出目录，不覆盖旧实验或恢复旧 Stage A。

```bash
TRAIN_STAGE=stage_a NPROC_PER_NODE=16 bash scripts/train_interaction_npu.sh \
  /absolute/path/stage_a_canvas_v2 \
  --parent /absolute/path/observe/best.pt \
  --corpus /absolute/path/observe_feedback_corpus \
  --steps 2000
```

也可用 `--train /absolute/path/train.jsonl --val /absolute/path/val.jsonl` 替代 `--corpus`。
默认训练单步、全局 batch 256；`--global-batch` 必须能被 worker 数整除。
新格式检查点可用 `--resume` 续训，保留 sampler、各 rank RNG 和优化器状态。

```bash
python -m qwen_latent_cot.interaction.cli sample \
  --model /absolute/path/Bagel-7B-MoT \
  --checkpoint /absolute/path/stage_a_canvas_v2/step_002000.pt \
  --caption 'Three red apples on a plate.' --seed 17 \
  --output /absolute/path/canvas_v2.png --device npu:0
```

零初始化对照使用 `--parent /absolute/path/observe/best.pt --config configs/interaction_stage_a.json` 替代 `--checkpoint`。
同样参数加 `--native` 得到原生基准。
推理 API：`engine.generate(prompt, seed=17, feedback_updates=3)`；默认仍为训练时的一次纠错。
`sample_feedback_sweep.py` 继续支持 Base / 1 / 2 / 3 / 4 轮，2–4 轮复用同一个速度门控。

## 验收与证据范围

本地完整回归为 107 项通过。CPU 小型真实 BAGEL MoT 测试覆盖：

- C2 权重不同于原版时，零初始化 1–3 轮采样与原生图片逐像素一致。
- 非零 consumer Query 不污染原生注意力；Memory 门为零或屏蔽 Memory 时恢复原生路径。
- canvas 与速度合成均使用上一轮速度；Memory 源位置不偏移。
- Teacher 共享 x/t，full 与 partial 缓存隔离。
- 零门启动、方向损失、激活重计算梯度一致、验证对照、保存和续训。

完整权重的 NPU 零初始化逐像素对照、1k–2k 步训练后的 32 条 Prompt 画风与语义检查、
以及 1–3 轮外推收益，仍需在部署主机执行。本次没有启动或停止远端任务。
独立 Softmax 和冻结原生投影消除了对应的结构干扰，但不能据此承诺训练后绝对不改变画风。
Teacher 增加一次无梯度前向；当前实现还会临时装载原版 UND 参数。
显存减半、速度翻倍需要实测，不能从纠错轮数减半直接推出。
