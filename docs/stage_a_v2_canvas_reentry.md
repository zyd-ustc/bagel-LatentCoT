# Stage A v2.1：预测画布重入与独立 Memory 注意力

本页描述当前实现。旧 Stage A 的训练结果保留在历史文档中，不能作为 v2 的质量或性能证据。

## 架构与兼容范围

保留 BAGEL 的 UND / GEN 全层交互、冻结的 C2 Reader、16 个 Memory slot、原生 CFG 和 Euler 采样。
Stage A 训练一次纠错，推理可以覆盖为 1–3 次；每轮重新写入 Memory、更新预测画布。
前半程 Euler 更新启用 Loop，后半程使用原版 BAGEL。原 SFT 的实现及 `bagel_full_loop_v3` 格式保持不变。

Stage A 使用新格式 `bagel_canvas_stage_a_v2_1`。旧 `bagel_full_loop_stage_a_v4` 的 adapter、bridge、优化器均不兼容。
初版 v2 的 `bagel_canvas_stage_a_v2` 也不直接加载：本次门控含义和损失已改变。
用 format-4 C2 Observe/Feedback parent 重新初始化，不迁移旧 Stage A 的 consumer 权重。

## 预测画布代替末层 hidden

同一噪声状态 `x_t`、时间 `t` 下，第零轮用 partial Prompt 得到 `v0`。
每次反馈计算：

```text
canvas = stop_gradient(x_t - t * previous_velocity)
canvas_embed = vae2llm(canvas)
anchor = vae2llm(x_t) + time_embed(t) + latent_pos_embed
entrance = anchor + tanh(10 * beta) * P(canvas_embed - anchor)
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
combined_image_attention = native_image_attention + tanh(10 * alpha[layer]) * memory_output
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

默认 `teacher_mix=0.8`，可设为 1.0 做纯蒸馏。`gamma=1`，在条件速度头合成后才执行原生 CFG。
推理额外轮次复用单步训练的 gamma，合成锚点随轮次更新，不一直围绕 `v0`。

纠错损失比较 `velocity - stop_gradient(previous_velocity)` 与 `target - stop_gradient(previous_velocity)`：

```text
correction = residual_MSE + 0.1 * residual_cosine_loss
total = correction + 0.1 * semantic_CE
```

alpha、beta 保持零初始化，gamma 初始化为 1。初始 raw_velocity 已等于基线，故仍保持原生输出，
主损失可以直接训练内层门控。`raw_correction_weight=0`，已移除辅助损失及其额外速度输出。
首步 consumer 梯度仍为零；alpha、beta 先更新，之后 Query LoRA 开始学习。
入口和 Memory 门均使用 `tanh(gate_scale * parameter)`，配置 `gate_scale=10`。
日志记录缩放后真正生效的门值，而不是未缩放参数的 tanh。

每个 TrainingState 对应一张图。余弦方向按全图 token 和通道计算，数值稳定项为 `1e-2`。
另以 `clamp(stop_gradient(norm(predicted_residual)) / 1e-2, max=1)` 平滑启用余弦项：
零残差时由 MSE 启动，范数达到 0.01 后使用完整方向权重。
停止该启用权重的梯度，避免模型通过缩小残差来降低方向损失。
日志 `cos_loss` 为乘过启用权重后的方向损失，初始值为 0；不能与初版 v2 的未加权 cos_loss 直接比较。
目标残差严格为零时不计方向损失。Hard-Patch 权重和 Keep Loss 已从 Stage A 移除。
Memory contrast 默认关闭，验证仍保存 Memory-off / shuffled 对照。

日志中的 `correct_flow`、`round_flow`、`off_flow`、`shuffled_flow` 改为对混合 Teacher 目标的 MSE；
`base_flow` 仍是对数据集 `epsilon-x0` 的 MSE，不能直接与前四项相减。
记录 `cos_loss`、`memory_gates`、`reentry_gate`，移除 `raw_correction_loss`。
`target_residual_rms` 使用混合 Teacher 目标；验证文件还记录 `round_cos`，移除 `round_raw_correction`。

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

## 提速时保持训练合同

全局 batch 通过逐样本梯度累积实现；调大全局 batch 不会扩大前向 microbatch，反而增加每步样本数。
若目的是缩短相同步数的时间，应保持全局 batch 不变。

Stage A 跳过会被覆盖的 partial Prompt 文本读出。full 与 partial 相同时，Teacher 同时充当
原生 baseline，复用其冻结速度和逐层 KV；不再重复原生 GEN 或 full Prompt 预填充。
Prompt 有信息差时仍分别执行 full Teacher 和 partial baseline。
这些复用只发生在同一 x/t、同一 Prompt 的冻结计算之间，不改变训练目标。

可在显存足够时设 `activation_checkpointing=false`，用更多显存减少反向传播中的重计算。
续训允许改变该执行选项，仍恢复优化器、sampler 和各 rank RNG；全局 batch、数据、损失等合同仍严格检查。
切换前需测量真实样本的速度、峰值显存，以及 BF16 下的前向和梯度差异。

## 验收与证据范围

本地回归 107 项及新增专项 1 项通过。CPU 小型真实 BAGEL MoT 测试覆盖：

- C2 权重不同于原版时，零初始化 1–3 轮采样与原生图片逐像素一致。
- 非零 consumer Query 不污染原生注意力；Memory 门为零或屏蔽 Memory 时恢复原生路径。
- canvas 与速度合成均使用上一轮速度；Memory 源位置不偏移。
- Teacher 共享 x/t，full 与 partial 缓存隔离。
- 零门启动、方向损失、激活重计算梯度一致、验证对照、保存和续训。

初版 v2 已完成完整权重的 NPU 零初始化对照：512px、5 个采样节点，速度和图片均逐位一致。
v2.1 保持 alpha、beta 为零，gamma=1 的初始化合同由 CPU 1–3 轮采样测试覆盖。
训练后 32 条 Prompt 的画风、语义补全及 1–3 轮外推收益仍需后续实验。
独立 Softmax 和冻结原生投影消除了对应的结构干扰，但不能据此承诺训练后绝对不改变画风。
Teacher 增加一次无梯度前向；当前实现还会临时装载原版 UND 参数。
显存减半、速度翻倍需要实测，不能从纠错轮数减半直接推出。
