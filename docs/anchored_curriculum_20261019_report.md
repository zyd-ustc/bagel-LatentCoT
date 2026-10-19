# Anchored Dual-State Loop：64 卡两阶段结果

日期：2026-10-19。数字来自这次跑完的目录

`/data/zyd_workspace/bagel-LatentCoT/outputs/anchored_curriculum_20261019`

里面的 `run.json`、`stage2.json`、`train.jsonl`（1000 行）和 `val.jsonl`（4 行）。本报告只记录速度场上的均方误差、门控和写入残差。没有调用采样器，没有生成图像，因此这些数字不表示出图变好或变差。

## 结论

课程按预定跑完：Stage 1 `gen_only` 400 步，Stage 2 `dual` 600 步。训练循环计时 `elapsed_seconds = 3011.7`（50 分 12 秒），最后一条验证写在 15:41:55。日志没有 Traceback。四张卡在结束后空闲。

固定的 128 张验证探针上，四次检查的第一轮修正 MSE 都低于同一次 native 前向。差距从 Stage 1 step 200 的 \(7.3\times10^{-5}\) 增到 Stage 2 step 600 的 \(1.27\times10^{-4}\)。Native MSE 是 0.359372，相对差距大约是它的 0.020% 到 0.035%。符号在这四次探针上一致，幅度很小。

Stage 2 把 Stage 1 的残差和门控冻住之后，`gate` 停在 **0.708340**。验证里 zero 干预和冻结的 gen_only 终点记录到同一数值 0.359284401。Shuffled 高于 dual-correct，也仍低于 native。两轮训练时，batch 平均的第二轮 MSE 与第一轮几乎重合，平均只高 \(2.6\times10^{-7}\)。

总损失全程停在 0.50 附近，没有下降趋势。它的大头是冻结 BAGEL 的 flow 误差，修正项只削掉 \(10^{-4}\) 这一档。

## 运行

| 项 | 值 |
|---|---|
| 机器 | 4 个实例 × 16 张 Ascend 910，`torch.distributed.run`，端口 29500 |
| 权重 | `/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT`，native 全冻结 |
| 代码 | `/data/zyd_workspace/bagel-LatentCoT/code/bagel-LatentCoT-anchored-20261019` |
| 配置 | `configs/anchored_curriculum_train_batch256.json`，`configs/anchored_gen_only.json` |
| 训练索引 | `datasets/observe_feedback_corpus`，fingerprint `4d4a16b4…bdde3`，`annotation_index`，文本字段 `goal_prompt` |
| `train_pairs` | 37,141,166 |
| `val_pairs` | 41,009，探针只用其中一次抽到的 128 张 |
| Reader | `checkpoints/observe_highnoise_full_20261017/best.pt` |
| Reader sha256 | `26520547f6baf74a4ce6b3981093141267197ab82619840837abbab72c6810da` |
| Reader 记录 | format 由加载器接受；`stage=C2-observe`，`source_mode=gen_full_query`，`source_policy=full_query`，全部层，query 为 neutral seed embeddings。Parent 是 `observe_nativeund_attn_20261015/best.pt`，parent 不决定 source mode |

同日第一次启动在 `state_free_delta` 的 NPU `index_put` 上失败：bf16 autocast 把目标张量收成 bf16，fp32 残差写不进去。残差改为在关闭 autocast 的 float32 里写完，再铸回 anchor 的 dtype。上面的 1000 步是这次修复后的重启。失败目录在 `outputs/anchored_curriculum_20261019_failed_dtype`。这次成功的代码还包含未提交的 gate 独立学习率、dual 起步时 `mem_out` 乘 0.01，以及 `mse_diff` / `gate` / `delta_ratio` 日志。

每个 step 抽 256 行。1000 步共 256,000 次抽取，约占 `train_pairs` 的 0.69%。抽样是否跨 step 重复，本报告不另作统计。`global_batch` 按卡均分后每卡顺序跑 4 张，所以增大 batch 加的是步时，不是单次激活。峰值显存 Stage 1 为 31.69 GiB，两轮阶段为 34.11 GiB，卡的 HBM 是 64 GiB。梯度裁剪阈值是 1，记录到的最大梯度范数是 0.049。

## 日程和可训练参数

Loop：`variant` 随阶段切换，body `[20, 28)`，`slots=16`，`rank=32`（4 head），`schedule=interval`，进度 \(u\in[0.15,0.65]\)，对应训练时间 \(t\in[0.35,0.85]\)，`timestep_shift=3.0`。`max_delta_rms_ratio=0.02`。`lambda_anchor=0.01`，`lambda_entry=0.001`。`round_loss_weights` 为空，flow 损失落在最后一轮速度上。`lambda_improve=1`，margin \(10^{-4}\)。分辨率 512，seed 17，weight decay 0。

| 阶段 | 步 | 训练对象 | 学习率 | 保存 |
|---|---|---|---|---|
| gen_only | 400，全程 1 轮 | `gate`、`adapter.norm_h/p`、`h_down`、`p_down`、`time_down`、`up` | base \(10^{-4}\)，`adapter.up` \(2\times10^{-5}\)，`gate.raw` \(5\times10^{-3}\) 且 weight decay 0，warmup 20 | 200、400 |
| dual | 600；step 301 起 2 轮 | `updater`、`q/k/v_proj`、`mem_out`。Reader、Stage 1 残差和 gate 冻结。`q_norm`、`k_norm`、`norm_m` 冻结 | base \(10^{-4}\)，`mem_out` \(2\times10^{-5}\)，warmup 20。没有 gate 组 | 300、600 |

进入 dual 的第一步把当时尚未训练的 `adapter.mem_out.weight` 乘了 0.01。`adapter.up` 和 gate 保持 Stage 1 结束时的值。

检查点只存 loop 模块、优化器和采样器，不存冻结的 BAGEL：

| 文件 | 大小 | 时间 |
|---|---|---|
| `gen_only_step_000200.pt` | 6.8M | 15:00 |
| `gen_only_step_000400.pt` | 6.8M | 15:08 |
| `dual_step_000300.pt` | 9.3M | 15:24 |
| `dual_step_000600.pt` | 9.3M | 15:41 |

`run.json` 写于 14:51。Step 计时合计：gen_only 1018 秒，dual 前 300 步 866 秒，dual 后 300 步 1065 秒。

## 日志里每个字段指什么

训练 `mse_0` 是该样本 native 速度对 flow 目标的均方。`mse_1` 是第一轮修正速度的均方。`mse_diff = mse_1 - mse_0`，先对 256 张做平均。负数表示这一批的第一轮修正比 native 更靠近目标。`error_fell_1` 是这一批里 `mse_1 < mse_0` 的样本比例，比较的是整张速度图的均方，不是逐 token。`mse_2` 只在两轮 step 出现，是第二轮速度的均方。`delta_ratio` 是最后一次写入残差在图像 token 上的 RMS，除以 anchor RMS，分母至少 \(10^{-6}\)。`gate` 是该 step 优化器更新前的 `gate.raw`。验证行里的 `gate` 是该 step 更新之后的值。

验证探针在开跑时抽一次 128 张，之后四次检查共用这些图、时间和噪声。探针把 `correction_rounds` 强制为 1，所以验证的 `val_mse_1` 永远是第一轮修正，不包含第二轮。Stage 2 另外在同一批图上跑冻结的 gen_only、dual zero、以及把另一张图的 feedback 当作 donor 的 shuffled。这些布尔值只做严格不等号比较，不停止训练，也没有设定“显著”的数值门槛。

## 验证探针

四次的 `val_mse_0` 都是 **0.35937178134918213**。Native 权重冻结，探针的图和时间也冻结，所以 native 误差应当保持不变。它确实保持不变。

| 检查点 | `val_mse_1` | 相对 native | 其他臂 |
|---|---:|---:|---|
| gen_only 200 | 0.35929834842681885 | \(-7.34\times10^{-5}\) | `v1_below_v0` true，gate 0.516625 |
| gen_only 400 | 0.3592844009399414 | \(-8.74\times10^{-5}\) | `v1_below_v0` true，gate 0.708340 |
| dual 300 | 0.3592597246170044 | \(-1.121\times10^{-4}\) | gen_only 0.3592844009399414；shuffled 0.35927408933639526；zero 0.3592844009399414 |
| dual 600 | 0.35924461483955383 | \(-1.272\times10^{-4}\) | gen_only 与 zero 仍是 0.3592844009399414；shuffled 0.3592701554298401 |

Step 600 上，从高到低（越低越靠近这一批 flow 目标）：

1. native 0.359371781
2. 冻结 gen_only = zero 0.359284401
3. shuffled 0.359270155
4. dual-correct 0.359244615

Dual-correct 比冻结 gen_only 低 \(3.98\times10^{-5}\)，比 shuffled 低 \(2.55\times10^{-5}\)。Shuffled 比冻结 gen_only 低 \(1.42\times10^{-5}\)。Zero 与冻结 gen_only 在记录精度上相同。Stage 2 的两个布尔值因此都为 true：`dual_below_gen_only`、`zero_rose`、`shuffled_rose`，且 `shuffled_measured` 为 true。Dual 300 的 shuffled 高出 dual-correct \(1.44\times10^{-5}\)，方向相同。

这组排序说的是：在这 128 张、一轮修正、同一个 \(t\) 上，Stage 1 残差已经占掉相对 native 的大部分 MSE 下降；记忆内容再往下移一点；把记忆换成另一张图的 feedback，下降比 correct 少一点，但没有回到 native。差值都在 \(10^{-5}\) 到 \(10^{-4}\)。

## 训练曲线

损失、flow 和 hinge 在 warmup 之后没有趋势。下面的均值是该阶段全部 step。

| 段 | 步数 | loss | flow | improve | mse_0 | mse_diff | error_fell_1 | delta_ratio | 步时（秒） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_only | 400 | 0.50201 | 0.48103 | 0.02098 | 0.36287 | \(-7.21\times10^{-5}\) | 0.783 | 0.00119 | 2.54 |
| dual，1 轮 | 300 | 0.50240 | 0.48138 | 0.02103 | 0.36353 | \(-1.114\times10^{-4}\) | 0.861 | 0.00168 | 2.89 |
| dual，2 轮 | 300 | 0.50231 | 0.48124 | 0.02107 | 0.36311 | \(-1.351\times10^{-4}\) | 0.888 | 0.00199 | 3.55 |

`mse_0` 在三段里都在 0.36 附近晃，这是每个 step 重新抽 \(t\) 和噪声带来的，不是拟合曲线。分段再看也一样：gen_only step 21–70 的 loss 均值 0.5018，step 331–400 是 0.5010；dual 后 50 步的 loss 均值 0.5002，前 50 个两轮 step 是 0.5028。

`mse_diff` 的符号：

| 段 | 小于 0 的 step | 等于 0 | 大于 0 |
|---|---:|---:|---:|
| gen_only | 387 / 400（96.8%） | 1（step 1，gate 为 0） | 12（3.0%） |
| dual 1 轮 | 300 / 300 | 0 | 0 |
| dual 2 轮 | 300 / 300 | 0 | 0 |

相对各自的 `mse_0`，`mse_diff` 的均值是 gen_only \(-0.020\%\)、dual 1 轮 \(-0.031\%\)、dual 2 轮 \(-0.037\%\)。

`error_fell_1` 在 gen_only 有 96.8% 的 step 超过 0.5，65.3% 的 step 超过 0.8。Dual 1 轮全部 step 超过 0.5，99% 超过 0.8。两轮阶段全部 step 超过 0.8。Step 600 这一批是 0.902。也就是大约九成样本的第一轮均方低于 native，但平均只低 \(10^{-4}\)。

两轮阶段的 `mse_2 - mse_1` 均值是 \(+2.63\times10^{-7}\)，中位数 \(+2.68\times10^{-7}\)。300 步里只有 42.7% 的 batch 平均满足 `mse_2 < mse_1`。`mse_2 - mse_0` 的均值是 \(-1.348\times10^{-4}\)，和 `mse_diff` 同一档。第二轮没有在这批训练 MSE 上再形成一层稳定的下降。验证没有测第二轮。

Hinge 项在 gate 离开 0 之后就停在 0.021 左右（gen_only step 1 是 \(10^{-4}\)，等于 margin 本身；step 400 是 0.02112；dual step 600 是 0.02141）。它没有收到 0。总损失里 flow 约 0.48，hinge 约 0.021。

## 门控和写入幅度

`gate.raw` 从 0 开始。Stage 1 的 gate 组学习率在 warmup 之后是 \(5\times10^{-3}\)。

| gen_only step | 更新前 gate | delta_ratio | mse_diff |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 20 | 0.0501 | \(5.4\times10^{-5}\) | \(+1.6\times10^{-6}\) |
| 50 | 0.210 | \(8.5\times10^{-4}\) | \(-2.4\times10^{-5}\) |
| 100 | 0.315 | 0.00100 | \(-5.4\times10^{-5}\) |
| 200 | 0.515 | 0.00135 | \(-8.7\times10^{-5}\) |
| 400（更新前） | 0.70756 | 0.00153 | \(-9.8\times10^{-5}\) |

Step 400 的验证行记的是更新后的 0.708340。Dual 的 600 步里这个值不变，记录的最小和最大都是 0.708340。梯度范数从 gen_only 末的约 0.009 增到 dual 1 轮末的 0.033，两轮阶段均值 0.028。

`delta_ratio` 在 gen_only 末是 0.00153，dual step 600 是 0.00217。上限是 0.02，所以最后一次写入大约用了上限的 11%。图像 token 上的残差 RMS 约为 anchor RMS 的 0.2%。

## 步时

| 段 | 均值 | 中位数 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| gen_only，含 step 1 的 6.65 秒 | 2.54 秒 | 2.51 | 2.17 | 6.65 |
| gen_only 最后 50 步 | 2.39 秒 | | | |
| dual 1 轮 | 2.89 秒 | 2.87 | 2.55 | 3.37 |
| dual 2 轮 | 3.55 秒 | 3.53 | 3.10 | 4.46 |

两轮相对一轮大约是 1.23 倍，不是 2 倍。

## 这份结果没有覆盖的事

采样网格仍是 native 的 50 步。`generate_image` 没有接到这次的 checkpoint 上。验证和训练都只走了 conditional 一支，没有 CFG。Body 层段、窗口、focal 权重和 hinge 的 margin 都是这次的设定，不是扫出来的。Reader 的 parent 和 C2 契约只说明读取来源是 `gen_full_query`，不说明这次生成误差的下降来自读取质量。

可以接着做、但还没做的测量，是用 `dual_step_000600.pt` 和 `gen_only_step_000400.pt` 在同一套提示上采样图像，再和 native、zero、shuffled 比图。在那之前，上面的 MSE 差就是这次运行的全部结果。
