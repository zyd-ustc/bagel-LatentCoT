# 两阶段 64 卡日程

这条入口是 `scripts/anchored_train_64.sh`。它和 `scripts/anchored_gen_only_64.sh` 分开。四个实例各 16 张卡，平台在每个实例上执行同一条命令。世界规模不是 64 时脚本退出。原生 BAGEL 保持冻结。训练没有启动，集群上的树仍是 `ed073a9`，这条代码还只在本地。

数据沿用 GEN-only 的读取方式：`ANCHORED_TRAIN` 是 caption-image JSONL，或现有 annotation index 目录。索引只打开 generation 的 train 视图，条件文本是 `goal_prompt`。不导出第二份 JSONL，也不打开 observe 视图。

Loop 配置默认仍是 `configs/anchored_gen_only.json`。采样窗口是 `u ∈ [0.15, 0.65]`（`t ∈ [0.35, 0.85]`），每个训练步都落在这个窗口里。日程在 `configs/anchored_curriculum_train.json`。

| 阶段 | 步数 | 学习率 | 训练的模块 | 轮数 |
|---|---|---|---|---|
| `gen_only` | 400 | `1e-4`，`adapter.up` 为 `2e-5` | 无状态残差和 gate | 1 |
| `dual` | 600 | `1e-4`，`adapter.mem_out` 为 `2e-5` | 冻结上一阶段的残差和 gate；训练 updater 和 `q_proj` / `k_proj` / `v_proj` / `mem_out` | 第 301 步起改为 2 |

`adapter.up` 是状态无关残差的输出投影。`adapter.mem_out` 是槽注意力的 `W_O`。`q_norm`、`k_norm` 和 `adapter.norm_m` 保持初始化。Stage 2 加载 Stage 1 的权重后不再更新那条重入通道，并冻结 C2-observe 读者。读者必须是 format-4 checkpoint，训练配置解析为 `gen_full_query`。`prompt_and_gen` 会在拷贝权重之前被拒绝。脚本不写死 checkpoint 路径，启动时设置 `OBSERVATION_CHECKPOINT`。

损失在原有 focal flow、anchor 和 entry 之外，加上系数 `lambda_improve=1` 的单边间隔。对每个图像 token，它使用通道上的平方和：

```text
mean_i relu(||v_corr,i - v_target,i||_2^2 - ||v_native,i - v_target,i||_2^2 + m)
```

`m` 是 `improvement_margin=1e-4`。原生误差不接收梯度。修正误差比原生误差至少低 `m` 时，这个 token 的间隔项为 0。平方和会随通道数放大，所以这项比按元素平均的 flow 项更大。`lambda_improve` 默认在旧调用里是 0，GEN-only 入口不打开它。

`train.jsonl` 记录当前训练 batch 的 `mse_0`、`mse_1`，两轮时还有 `mse_2`，以及 `improve` 和 `gate`。保存步和阶段结束时另写 `val.jsonl`。探针是启动时从 generation 的 val 划分抽出的 128 张图，之后每次检查都用这同一批图和同一批时间。索引目录直接使用其中的 val 划分；训练源如果是 JSONL，则要另给 `ANCHORED_VAL`。探针不打开 observe 视图。

Stage 1 的探针比较第一轮修正和原生速度的无权重均方误差，字段是 `v1_below_v0`。Stage 2 的探针把轮数固定为 1，因此量的是 `v1`：dual 是否低于冻结的 GEN-only 路径，以及 shuffled、zero 是否高于 dual。训练在第 301 步之后仍用最后一轮速度做损失，探针不改成 `v2`。这些比较写在日志里，不让训练中断，也不表示生成图变好。Shuffled 和 zero 只在这次检查里前向，不单独训练。
