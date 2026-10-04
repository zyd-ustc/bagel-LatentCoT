# UMM T2ILoop on BAGEL

本分支按 `UMM_T2ILoop_Anchored_Loop_Design.docx` 及用户后续 P0/P1 修订重构为 **Anchored GEN Loop + Memory Scratchpad**。
原生 prompt KV、同一 diffusion timestep 的 `x_t` 和 loop entry `G₀` 是固定 anchor。
主要循环状态是 GEN correction `ΔG`；memory 是可选的 UND 工作区。

```text
native prefix → G₀ → native body → G_base
                    extra body × R:
                    G₀ + A(ΔG, M) → body + GEN correction gate → G_r, M_r
                    G_base + α(G_r − G_base) → shared suffix → v_r
final velocity → one native Euler update of x_t
```

`runtime_loop_depth=R` **仅计额外 body 执行次数**。`allocated_max_loop_depth` 另设运行上限，默认 4。
α 跨轮共享，不再按 R 分配参数；Stage 1 可训练 R≤3，再验证 unseen R4。
旧 Current MemLoop 仅通过 `legacy_memory_only` 保留为冻结控制组。动态 prompt KV、pair-memory / teacher distillation、GRPO、FlowEdit
及显式反思链入口已经删除。旧 checkpoint 和旧配置不兼容新循环架构。
原生模型权重名称与计算路径保留；新 checkpoint 只保存 `t2i_loop` 参数。当前格式为 v4；拒绝 v1/v2/v3，其中 v3 的 gate 会反复缩小已有 GEN correction。

## 代码

| 文件 | 职责 |
| --- | --- |
| `qwen_latent_cot/bagel/anchored_loop.py` | 统一配置、re-entry、gate、memory、输出 merge、循环与直接 flow loss |
| `qwen_latent_cot/bagel/legacy_memory.py` | Current MemLoop 的独立兼容控制组，复用冻结 parent kernels |
| `qwen_latent_cot/bagel/flow_time.py` | 原生 logit-normal timestep 与 shift 变换 |
| `qwen_latent_cot/bagel/navit_loop.py` | 原生 prefix/base/body/suffix、MoT 路由和只读 KV |
| `qwen_latent_cot/bagel/modeling/bagel/bagel.py` | velocity、分支隔离、CFG 组合与原生 sampler |
| `qwen_latent_cot/bagel/inferencer.py` | T2I prompt cache、输入准备与 VAE decode |
| `scripts/train/train_t2i_loop.py` | adapter-only Stage 1 与可变循环深度 |
| `scripts/evaluate/t2i_loop_matrix.py` | 模式/深度对照、memory 干预和 timestep 诊断 |
| `scripts/evaluate/evaluate_t2i_loops.py` | 统一 GenEval2 / TIIF-spatial / quality / Repair-Damage 结果验证 |

完整方案的文本转录见 [设计原文](docs/UMM_T2ILoop_Anchored_Loop_Design.md)。
逐项实现与验收范围见 [实现对应表](docs/IMPLEMENTATION.md)。

## 安装与测试

在仓库根目录执行。先安装与目标 CUDA / Ascend 环境匹配的 PyTorch 与 torchvision。

```bash
pip install -e '.[dev]'
pytest -q
```

## Training-free topology 对照

需要原生 BAGEL-7B-MoT 权重、其 tokenizer 配置和 `ae.safetensors`。
参数 `--alpha` 是手动 topology intervention，不能当作训练收益。
所有 arm 使用同一初始噪声、prompt、CFG、schedule 与 body 区间。
循环开始后各 arm 的 sampler 轨迹会随 velocity 自然分叉。

```bash
python scripts/evaluate/t2i_loop_matrix.py \
  --model-path /path/to/BAGEL-7B-MoT \
  --prompts experiments/data/geneval2_hard_16.jsonl \
  --output-dir outputs/topology \
  --modes gen_only,gen_memory_anchored,memory_only,legacy_memory_only,direct_native_gen_only \
  --depths 0,1,2,3,4 --memory-slots 8 \
  --start-layer 16 --end-layer 24 \
  --alpha 0.1 --gate 0.02 --save-readouts
```

输出包含各 arm 的最终图、每轮 velocity、early/middle/late 的 `x0` 估计图和诊断记录。
`x0` 估计图是固定 `x_t` 上的 functional probe，不是完整采样得到的图。
`memory_only` 是新 workspace 拓扑消融；历史 Current MemLoop 对照必须使用 `legacy_memory_only`。
legacy 的 R 对应旧版 `memory_loop_repeat=1+R`，包括一次 strict Read 和 R 次 Write。
memory 经过原生 prefix、body 与 suffix；round0 阻止非 memory query 读取 memory；跨轮只 recycle M。
legacy 不使用新 gate、α 或 adapter，也不读取训练后的 memory 参数；m0 固定使用 parent 的 boundary + seed-0 slot noise。

`--memory-control correct|zero|frozen|shuffled` 提供新 workspace 的内容对照。
运行这些干预时，从 `--modes` 中移除 `legacy_memory_only`，保持历史控制组的原有计算。
`shuffled` 固定一个无自映射的 donor permutation，要求每个 batch 至少有两个样本。
独立 canonical writer 沿 correct recurrence 生成 Memory；GEN reader 只读取该 writer 在相同 layer depth 的 donor Memory。
reader 的 Memory 写入会被丢弃；adapter 的 memory→entry 读取也使用同一个 donor 映射。
每层记录 `memory_read_delta_ratio`。初始 M 在所有 sample 间相同，因此第一层读取差异为零；单层 body 的 R=1 不构成有效内容干预。
shuffled 多执行一次 writer body，日志记录实际 body pass 数；不能将其 wall time 当作与 correct 等计算量的对照。
比较 `K=8` 与 `K=16` 时分别运行上述矩阵。
未经训练、adapter 为零或仅使用固定 ΔG 映射时，GEN-only 初始 ΔG=0，等价于 native；它不能代表有效的训练收益。

纯 Direct Native 负对照为 `direct_native_gen_only`，强制 K=0。
它完整执行 `G_next=B(G_previous)`，绕过新 gate 与 α merge；共享 suffix 直接读取本轮 GEN endpoint。
可选 `direct_native_memory` 使用 K>0，必须显式选择。含糊的旧标识 `direct_native` 已移除。
`legacy_memory_only` 与两个 Direct Native arm 均忽略 `--alpha/--gate`；manifest 分 arm 记录实际 gate/readout 语义。

## Stage 1 训练

完成小规模 training-free 工程检查后可运行 Stage 1。未训练模块的语义增益不是 Stage 1 的前置条件；语义与质量门槛用于判断训练后的模块能否进入后续阶段。
训练集为 JSONL。每行包含 `prompt`、`image` 和可选 `bucket`。
相对图片路径按 JSONL 所在目录解析；`bucket` 支持 `ordinary`、`structural`、`easy`、`noop`。
首轮数据应排除复杂文字、风格和文化实体。

```json
{"prompt":"two red cubes left of a blue sphere","image":"images/0001.png","bucket":"structural"}
```

先修改 `configs/training/t2i_loop_stage1.yaml` 中的模型、数据和输出路径，再运行：

```bash
python scripts/train/train_t2i_loop.py \
  --config configs/training/t2i_loop_stage1.yaml
```

单机 8 卡使用 PyTorch DDP，每张卡加载相同冻结原生权重，仅同步新增模块的梯度：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train/train_t2i_loop.py --config configs/training/t2i_loop_stage1.yaml
```

`batch_size` 是每卡 batch size；设为 1 时全局 batch size 为 8，`steps` 仍是 optimizer step 数。
各 rank 共享 bucket 抽样与循环深度，分别获取 global batch 的数据分片，并独立抽取 timestep、noise 和 condition dropout。
可变分辨率使各卡 token 数不同；梯度按全局 packed token 数归一化，避免简单平均 rank loss 改变训练目标。
只有 rank 0 写日志和 checkpoint。日志保存各 rank 的 index、timestep、condition、图像长宽和循环诊断；checkpoint 记录 world size 与有效 batch size。

训练冻结全部原生权重。仅新增 adapter、每层 gate、共享 α 和 memory 参数可训练。
adapter 最后一层的 weight/bias 与 memory-to-entry projection 初始化为零；gate 默认 0.02。
GEN gate 保留已有 correction `d_in`，只缩放本层新增的 write：
`d_out = d_in + g × (loop_GEN_output − native_GEN_output − d_in)`，输出为 `native_GEN_output + d_out`。
这等于保留原生 layer update，并对 loop 相比原生路径新增的 residual update 门控；不再对全部 correction 连乘每层 gate。
GEN correction 在 body 内使用 FP32 累积；adapter、gate、α、memory 参数及 AdamW 状态使用 FP32，原生模型仍为 BF16。
checkpoint 加载保留新增参数的 FP32 精度。保存与加载均使用 v4，不能直接加载旧 v3 训练结果。
UND 和 memory 执行完整的原生 expert update，不经过 GEN gate。
GEN+Memory 默认 α=0，训练仍执行循环，以计算 α 梯度。
GEN-only 使用 `configs/training/t2i_loop_stage1_gen_only.yaml`，α=0.01；零 adapter 使初始输出仍与 native 完全一致。
GEN-only 的 α 与 adapter 同时为零会产生零梯度，训练入口会拒绝该配置。
GEN-only 从 adapter bias 的有效梯度启动；R≥2 后，上一轮 correction 为 low-rank adapter 提供输入。
日志同时记录 adapter weight/bias、gate、α 的梯度和参数范数。可据此区分“能执行”和“有实际更新”。

训练后可使用 `scripts/evaluate/stage1_8gpu.py --phase all --output-dir NEW_DIR --training-dir TRAIN_DIR --tools-dir TOOLS_DIR`。
入口检查八张 GPU 的可用显存和两份最终 v4 checkpoint；每张卡独立生成，不使用训练 DDP。
R=1–4 的 GEN-only 与 GEN+Memory 各占一张卡，LegacyMem 控制分配到浅层任务所在的卡；仅生成一次 Base。
同一 prompt 在所有 arm 中使用相同初始噪声；合并时强制检查 prompt、seed、noise hash 和完整 arm 覆盖。
默认生成 hard128 与 easy16，各 13 个 arm，共 1872 张最终图像；R4 标为 unseen，不作为已训练深度。
每个进程只加载一次原生模型，在作业间替换完整 loop module；关闭逐轮 suffix/诊断读出，保留最终采样输出。
随后八张卡按 prompt 分片评分。评分时映射局部 benchmark index，汇总前恢复全局 index，再统一计算 paired CI、Repair/Damage 与质量代理。
`--phase check` 仅检查资源；`--phase generate` 和 `--phase score` 可分开运行。独立评分必须保留原生成的 benchmark、seed、batch size 和 timestep 设置。
memory 初始化为 boundary embedding 加 `1e-4` 独立 slot noise。
训练默认开启诊断，逐轮记录 centered effective rank、pairwise cosine、slot std、sigma1 ratio 和 memory update ratio。
Stage 1 timestep 使用 `raw N(0,1) → sigmoid → timestep_shift`，与原生 BAGEL forward 共用实现。
训练 shift 读取模型配置（默认 1.0）；推理 schedule 的 shift 单独指定（默认 3.0）。
损失为最终轮 flow MSE，加上中间轮 flow MSE 的平均值乘 `loop_ds_weight`。
目标 velocity 为 `epsilon − x1`。训练不使用最终轮 self-distillation、RL 或 monotonic margin loss。

训练的 `max_train_loop_depth` 与 curriculum 最高深度一致；它不决定 checkpoint 的参数形状。
默认 `allocated_max_loop_depth=4`、`runtime_loop_depth=3`、`max_train_loop_depth=3`。
两个 Stage 1 默认配置均使用：前 30% `{1}`、中间 40% `{1,2}`、后 30% `{1,2,3}`。
每个阶段首次引入的新最大深度会立即执行，之后在本阶段的集合内均匀抽样。
若只训练 `{1,2}`，设置 `max_train_loop_depth=2`，仍可保持 allocation=4。
日志和 checkpoint 保存 `training_depth_counts`、`round_training_steps` 与实际阶段边界。
评估可运行 R0–4，分别标注 `seen`、`unseen`、`unknown`；Base、LegacyMem、Direct Native 标记 `training_free`。
旧 v1/v2 需要重新训练，不会静默迁移到共享 α。
workspace 的 `--memory-slots` 不得超过训练时的分配。

```bash
python scripts/evaluate/t2i_loop_matrix.py \
  --model-path /path/to/BAGEL-7B-MoT \
  --checkpoint outputs/umm_stage1/step_001000 \
  --prompts experiments/data/geneval2_hard_16.jsonl \
  --output-dir outputs/stage1_eval \
  --modes legacy_memory_only,gen_memory_anchored \
  --depths 0,1,2,3,4 --memory-slots 8 --save-readouts
```

训练使用原生 BAGEL 无裁剪 resize；一个 batch 可包含不同长宽和 GEN token 数。
默认 bucket 比例为 ordinary/structural/easy/noop = 40/30/20/10%，并分别记录 loss、α、gate。
默认 10% 文本 condition dropout 完全跳过 prompt segment，对齐 inference text-removed 分支；纯 T2I 暂不训练 image-removed。
普通推理只执行最终 suffix/readout。训练开启深监督或显式开启诊断时，才计算中间 readout。

统一结果 evaluator 的准备、命令和指标定义见 [结果验证](docs/RESULT_VALIDATION.md)。
它直接调用官方 GenEval2 scorer、原题 TIIF 本地 judge，并输出质量代理、invalid 和配对 Repair/Damage。
人工偏好独立导出和导入；缺失评分保留 null，不能由 velocity 日志推断。
必须同时评估结构任务分数、通用质量、invalid rate、Repair/Damage 和计算成本。
诊断变化或更低训练 loss 均不能单独证明语义提升。

## 当前范围

已实现第一版架构、training-free 对照和 Stage 1 训练路径。
Stage 2 workspace 专项训练与 Stage 3 body LoRA 是文档规定的后续实验阶段，尚未开放。
本次本地验收不包含完整 7B 模型的图像质量、benchmark 收益或 accelerator FLOPs。
GEN state 保持 packed `[N_total,D]`；memory 不跨 diffusion timestep 持久化。
