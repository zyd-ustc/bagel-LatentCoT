# UMM T2ILoop on BAGEL

本分支按 `UMM_T2ILoop_Anchored_Loop_Design.docx` 重构为 **Anchored GEN Loop + Memory Scratchpad**。
原生 prompt KV、同一 diffusion timestep 的 `x_t` 和 loop entry `G₀` 是固定 anchor。
主要循环状态是 GEN correction `ΔG`；memory 是可选的 UND 工作区。

```text
native prefix → G₀ → native body → G_base
                    extra body × R:
                    G₀ + A(ΔG, M) → body + GEN correction gate → G_r, M_r
                    G_base + α_r(G_r − G_base) → shared suffix → v_r
final velocity → one native Euler update of x_t
```

`loop_depth=R` **仅计额外 body 执行次数**。它与旧版含首轮 Read 的 `loop_depth` 含义不同。
旧 Current MemLoop 仅通过 `legacy_memory_only` 保留为冻结控制组。动态 prompt KV、pair-memory / teacher distillation、GRPO、FlowEdit
及显式反思链入口已经删除。旧 checkpoint 和旧配置不兼容新循环架构。
原生模型权重名称与计算路径保留；新 checkpoint 只保存 `t2i_loop` 参数。当前格式为 v2；v1 的 whole-layer gate checkpoint 会被拒绝。

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
  --modes gen_only,gen_memory_anchored,memory_only,legacy_memory_only,direct_native \
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
`shuffled` 在 batch 维度交换 memory，要求每个 batch 至少有两个样本。
比较 `K=8` 与 `K=16` 时分别运行上述矩阵。
未经训练、adapter 为零或仅使用固定 ΔG 映射时，GEN-only 初始 ΔG=0，等价于 native；它不能代表有效的训练收益。

## Stage 1 训练

仅在 topology 实验达到文档中的语义与质量门槛后运行训练。
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

训练冻结全部原生权重。仅新增 adapter、每层 gate、每轮 α 和 memory 参数可训练。
adapter 最后一层的 weight/bias 与 memory-to-entry projection 初始化为零；gate 默认 0.02。
GEN gate 使用 `native_GEN_output + g × (loop_GEN_output − native_GEN_output)`。
UND 和 memory 执行完整的原生 expert update，不经过 GEN gate。
GEN+Memory 默认 α=0，训练仍执行循环，以计算 α 梯度。
GEN-only 使用 `configs/training/t2i_loop_stage1_gen_only.yaml`，α=0.01；零 adapter 使初始输出仍与 native 完全一致。
GEN-only 的 α 与 adapter 同时为零会产生零梯度，训练入口会拒绝该配置。
memory 初始化恢复为 boundary embedding 加 `1e-4` 独立 slot noise；日志记录 centered effective rank 和 pairwise cosine。
Stage 1 timestep 使用 `raw N(0,1) → sigmoid → timestep_shift`，与原生 BAGEL forward 共用实现。
训练 shift 读取模型配置（默认 1.0）；推理 schedule 的 shift 单独指定（默认 3.0）。
损失为最终轮 flow MSE，加上中间轮 flow MSE 的平均值乘 `loop_ds_weight`。
目标 velocity 为 `epsilon − x1`。训练不使用最终轮 self-distillation、RL 或 monotonic margin loss。

配置中的 `loop_depth` 是参数分配的最大深度，`depth_curriculum` 是当前采样的训练深度。
评估 checkpoint 时 `--depths` 不得超过该分配上限，`--memory-slots` 不得超过训练时的分配。

```bash
python scripts/evaluate/t2i_loop_matrix.py \
  --model-path /path/to/BAGEL-7B-MoT \
  --checkpoint outputs/umm_stage1/step_001000 \
  --prompts experiments/data/geneval2_hard_16.jsonl \
  --output-dir outputs/stage1_eval \
  --modes gen_only,gen_memory_anchored \
  --depths 0,1,2,3 --memory-slots 8 --save-readouts
```

外部 GenEval2 scorer 的分数可用 `scripts/evaluate/geneval2_report.py` 汇总。
必须同时评估结构任务分数、通用质量、invalid rate、Repair/Damage 和计算成本。
诊断变化或更低训练 loss 均不能单独证明语义提升。

## 当前范围

已实现第一版架构、training-free 对照和 Stage 1 训练路径。
Stage 2 workspace 专项训练与 Stage 3 body LoRA 是文档规定的后续实验阶段，尚未开放。
本次本地验收不包含完整 7B 模型的图像质量、benchmark 收益或 accelerator FLOPs。
单个 packed batch 要求相同 image-token 数量；memory 不跨 diffusion timestep 持久化。
