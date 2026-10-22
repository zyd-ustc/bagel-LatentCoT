# 方法：冻结 GEN 上的空间 UND 速度纠错

## 1. 问题设定

BAGEL 的生成侧是 flow-matching 的 MMDiT：给定 `x_t = (1−t)·x0 + t·ε`，网络预测速度 `v`，采样用 Euler 从 `t=1` 推到 `t=0`。实际部署时还会做 CFG 和 global renormalization，得到 `v_guided`。

本项目不改动这个骨干，只在冻结骨干之外训练一个**速度修正头**：

```text
v = v_native + r,    r = Corrector(x_t − t·v_native, t, prompt, GEN 同层 KV)
```

约束是：`r = 0` 时必须与原生采样完全等价；GEN 的参数、注意力、hidden state 和输出头都不参与训练。

## 2. 原生通路（全部冻结）

`qwen_latent_cot/correction/runtime.py` 把原生 BAGEL 包装成一个只读运行时：

- `flow(prompt)`：文本侧前向得到 prompt KV cache，并准备图像 latent 的 packed token、位置和 CFG 分支缓存。
- `predict(x_t, t, flow, capture)`：调用原生 `_forward_flow` 得到 conditional 速度；同时用 forward pre-hook 抓取每层 GEN 的 image-token K/V（prompt K/V 来自 cache，二者拼接后 detach）。
- `reference(...)`：无条件分支，只用于 CFG。
- `proposal(x_t, t, flow)`：返回 `v_native = guided(conditional, reference)` 和每层 GEN 上下文。
- `encode/decode`：VAE 后验与解码。训练时同一条样本的 VAE 后验使用固定种子（`crc32(sample_id) + seed`），噪声与时间步每次独立采样。

CFG 与 BAGEL 原生实现一致：`reference + scale·(conditional − reference)`，再按 `cfg_renorm_type` 做 global/channel 归一化并截断。

所以 "GEN 同层 KV" 是**冻结上下文**，不是可训练的 Memory bank；这里没有回灌、没有第二遍 GEN 前向，也没有语义 CE。

## 3. 纠错器

`qwen_latent_cot/correction/model.py` + `attention.py`：

| 模块 | 实现 |
|---|---|
| 主干 | 原生预训练 UND 的独立 `deepcopy`，28 层 |
| 适配 | 每层 `q/k/v/o` 加 rank-64 LoRA（`b` 零初始化，`base` 冻结） |
| 输入 | `proposal = x_t − t·v_native`（detach）→ 原生 `vae2llm` → 单位阵初始化的 `input_projection` → `+ time_embedder(t)` → `+ latent_pos_embed`（原生 2D 位置） |
| 每层计算 | 原生 LayerNorm → q/k/v → 原生 RoPE（图像 token 的原始位置）→ 与 `[冻结 prompt/GEN KV ; 本层 patch K/V]` 拼接做 SDPA → `o_proj` → 残差 → MLP |
| 输出 | `decoder.norm` → 每 patch 一个 query → FP32 `Linear`，输出该 patch 的速度修正（`latent_channel × patch²` 通道） |

- 一个图像 patch 对应一个纠错 query，保留原生 2D 位置嵌入和当前时间。
- 输出 head 零初始化 + 输入投影单位阵 ⇒ 初始化时 `r = 0`，采样与原生逐像素一致。
- 修正头用 FP32 输出，避免小幅修正被 BF16 速度转换吃掉。
- 参数分三组（`und` / `input` / `head`）分别记录梯度范数。总计 53,444,672 个可训练参数。

### 全局常量场基线

`calibration.py` 为每个启用的时间节点维护一个常量速度修正，用训练目标的最小二乘均值估计（`sum/count`，分布式 all-reduce 累积）。它**不参与**纠错器的 loss 或优化器，只作为采样时的第三条对照臂（native / global / corrected），用来回答"纠错器是否学到了比时间相关常数更多的信号"。

## 4. 训练目标

### 4.1 直接残差（前两轮）

```text
target = ε − x0 − v_base
loss   = MSE(r, target)
```

- `correction_space` 缺省（v1，1000 步实验）：`v_base = v_conditional`，修正加在条件速度上，之后再做 CFG。
- `correction_space = "post_cfg"`（5000 步实验）：`v_base = v_guided`，修正直接加在 CFG 之后的速度上。

### 4.2 guidance-safe 条件残差（当前主线）

诊断发现 post-CFG 目标里的 `−(v_guided − v_conditional)` 会被网络学成"抵消 CFG"。于是把目标改成只保留 guidance 不冲突的分量：

```text
g      = v_guided − v_conditional
H_g(z) = z − min(⟨z, g⟩ / ‖g‖², 0)·g        （g = 0 时恒等）
target = H_g(ε − x0 − v_conditional)
loss   = MSE(r_raw, target)
r_used = H_g(r_raw)                          （推理时，global 基线同样投影）
```

三个关键取舍：

1. **loss 用原始输出，不对 `r_raw` 先投影。** 否则沿 `−g` 的预测会落进零梯度区，训练信号消失。
2. **推理与训练共用同一个投影。** 部署时实际施加的修正一定满足 `⟨r, g⟩ ≥ 0`，因此 `applied_negative_fraction` 应恒为 0；`raw_negative_fraction` 可以在训练中为正。
3. **该目标不再等价于完整 guided-flow 回归。** 它有意排除了反向 guidance 残差，所以 `loss` 与 `base_mse`（物理 guided 速度误差）不是同一个量的不同实现，**不能**用二者算"改善百分比"。

### 4.3 已知代价

- 若某个有益修正恰好沿 `−g` 方向（例如压低过饱和），会被投影直接删除。
- 目标改的是回归任务本身，不是模型容量；语义是否改善仍要由实际图片判断。

## 5. 采样

`qwen_latent_cot/correction/sampling.py`：

- 50 个时间节点、`timestep_shift = 3`，前 `ceil((50−1)/2) = 25` 次更新启用纠错，之后纯原生。
- 每个启用节点只做一次纠错、一次 GEN proposal，然后推进外层 Euler：

```text
x ← x − (t − t_next)·v_native          # 原生 BF16 乘积，保持逐像素一致
x ← x − (t − t_next)·r_used            # FP32 只加一次
```

- 三臂共用同一采样器与 seed：`native=True`（无纠错）、`global_only=True`（只加常量场）、默认（完整 UND 纠错器）。
- 轨迹 JSON 记录每步的 `delta_rms`、`gen_passes`、`und_passes`，以及 guidance-safe 模式下的 `raw_delta_rms`、`guidance_projection_raw/applied`。

## 6. 训练契约

| 项 | 取值 |
|---|---|
| 数据 | 天然图文对（generation view，完整 caption），沿用已有 family 划分与 image-disjoint split |
| 全局 batch | 256（4 节点 × 16 NPU，per-rank micro-batch 1，累积 4） |
| 优化器 | AdamW，lr 1e-4，weight decay 0，clip 1.0，warmup 100 |
| 精度 | bfloat16 autocast；输入投影与输出 head 为 FP32 |
| 分辨率 | 512，VAE 后验固定种子 |
| 检查点 | 只存可训练参数、optimizer、sampler、provenance、每 rank RNG、calibration；冻结权重从原生模型重建 |
| 续训 | 要求同原生模型、同 corpus fingerprint、同方法/优化配置、同 world size；仅允许改 `steps`/`save_steps`/`val_images`/`activation_checkpointing` |
| 冻结保护 | `FrozenGuard` 每步检查冻结参数无梯度且 `_version` 未变 |

保存点形如 `step_005000.pt`，训练日志为 `train.jsonl`（每步一行，含 loss、base_mse、delta_rms、各组梯度范数、显存、耗时）。

## 7. 度量定义

| 字段 | 含义 |
|---|---|
| `loss` | 原始预测对"当前配置目标"的 MSE |
| `zero_correction_mse` | 同一目标下零输出的 MSE（即基线） |
| `base_mse` / `applied_base_mse` | 原生 / 实际纠错后速度对 `ε − x0` 的物理 MSE |
| `residual_mse` | 验证时实际施加修正后的物理残差 |
| `delta_rms` | 实际施加修正的均方根 |
| `conditional_base_mse` / `conditional_corrected_mse` | 相对条件速度的物理误差 |
| `guidance_projection_raw/applied` | `⟨r, g⟩/‖g‖²`，投影前后的系数 |
| `raw/applied_negative_fraction` | 系数为负的比例；`applied` 应在容差内恒为 0 |
| `target_anti_removed_rms` | 目标中被投影删除的分量大小 |

解释规则：`loss` 只能和 `zero_correction_mse` 比（同一目标）；`base_mse` 只能和 `native` 臂的 `base_mse` 比（同一物理量）。跨这两组算比例会得到误导性数字，历史文档中已明确禁止这种写法。

## 8. 与旧架构的关系

这条分支上的 Memory bank、consumer、reentry、semantic/partial-prompt 训练、Stage A 与 C2 方案都已被删除，只保留原生 BAGEL、语料、历史 checkpoint 与生成图片。检查点带 `kind` 校验，旧架构的权重不能用来续训当前模型。

## 9. 已知局限

- **离线误差与采样轨迹不一致**：训练在单步状态上回归，纠错后的状态会改变后续 GEN 输入，单步误差下降不代表整条轨迹更好。
- **表征对齐有限**：UND 只能通过冻结 KV 与 LoRA 适配 GEN，容量与接口都比"GEN 自己输出速度"窄。
- **只验证了 512px / 50 步 / shift 3 / CFG 4**；其它调度与分辨率未测。
- **guidance 先验很强**：`H_g` 假设"抵消 guidance 一律有害"，这在当前测量中成立，但不是普适结论。
