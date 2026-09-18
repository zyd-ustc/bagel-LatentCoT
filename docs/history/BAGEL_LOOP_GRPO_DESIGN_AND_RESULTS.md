# BAGEL 同 timestep 循环与 GRPO：原理、实现和结果记录


| 字段    | 内容                                       |
| ----- | ---------------------------------------- |
| 文档状态  | 持续维护                                     |
| 方案版本  | Loop GRPO V2                             |
| 结果快照  | 2026-09-13 14:37 CST                     |
| 基础模型  | BAGEL-7B-MoT                             |
| RL 配置 | `configs/training/loop_grpo_v2.yaml`     |
| RL 入口 | `scripts/train/bagel_loop_grpo_train.py` |


本文只记录已经确定的设计、代码行为和实验观测。V2 的训练结果是运行中快照，不是完整 1000 step 的最终结果。

## 1. 任务定义

目标是在不引入新视觉 latent、视觉 delta 预测器或独立反思 expert 的条件下，提高 BAGEL 文生图在 attribute、counting 和 position 上的准确率。

最终主链保留 BAGEL 原生文生图分布：

- prompt 仍由 BAGEL 的文本路径编码；
- 图像状态仍是 BAGEL 原生 16 通道 VAE latent；
- flow velocity 仍由 BAGEL 原生 `llm2vae` 输出；
- 不生成反思文字，不把图像解码后重新输入模型；
- 不在去噪中途切换 prompt；
- 不训练新的视觉表征空间或输出头。

额外计算发生在一次 BAGEL flow forward 内部，因此循环前后使用相同的 `z_t`、timestep、prompt KV cache 和 position IDs。

## 2. BAGEL 内部循环



### 2.1 层级执行

BAGEL 的 28 层 Transformer 被划分为：

```text
prefix: layers [0, 8)
body:   layers [8, 20)
suffix: layers [20, 28)
```

depth 1 是原始 BAGEL 路径：

```text
h0 = prefix(input)
h1 = body(h0)
v1 = llm2vae(suffix(h1))
```

depth 2 在同一 forward 内重复 body：

```text
h0 = prefix(input)
h1 = body_base(h0)
h2_raw = body_base+loop_LoRA(h1)
h2 = bounded_residual_merge(h1, h2_raw)
v2 = llm2vae(suffix(h2))
```

prefix 只执行一次，body 执行两次，suffix 对所需 depth 的 body 输出执行。循环执行时禁止更新 prompt KV cache，并禁用 TaylorSeer。

### 2.2 MoT 路由

生成模式下：

- text/boundary tokens 使用 BAGEL 的共享 understanding 路径；
- VAE tokens 使用 generation expert；
- 两类 token 保持在 BAGEL 原生 joint attention 中交互。

V2 在额外 body pass 中同时启用 generation attention LoRA 和 text K/V LoRA。depth 1 全程关闭这些 LoRA，因此 depth 1 保持原始 BAGEL 执行路径。

### 2.3 timestep 门控

采样器先使用 BAGEL 的原生 timestep shift：

$t' = \frac{s t}{1 + (s-1)t}.$

V2 使用 `timestep_shift=3.0`。只有 shift 后的 t'\ge 0.75 执行 depth 2；其余 timestep 执行 depth 1。在 30-point schedule 对应的 29 次 flow evaluation 中，循环激活 15 次。

### 2.4 hidden residual 限幅

令 base body hidden 为 h_1，第二次 body 的输出为 \tilde h_2，则：

\Delta h = \tilde h_2-h_1,

c = \min\left(1,\frac{\gamma\operatorname{RMS}(h_1)}{\operatorname{RMS}(\Delta h)+\epsilon}\right),

h_2=h_1+\alpha c\Delta h.

限幅按 token 独立计算。当前 `residual_scale=γ=0.05`、`α=1`；当 `α=0` 或 `γ=0` 时直接返回 `h1`。

## 3. 参数更新范围



### 3.1 V2 可训练参数

V2 在 layers `[8,20)` 注入 rank-8、alpha-16、dropout-0 的 loop-gated LoRA：

```text
self_attn.q_proj_moe_gen
self_attn.k_proj_moe_gen
self_attn.v_proj_moe_gen
self_attn.o_proj_moe_gen
self_attn.k_proj
self_attn.v_proj
```

前四项作用于 generation expert 的 attention 投影；后两项作用于文本 token 使用的共享 attention K/V 投影。共有 144 个 LoRA A/B tensor，2,949,120 个可训练参数。

LoRA 参数和优化器 master weights 使用 FP32；BAGEL forward 使用 BF16。每个 LoRA 的计算为：

y=W_0x+\frac{\alpha}{r}B(Ax),

其中 W_0 冻结，LoRA 分支只在额外 loop pass 激活。

### 3.2 冻结参数

以下参数保持冻结：

- BAGEL UND/GEN 基座参数；
- attention 基座权重、全部 MLP 和 norm；
- token embedding、time embedding 和 position embedding；
- ViT、VAE、`vae2llm` 和 `llm2vae`；
- reward model；
- RL 启动时复制的 reference adapter。

代码通过 trainable-name allowlist 检查更新范围。8 卡训练在每个 optimizer step 前对 LoRA gradient 做 all-reduce 和 world-size 平均。

## 4. SFT 初始化

RL policy 由 loop SFT step 3000 初始化：

```text
/private/yida_workspace/outputs/loop_sft_full_20260911_145617/
loop_adapter_step_0003000.safetensors
```

SFT 使用 BAGEL 原生 rectified-flow target。令 v^*=\epsilon-z_0，depth 1 和 depth 2 的 token-level flow error 分别为 \ell_1 与 \ell_2：

L_{SFT}=\ell_2+\lambda_g\operatorname{ReLU}
\left(\ell_2-\operatorname{stopgrad}(\ell_1)+m\right).

实际运行参数为 `λg=1`、`m=0`、8 卡、每卡 batch 1、gradient accumulation 4、learning rate `1e-5`、3000 optimizer steps。SFT 只更新 generation attention Q/K/V/O LoRA，共 96 个 tensor、2,162,688 个参数。

V2 加载该 generation-only adapter 时，只允许 text K/V LoRA key 缺失。新增 text K/V LoRA 的 B 矩阵以零初始化，A 矩阵按 LoRA 初始化后由 rank 0 广播到全部 rank；随后 current policy 和 frozen reference 从同一份完整 V2 adapter 状态开始。

## 5. Flow-GRPO 训练链路



### 5.1 配对 rollout

每个 rank 每 step 处理一个 GenEval prompt。每个 prompt 生成 4 个 seed；每个 seed 生成一对候选：

```text
same prompt + same initial noise + same SDE noise
    ├── depth 1 frozen BAGEL       -> image_base, z0_base
    └── depth 2 loop policy        -> image_loop, z0_loop, trajectory
```

depth 1 作为同轨迹 control。reward 使用 `loop - base` 的配对差值。

### 5.2 随机转移和可重放概率

BAGEL 原生 Euler ODE 没有动作概率。训练在 `sde_step_indices=[8]` 注入一个随机转移，并保存：

```text
z_t, z_next, timestep, next_timestep,
old_log_prob, sigma_max, noise_level
```

当前 `sde_noise_level=0.8`。令 `dt=t_next-t<0`、flow velocity 为 v_\theta、SDE scale 为 \sigma，代码中的转移为：

\mu_\theta=z_t\left(1+\frac{\sigma^2}{2t}dt\right)
+v_\theta\left(1+\frac{\sigma^2(1-t)}{2t}\right)dt,

z_{next}=\mu_\theta+\sigma\sqrt{-dt}\epsilon.

rollout 在 no-grad 下完成，只保留选中的随机转移。训练阶段对同一个 `z_t -> z_next` 转移重新执行当前 policy，得到 `new_log_prob`。reference policy 使用 RL 启动时冻结的 loop adapter 快照。

### 5.3 Reward

语义 reward 使用 decoded image 上的 GenEval strict score：

\Delta R_{sem}=R_{GenEval}^{loop}-R_{GenEval}^{base}.

质量 reward 使用冻结的 FLUX Diffusion-RM。BAGEL packed clean latent 只做逆 patchify：

```text
[B, H/16 * W/16, 64]
    -> [B, 16, H/8, W/8]
```

该路径不经过 VAE decoder，也不重新编码图片。RM 使用近干净噪声 `u=0.05` 和 prompt 对应的固定噪声，输出：

\Delta R_q=R_{FluxRM}^{loop}-R_{FluxRM}^{base}.

质量只构成单侧惩罚：

P_q=\lambda_q\operatorname{ReLU}(-\tau_q-\Delta R_q),

J=\Delta R_{sem}-P_q.

当前 `quality_tolerance=0`、`quality_penalty_weight=1`。同一 prompt 的 4 个 seed 在组内标准化：

A_i=\frac{J_i-\operatorname{mean}(J)}
{\operatorname{std}(J)+10^{-4}}.

GenEval 和 FLUX RM 都不参与反向传播。

### 5.4 Policy loss

对保存的 SDE 转移计算：

\rho_i=\exp(\log p_\theta-\log p_{old}),

L_{policy}=-\operatorname{mean}\left[
\min(\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)
\right],

L_{RL}=L_{policy}+\beta D_{KL}
(p_\theta\Vert p_{ref}).

reference KL 使用当前 policy 和 frozen reference 的高斯转移均值计算。V2 使用 `clip_range=1e-3`、`β=0.01`，每组 rollout replay 两个 policy epoch。

梯度路径为：

```text
GRPO loss
  -> new_log_prob
  -> transition mean
  -> BAGEL flow velocity
  -> repeated body attention
  -> loop-gated LoRA
```



## 6. V2 固定配置


| 项目                      | 数值                 |
| ----------------------- | ------------------ |
| 图像尺寸                    | 512 × 512          |
| BAGEL flow evaluations  | 29（`num_steps=30`） |
| CFG text scale          | 4.0                |
| CFG interval            | `[0.4, 1.0]`       |
| loop depth              | 2                  |
| loop layers             | `[8,20)`           |
| loop timestep threshold | 0.75               |
| hidden residual cap     | 0.05               |
| SDE step index          | 8                  |
| SDE noise level         | 0.8                |
| group size              | 4                  |
| LoRA rank / alpha       | 8 / 16             |
| trainable LoRA tensors  | 144                |
| trainable parameters    | 2,949,120          |
| learning rate           | `5e-6`             |
| policy epochs           | 2                  |
| clip range              | `1e-3`             |
| KL beta                 | 0.01               |
| max grad norm           | 1.0                |
| max steps               | 1000               |
| world size              | 8                  |




## 7. 实验结果



### 7.1 SFT step 3000


| Step | total loss | loop flow | base flow | loop-base | gain success |
| ---- | ---------- | --------- | --------- | --------- | ------------ |
| 1    | 0.423853   | 0.383624  | 0.348749  | +0.034875 | 0.240        |
| 3000 | 0.398097   | 0.373728  | 0.354890  | +0.018838 | 0.291        |


这里的正 `loop-base` 表示该日志点上 depth 2 flow MSE 高于 depth 1。SFT step 3000 随后作为 V1/V2 RL policy 的初始化权重。

### 7.2 FLUX Diffusion-RM

RM checkpoint：

```text
/private/yida_workspace/outputs/dina_flux_rm/
DiNa-FLUX-HPDv3-19layers_2026.09.12_08.55/checkpoints/epoch_001
```

在 256 个 held-out latent pair 上：


| 近干净噪声 | Pairwise accuracy | 95% CI        | Loss    |
| ----- | ----------------- | ------------- | ------- |
| 0.05  | 95.70%            | 92.47%–97.58% | 0.03547 |
| 0.10  | 95.31%            | 91.99%–97.30% | 0.03627 |




### 7.3 RL 前 loop 质量约束实验

固定 2 个 prompt × 4 个 seed，共 8 个 depth1/depth2 配对：


| Loop 设置                           | 平均 FLUX RM `depth2-depth1` |
| --------------------------------- | -------------------------- |
| 无 hidden residual 限幅的 direct loop | -0.8151                    |
| `t>=0.45`, cap `0.10`             | -0.2943                    |
| `t>=0.75`, cap `0.05`             | -0.0990                    |


`t>=0.75, cap=0.05` 下：

- teapot + two cups：depth 1 counting 为 `3/4`，depth 2 为 `3/4`；
- astronaut + white horse：实体和关系在 depth 1、depth 2 均为 `4/4`；
- 该配置被固定为后续 GRPO rollout 参数。



### 7.4 GRPO V1

V1 使用 generation attention Q/K/V/O LoRA、learning rate `1e-6`、clip `1e-4`、每次 rollout 一个 policy epoch。运行在 step 431 后停止，共记录 3448 个 rank-step 样本。


| 指标                  | 全部 step 平均 | 前 50 step  | 后 50 step |
| ------------------- | ---------- | ---------- | --------- |
| semantic base       | 0.03318    | 0.02760    | 0.02786   |
| semantic loop       | 0.03314    | 0.02417    | 0.02396   |
| semantic delta      | -0.00004   | -0.00344   | -0.00391  |
| quality delta       | -0.06100   | -0.06070   | -0.05961  |
| policy loss         | `5.53e-10` | `9.87e-10` | `1.26e-9` |
| KL                  | `2.81e-6`  | `2.75e-6`  | `2.82e-6` |
| ratio mean          | 1.0        | 1.0        | 1.0       |
| ratio max deviation | 0          | 0          | 0         |
| grad norm           | `1.15e-5`  | `1.13e-5`  | `1.14e-5` |


V1 的日志记录发生在该 rollout 唯一一次 optimizer update 之前，因此 replay ratio 记录为 1。

### 7.5 GRPO V2 运行中快照

快照范围为 step 1–167，共 1336 个 rank-step 样本。V2 相对 V1 增加 text K/V LoRA、将 learning rate 调为 `5e-6`、policy epochs 调为 2、clip 调为 `1e-3`。


| 指标                  | step 1–167 平均 | step 1–50  | 最近 50 step |
| ------------------- | ------------- | ---------- | ---------- |
| semantic base       | 0.03778       | 0.02760    | 0.04141    |
| semantic loop       | 0.03911       | 0.02839    | 0.03906    |
| semantic delta      | +0.00133      | +0.00078   | -0.00234   |
| quality base        | 21.40127      | 21.35789   | 21.39719   |
| quality loop        | 21.34017      | 21.29563   | 21.33461   |
| quality delta       | -0.06110      | -0.06227   | -0.06258   |
| policy loss         | `-6.35e-8`    | `-6.46e-9` | `2.39e-7`  |
| KL                  | `2.85e-6`     | `2.84e-6`  | `2.86e-6`  |
| ratio mean          | 0.99999715    | 0.99999730 | 0.99999703 |
| ratio max deviation | `1.40e-5`     | `1.39e-5`  | `1.45e-5`  |
| grad norm           | `1.17e-5`     | `1.15e-5`  | `1.17e-5`  |


最近 50 step 的 tag 分解：


| Tag        | 样本数 | semantic base | semantic loop | semantic delta | quality delta |
| ---------- | --- | ------------- | ------------- | -------------- | ------------- |
| color_attr | 75  | 0.03250       | 0.04833       | +0.01583       | -0.05667      |
| colors     | 27  | 0.18981       | 0.16667       | -0.02315       | -0.11458      |
| counting   | 104 | 0.00000       | 0.00000       | 0.00000        | -0.03005      |
| position   | 178 | 0.04494       | 0.03792       | -0.00702       | -0.07918      |
| two_object | 16  | 0.06250       | 0.04688       | -0.01563       | -0.02930      |


checkpoint 参数从 step 10 到 step 150 的变化：


| 参数组                            | Tensor 数 | 参数差值 L2 | 相对差值   | 最大绝对差值   |
| ------------------------------ | -------- | ------- | ------ | -------- |
| generation expert Q/K/V/O LoRA | 96       | 0.05475 | 0.460% | 0.000412 |
| shared text K/V LoRA           | 48       | 0.01181 | 0.148% | 0.000293 |


快照时训练进程、8 个 rank 和 GenEval 服务均在运行，未检出 Traceback、NaN、OOM 或 NCCL 错误。

## 8. 验证与代码契约

当前测试覆盖如下：


| 测试范围           | 已验证行为                                                         |
| -------------- | ------------------------------------------------------------- |
| depth 1 基线     | 不启用 loop LoRA；residual scale 为 0 时与 depth 1 精确一致              |
| hidden 合并      | residual 按 token RMS 限幅                                       |
| SDE            | `noise_level=0` 等于 Euler step；log probability 对 velocity 存在梯度 |
| rollout/replay | 第一次更新前 ratio 为 1；第二个 policy epoch 能观察到非 1 ratio               |
| reference      | adapter 临时替换后恢复 current policy                                |
| 参数安全           | trainable 参数不能逃出 LoRA allowlist                               |
| latent reward  | BAGEL packed latent 到 FLUX latent 的几何变换保持一致                   |
| checkpoint     | generation-only SFT adapter 可以加载到扩展后的 text K/V policy         |


V2 改动完成时，本地完整测试为 43 passed；远端 RL 相关测试为 24 passed。

## 9. 结果与实现来源


| 内容                         | 路径                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| 最终 V2 配置                   | `configs/training/loop_grpo_v2.yaml`                                                      |
| RL trainer                 | `scripts/train/bagel_loop_grpo_train.py`                                                  |
| 内部循环和 LoRA                 | `qwen_latent_cot/bagel/loop.py`                                                           |
| BAGEL segmented loop       | `qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py`                                     |
| SDE、advantage、GRPO loss    | `qwen_latent_cot/bagel/flow_grpo.py`                                                      |
| rollout replay 和 reference | `qwen_latent_cot/bagel/loop_grpo.py`                                                      |
| GenEval 与 FLUX RM          | `qwen_latent_cot/bagel/rewards.py`                                                        |
| SFT 原始日志                   | `/private/yida_workspace/outputs/loop_sft_full_20260911_145617/train.log`                 |
| RM held-out 结果             | `/private/yida_workspace/outputs/dina_flux_rm/eval/near_clean_256_seed42.json`            |
| V1 指标                      | `/private/yida_workspace/outputs/bagel_loop_grpo_overnight/metrics_rank*.jsonl`           |
| V2 指标                      | `/private/yida_workspace/outputs/bagel_loop_grpo_v2/metrics_rank*.jsonl`                  |
| V2 checkpoint              | `/private/yida_workspace/outputs/bagel_loop_grpo_v2/loop_grpo_adapter_step_*.safetensors` |
| 固定 8-pair 质量实验             | `artifacts/experiment/residual_loop_t075_c005_20260912/RESULTS.md`                        |




## 10. 维护规则

1. 架构与公式以当前代码行为为准；配置数值以对应运行目录中的 `resolved_config.json` 为准。
2. 运行中的结果必须标注快照时间、最大 step 和 rank-step 样本数。
3. V1、V2 或后续版本分别保留结果表，不把不同参数范围和优化器设置的指标合并。
4. 完整训练结束后新增“完成快照”，保留本节 V2 step 167 快照作为训练过程记录。
5. 数值表必须能够回溯到 JSON、JSONL、日志或 checkpoint；人工观察单独标明样本数。

