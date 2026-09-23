# BAGEL LatentCoT Phase 1 重构方案
## Pair-Grounded Latent Memory Training

> **目标分支基线**：`codex/phase1-structured-reflection`
> **当前分支 HEAD（方案制定时）**：`5b1b637b671844a4b9b79ff09523fb13fcf88f31`
> **Phase 2**：强化学习，继续后置，不与 Phase 1 同时开发。
> **Structured Reflection**：从 Phase 1 主线降级为 ablation / diagnostic，不再作为主要 teacher。

---

# 1. 动机 + 原理

## 1.1 为什么停止以 Structured Reflection 作为主 teacher

当前 pilot 表明：

- no-op：`9/16` teacher 有效；
- edit：`1/16` teacher 有效；
- edit gate 从 `0.001` 放宽到 `0.02`，仍然只有同一条通过；
- 因此问题不是 gate 太严格，而是 reflection 本身几乎没有稳定提供新的有效信息。

当前 reflection 的主要形式是：

```text
instruction
    +
instruction 的结构化复述
    +
通用 Preserve 约束
```

它没有可靠引入模型原先不知道的视觉信息。

因此旧 Phase 1 实际依赖：

\[
[I_0,e]
\rightarrow
[I_0,e,r]
\rightarrow
\Delta v_T
\]

其中 \(r\) 是一个不稳定的中间变量。

即使把 reflection 做成“绝对正确的详细计划”，也会把研究问题改变成：

> 如何构造最好的显式 edit plan / mask / oracle instruction？

这已经是另一条 editing-control 路线，而不是我们想回答的核心问题：

\[
\boxed{
\text{连续 latent memory 能否学会 edit-specific internal state？}
}
\]

所以新的 Phase 1 不再要求“teacher 比 base 更聪明”，而直接利用数据中已经存在的正确视觉终点。

## 1.2 现有数据真正可靠的监督是什么

当前 Phase 1 full 数据：

```text
Edit   : 17,713
No-op  : 4,428
Total  : 22,141
```

所有 17,713 条 edit 都有：

```text
source_image
target_image
instruction
trajectory_step
```

所有样本还有：

```text
target_prompt
source_analysis
trajectory_id
```

因此我们已经天然拥有：

\[
\boxed{
(I_0,e,I_1)
}
\]

其中：

- \(I_0\)：source / negative state；
- \(e\)：edit instruction；
- \(I_1\)：target / positive state。

对 no-op：

\[
I_1 \equiv I_0.
\]

这比 reflection teacher 更可靠，因为“应该变成什么”不是另一个模型推断出来的，而是由真实 target image 直接给出。

Phase 1 的核心监督应从：

```text
text teacher
```

切换到：

\[
\boxed{\text{paired visual supervision}}
\]

## 1.3 Phase 1 重新定义

Phase 1 不再定义为：

> Distill structured reflection into latent memory.

改为：

> **Pair-Grounded Latent Memory Training**

目标拆成两个独立问题。

### Read 问题

给定：

\[
I_0,e,x_t,t
\]

memory 应该形成什么内部状态，才能表示：

\[
I_0 \rightarrow I_1
\]

需要发生的视觉变化？

即：

\[
\boxed{
\text{source + instruction}
\rightarrow
M_{\rm read}
}
\]

需要学习 edit delta。

### Write 问题

当已经有了一个 grounded：

\[
M_{\rm read}
\]

GEN 如何使用它，把当前 flow state：

\[
x_t
\]

推向真实 target：

\[
I_1
\]

？

即：

\[
\boxed{
M_{\rm read}
\rightarrow
GEN
\rightarrow
v_t
}
\]

## 1.4 总体训练分解

新的 Phase 1：

```text
Phase 1.1
Pair-ground Memory Read
        ↓
Memory 知道“该改什么”

Phase 1.2
Target Flow SFT
        ↓
GEN 学会“如何使用 Memory 去改”

Phase 1.3
Joint Relaxation + Memory Necessity + Persist
        ↓
验证 Memory 真的是因果 bottleneck，而不是旁路

Phase 1.4
Curriculum Expansion
        ↓
从简单结构编辑扩展到更复杂编辑
```

最后才进入：

```text
Phase 2
RL / GRPO
```

## 1.5 当前 Read–Write architecture 保持不变

继续固定：

```yaml
K: 8
R: 2
body: [12, 20)
loop_recycle_mode: same_depth
round0_memory_write_enabled: false
```

路径：

\[
F_{0:12}
\rightarrow
Read_{12:20}
\rightarrow
Write_{12:20}
\rightarrow
F_{20:L}.
\]

Read：

\[
GEN/context \rightarrow M
\]

同时：

\[
M \nrightarrow GEN/context.
\]

Read round 后：

- 保留 \(M_{\rm read}\)；
- 丢弃第一遍 GEN body hidden；
- non-memory hidden reset 到相同的 \(h_{\rm base}\)。

Write：

\[
M_{\rm read}\leftrightarrow GEN.
\]

因此不修改 Phase 0.5 已经验证过的核心结构。

## 1.6 新训练目标一：Pair-Grounded Memory Delta

Phase 1.1 不使用 reflection。

对同一个训练 pair：

\[
(I_0,e,I_1)
\]

构造三个 **Read-only** 状态。

### Frozen Source Reference

只给 source image，不给 edit instruction：

\[
M_0^{ref}
=
R_{\theta_0}(x_t,t\mid I_0).
\]

### Frozen Target Oracle

只给 target image：

\[
M_1^{ref}
=
R_{\theta_0}(x_t,t\mid I_1).
\]

Source / Target reference 使用：

- 同一个 frozen BAGEL；
- 同一个 frozen \(m_0\)；
- 同一个 \(x_t,t\)；
- 同样的 strict Read mask；
- LoRA OFF。

定义真实 pair-memory direction：

\[
\boxed{
\Delta M^{pair}
=
M_1^{ref}-M_0^{ref}
}
\]

它表达：

> 在 BAGEL 原生 hidden space 中，从 source visual state 到 target visual state，memory 应该移动的方向。

### Student Read

Student 看：

\[
I_0 + e
\]

但不能看到 \(I_1\)：

\[
M_S
=
R_{\theta_0,\phi_R}(x_t,t\mid I_0,e).
\]

定义：

\[
\Delta M_S
=
M_S-M_0^{ref}.
\]

训练：

\[
\boxed{
\Delta M_S
\rightarrow
\Delta M^{pair}
}
\]

其含义变成：

> instruction 应该使 memory 从“source state”移动到“target state”。

这里完全不需要任何显式 reasoning teacher。

## 1.7 为什么 target reference 不使用 instruction

不建议：

\[
R(I_1,e)
\]

作为默认 target reference。

例如 instruction：

> Add one red cube.

当 \(I_1\) 已经包含新增 red cube 时，再把同一个操作性 instruction 放进 target teacher，会产生歧义：

> 是否还应该再加一个？

因此 Phase 1.1 主方案采用纯视觉 reference：

\[
R(I_0),\qquad R(I_1).
\]

Student 才使用：

\[
R(I_0,e).
\]

这样：

\[
\Delta M^{pair}
\]

尽可能只表达 visual state difference。

`target_prompt` 可在 Phase 1.4 作为 teacher-side semantic enrichment ablation，而不是 Phase 1.1 默认输入。

## 1.8 \(x_t\) 如何构造

对 target image：

\[
x_1=VAE(I_1).
\]

采样：

\[
\epsilon\sim \mathcal N(0,I).
\]

按照 BAGEL 当前 flow convention：

\[
x_t=(1-t)x_1+t\epsilon.
\]

Phase 1.1 的：

\[
M_0^{ref},M_1^{ref},M_S
\]

全部使用同一个：

\[
(x_t,t).
\]

因此三个 memory state 的差异只来自：

```text
source visual context
target visual context
source + instruction student context
```

而不是 query state 不一致。

## 1.9 新训练目标二：真实 Target Flow SFT

Phase 1.2 不需要 Base / Teacher velocity。

BAGEL 当前 training convention：

\[
v^\star
=
\epsilon-x_1.
\]

Student：

\[
v_S
=
v_{\theta_0,\phi}^{loop}
(x_t,t\mid I_0,e).
\]

直接优化：

\[
\boxed{
L_{\rm flow}
=
\|v_S-v^\star\|^2
}
\]

这是 ground-truth flow supervision。

不再使用：

\[
v_B,\quad v_T.
\]

### 为什么 Base velocity 可以删除

旧 residual distillation：

\[
\Delta v_S=v_S-v_B
\]

如果真实 target velocity 是：

\[
v^\star,
\]

那么 target residual：

\[
\Delta v^\star=v^\star-v_B.
\]

两边相减：

\[
\Delta v_S-\Delta v^\star
=
(v_S-v_B)-(v^\star-v_B)
=
v_S-v^\star.
\]

所以 paired supervised training 中：

\[
\boxed{
v_B \text{ 数学上直接抵消}
}
\]

Base forward 没必要。

旧方案每 state：

```text
Base forward
Teacher forward
Student forward
```

新 Phase 1.2：

```text
Student forward only
```

计算开销显著下降。

## 1.10 No-op 不再需要特殊 teacher loss

对 no-op：

\[
I_1=I_0.
\]

直接：

\[
x_1=VAE(I_0).
\]

训练：

\[
v_S
\rightarrow
\epsilon-x_1.
\]

因此 no-op 自动成为：

\[
\boxed{\text{I2I reconstruction / preservation training}}
\]

Edit / No-op 可以统一成同一个 flow objective。

## 1.11 Memory 初始化

当前：

\[
m_0
=
\frac{E_{SOI}+E_{EOI}}{2}
+
10^{-4}\epsilon.
\]

这是 Phase 0/0.5 一个合理的选择：

- 位于 BAGEL native embedding space；
- 初始 perturbation 极小；
- 避免随机大幅破坏 pretrained prior；
- K slots 通过小 deterministic noise 打破完全对称。

Phase 1.1 主方案仍然：

```text
m0 frozen
```

只训练：

```text
UND-Q LoRA
```

这样可以直接回答：

> 固定 memory basis 下，UND-Q 能否学出正确的 pair visual delta？

`m0` learnable 放到 Phase 1.3 ablation，而不是第一轮 grounding。

---

# 2. 参考工作的结论

> 这些工作用于确定训练原则，不直接证明 BAGEL latent loop 一定有效。

## 2.1 Qwen-Image

**Qwen-Image Technical Report**, arXiv:2508.02324。

最相关结论：

1. 编辑 consistency 不只依赖 T2I/TI2I；Qwen-Image 显式加入 **I2I reconstruction**。
2. 原图分别进入 Qwen2.5-VL 和 VAE Encoder，承担 semantic 与 reconstructive/appearance 表征。

对本项目：

\[
\boxed{
\text{no-op 是主训练数据；ViT + VAE source condition 都应保留}
}
\]

## 2.2 Qwen-Image-Edit

官方方法强调：

- semantic editing：允许较大 pixel change，但保持 identity/semantic consistency；
- appearance editing：非目标区域尽可能保持。

对 Phase 1：

> 不应在第一轮把 text/style/identity/复杂 replacement 和 count/relation/move/add-delete 混成均匀训练分布。

## 2.3 BAGEL

**Emerging Properties in Unified Multimodal Pretraining**, arXiv:2505.14683。

BAGEL 本身联合使用：

```text
ViT semantic features
+
VAE visual/generation features
```

官方结果也指出 VAE + ViT 联合视觉特征对 intelligent editing 有明显作用。

因此：

\[
\boxed{
\text{Memory 是 recurrent edit state，不应替代原始 source condition}
}
\]

## 2.4 Mirage

**Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens**, arXiv:2506.17218。

训练范式最值得借鉴：

```text
Stage 1:
用 ground-truth image embeddings 监督 latent token
        ↓
先建立视觉 grounding

Stage 2:
移除强 latent target，主要优化最终 task objective
        ↓
让 latent 为任务服务

随后:
RL
```

对应：

```text
Phase 1.1  memory grounding
Phase 1.2  target flow SFT
Phase 1.3  joint relaxation
Phase 2    RL
```

Mirage 不是 image editing，因此只借训练范式，不直接复用其 loss。

## 2.5 BLIP-2

**BLIP-2**, arXiv:2301.12597。

核心启发：

- frozen image encoder；
- frozen LLM；
- 小型 Querying Transformer；
- 两阶段训练 query bottleneck。

支持：

\[
\boxed{
\text{先冻结 BAGEL 主干，只训练 memory-row UND-Q interface}
}
\]

## 2.6 Learnable Memory / Recurrent Memory Transformer

**Fine-tuning Image Transformers using Learnable Memory**, arXiv:2203.15243。
**Recurrent Memory Transformer**, arXiv:2207.06881。

共同启发：

- 少量 memory token 可以作为任务状态；
- memory 可以通过 attention 被 read/write；
- recurrent memory 可以承载跨段信息。

它们不证明 K=8 或当前 m0 最优，因此 K/m0 保留为 ablation。

## 2.7 Uni-Edit

**Uni-Edit: Intelligent Editing Is A General Task For Unified Model Tuning**, arXiv:2605.21487。

对 BAGEL/Janus-Pro 的结论支持：

> 高质量 paired image editing 本身可以成为统一模型的有效 tuning task。

进一步支持：

\[
\boxed{
(I_0,e,I_1)
}
\]

直接作为主监督，而不是额外制造 text teacher。

## 2.8 汇总原则

1. GT target image 优先于 synthetic text teacher。
2. 先 ground latent memory，再优化最终 generation task。
3. Source ViT + VAE condition 都保留。
4. no-op 是 reconstruction supervision。
5. 先训练小型 Read/Write interface，不动 BAGEL 主干。
6. RL 放在 paired SFT 之后。

---

# 3. Phase 1.1–1.4 代码实现指南

# Phase 1.1 — Pair-Grounded Memory Read

## 3.1.1 研究问题

只回答：

\[
\boxed{
\text{Memory 能否从 }(I_0,e)\text{ 学到 }I_0\rightarrow I_1\text{ 的视觉方向？}
}
\]

暂时不训练 GEN-Q。

## 3.1.2 数据范围

首轮：

```text
count
relation
move
addition
deletion
simple attribute / binding
noop
```

暂时排除：

```text
text
style / cultural identity
complex replacement
high-level semantic transformation
```

## 3.1.3 新增 Read-only API

当前完整 path：

```text
prefix
→ read
→ write
→ suffix
```

新增：

```text
prefix
→ read
→ STOP
```

推荐：

```python
forward_memory_read(
    ...,
    memory_loop_start,
    memory_loop_end,
    memory_body_in=None,
    adapter_mode="read",   # "read" | "off"
)
```

返回：

```python
MemoryReadOutput(
    memory_read=M_read,
)
```

Reference：

```python
adapter_mode="off"
```

Student：

```python
adapter_mode="read"
```

不要通过 `collect_round_diagnostics=True` 拿 `M_read`，因为 diagnostics 会额外跑 suffix。

## 3.1.4 新代码

```text
qwen_latent_cot/bagel/loop_pair_ground.py
scripts/train/bagel_loop_pair_memory.py
configs/training/loop_pair_memory_early.yaml
```

旧：

```text
bagel_loop_delta_v_distill.py
```

保留为 structured-reflection ablation。

## 3.1.5 Context

新增：

```python
build_visual_reference_context(image)
build_student_edit_context(source_image, instruction)
```

Source ref：

```text
source image only
K=8
adapter off
```

Target ref：

```text
target image only
K=8
adapter off
```

Student：

```text
source image + instruction
K=8
adapter read
```

## 3.1.6 Memory target

\[
D_T=\operatorname{sg}(M_1^{ref}-M_0^{ref})
\]

\[
D_S=M_S-\operatorname{sg}(M_0^{ref})
\]

训练：

\[
D_S\rightarrow D_T.
\]

## 3.1.7 Loss

Direction：

\[
L_{\rm mem-dir}
=
\frac1K\sum_k
(1-\cos(D_{S,k},D_{T,k}))
\]

Magnitude：

\[
L_{\rm mem-mag}
=
SmoothL1(RMS(D_S),RMS(D_T))
\]

Regression：

\[
L_{\rm mem-reg}
=
SmoothL1(D_S/\sigma,D_T/\sigma)
\]

No-op：

\[
L_{\rm noop-mem}=RMS(D_S)^2
\]

建议：

```yaml
lambda_mem_dir: 1.0
lambda_mem_mag: 0.1
lambda_mem_reg: 0.25
lambda_noop_mem: 1.0
```

## 3.1.8 Trainable

只训练：

```text
layers[12:20].UND-Q LoRA
```

冻结：

```text
GEN-Q
GEN-O
K/V
FFN
Norm
m0
VAE
ViT
vae2llm
llm2vae
```

建议：

```yaml
lora_rank: 8
lora_alpha: 16
learning_rate: 5e-6
max_grad_norm: 1.0
```

## 3.1.9 Reference cache

先 online smoke。

若跑通，可固定 3 个 timestep anchor + deterministic noise，缓存：

```python
{
  sample_id,
  step_index,
  source_memory_ref,
  delta_memory_target,
}
```

从而正式训练只跑 Student Read。

## 3.1.10 Go condition

必须看到：

\[
\cos(D_S,D_T)\uparrow
\]

\[
\frac{\|D_S-D_T\|}{\|D_T\|+\epsilon}\downarrow
\]

同时：

```text
no-op memory RMS 低
effective rank 不 collapse
```

再做 instruction retrieval：

```text
correct instruction
vs
wrong instruction
```

正确 instruction 的 memory delta 必须更接近 \(D_T\)。

---

# Phase 1.2 — Target Flow SFT / Learn to Write

## 3.2.1 初始化

加载 Phase 1.1 winner。

初始：

```text
UND-Q: loaded + frozen
GEN-Q: zero-init + train
m0: frozen
```

## 3.2.2 Student forward

输入：

```text
source image
instruction
target-noised x_t
```

完整：

\[
F_{0:12}\rightarrow Read\rightarrow Write\rightarrow F_{20:L}.
\]

SFT 第一版只训练 conditional branch：

```text
cfg_text_scale=1
cfg_img_scale=1
```

CFG 留给 validation/inference。

## 3.2.3 Flow target

\[
x_1=VAE(I_1)
\]

\[
x_t=(1-t)x_1+t\epsilon
\]

\[
\boxed{
v^\star=\epsilon-x_1
}
\]

优化：

\[
\boxed{
L_{\rm flow}=MSE(v_S,v^\star)
}
\]

第一版尽量和 BAGEL 原生 flow training 保持一致。

## 3.2.4 Timestep

建议第一轮：

```text
early 50%
mid   30%
late  20%
```

后续再做 uniform 对照。

## 3.2.5 No-op

统一：

```text
target_image = source_image
```

走同一个 \(L_{\rm flow}\)。

batch 中建议：

```text
no-op ≈ 20%
```

## 3.2.6 两段训练

### 1.2A Write warm-up

训练：

```text
GEN-Q only
```

### 1.2B Small joint relaxation

若 1.2A 有正增益：

```text
GEN-Q lr = 5e-6
UND-Q lr = 1e-6
```

可对 25% batch 加 cached memory grounding：

\[
L=L_{\rm flow}+0.05L_{\rm mem}.
\]

## 3.2.7 新代码

```text
scripts/train/bagel_loop_pair_flow_sft.py
configs/training/loop_pair_flow_early_fresh.yaml
```

公共 helper：

```python
prepare_flow_training_state(
    clean_latent,
    timestep,
    noise,
)
```

返回：

```python
x_t
target_velocity
```

避免重新手写与 BAGEL 不一致的 flow convention。

## 3.2.8 Go condition

比较：

```text
Vanilla
Zero-shot early
Phase1.1 only
Phase1.2A
Phase1.2B
```

同时记录：

```text
semantic edit
preservation
ΔM
ΔG
Δv
memory rank
```

---

# Phase 1.3 — Joint Relaxation + Memory Necessity + Persist

## 3.3.1 研究问题

即使 image score 上升，也必须证明：

\[
\boxed{
\text{模型真的依赖 memory 内容}
}
\]

否则只能说明 recurrent-body LoRA 有用。

## 3.3.2 External-Memory Write API

当前 prefix mask 和 round0 write 耦合。

新增：

```python
forward_with_read_memory(
    ...,
    external_memory=M_read,
    prefix_block_memory=True,
    write_body=True,
)
```

真实执行：

```text
prefix [0:12): memory -> non-memory blocked
inject M_read at body entry
write [12:20): memory <-> GEN
suffix [20:L): open
```

用于 correct/shuffled/zero/m0 memory counterfactual。

## 3.3.3 Memory swap

正确：

\[
L_i^+
=
L_{\rm flow}(M_i)
\]

交换：

\[
L_i^-
=
L_{\rm flow}(M_j)
\]

定义：

\[
L_{\rm swap}
=
\max(0,m+L_i^+-L_i^-).
\]

只对约 25% batch 做 swap。

## 3.3.4 Causal controls

评估：

```text
correct M_read
m0
zero
shuffled M
```

预期至少：

\[
L_{\rm correct}<L_{\rm shuffled}.
\]

如果两者无差异，不能声称 memory semantic state 有因果作用。

## 3.3.5 Fresh vs Persist

Fresh：

\[
M_{t_i}^{in}=m_0
\]

Persist：

\[
M_{t_{i+1}}^{in}
=
\operatorname{sg}(M_{t_i}^{out})
\]

继续：

```text
forward persistent
temporal gradient detached
```

不做 full-trajectory BPTT。

Persist 第一版复用：

```text
no-grad rollout
→ capture x_t,t,m_in
→ selected-state target-flow replay
```

## 3.3.6 trajectory_id

在做“跨 edit step memory”前先 audit：

\[
target(I_i)\stackrel{?}{=}source(I_{i+1})
\]

默认 Phase 1.3 只研究：

```text
同一次 diffusion/flow trajectory 内 persist
```

不把 dataset trajectory 和 denoising persistence 混在一起。

## 3.3.7 m0 learnable ablation

Main：

```text
boundary-init m0 frozen
```

Ablation：

```text
boundary-init m0 learnable
m0_lr = 5e-7
```

加：

\[
L_{anchor}=\|m_0-m_0^{init}\|^2.
\]

只有 memory grounding、editing、preservation 同时改善才升级。

## 3.3.8 K ablation

K=8 继续主线。

Memory necessity 成立后再测：

```text
K=4
K=8
```

## 3.3.9 Loss

\[
L=L_{\rm flow}+\lambda_{mem}L_{\rm mem}+\lambda_{swap}L_{\rm swap}
\]

建议：

```yaml
lambda_mem: 0.02 ~ 0.05
lambda_swap: 0.1
```

## 3.3.10 Go condition

1. `correct memory < shuffled memory` 稳定成立；
2. Persist 相比 Fresh 有明确的结构编辑或一致性收益；
3. Phase 1.1 grounding 没完全崩掉。

---

# Phase 1.4 — Curriculum Expansion

## 3.4.1 Curriculum

### Stage A

```text
count
relation
move
addition
deletion
simple attribute binding
```

### Stage B

```text
replacement
material
multi-attribute
hard composition
```

### Stage C

```text
text
style
identity
high-level transformations
```

不要在机制尚未稳定时让 Stage C 主导梯度。

## 3.4.2 Balancing

按以下字段分层采样：

```text
edit_type
difficulty
source_dataset
noop/edit
```

保持：

```text
no-op ≈ 20%
```

避免 replacement 数量过高淹没结构任务。

## 3.4.3 Hard negative memory pair

如果同 source 有多个不同 edit：

```text
I0 + eA -> I1A
I0 + eB -> I1B
```

构造：

```text
M_A
M_B
```

要求：

\[
M_A
\]

对 target A 的 flow loss 低于 \(M_B\)。

这种 hard negative 比随机 batch swap 更强。

## 3.4.4 target_prompt

Phase 1.4 才测试：

```text
R(I1)
vs
R(I1, target_prompt)
```

只把 target_prompt 当 target representation augmentation，不重新升级成主 teacher。

## 3.4.5 参数开放顺序

```text
1. UND-Q memory rows
2. GEN-Q write
3. joint UND-Q + GEN-Q
4. optional learnable m0
5. optional GEN-O
6. optional K/V
```

如果：

\[
\Delta M>0,\quad \Delta G>0,\quad \Delta v\text{ 太小}
\]

才测试 GEN-O。

K/V 最后。

## 3.4.6 最终评估矩阵

```text
B0   Vanilla BAGEL
Z3   Zero-shot early loop
P11  Memory-grounded only
P12  Pair Flow SFT Fresh
P13F Joint Fresh
P13P Joint Persist
```

Causal controls：

```text
correct memory
shuffled memory
m0 memory
zero memory
```

## 3.4.7 T2I regression

最终必须重跑：

```text
GenEval2 hard
GenEval2 full
```

目的不是要求 T2I 大幅提升，而是确认：

\[
\boxed{
\text{editing SFT 没有破坏 BAGEL 原生 generation prior}
}
\]

继续记录：

```text
AM
GM
Pixel MAE vs vanilla
```

---

# References

1. **Qwen-Image Technical Report** — arXiv:2508.02324.
2. **Qwen-Image-Edit** — Qwen Team, 2025.
3. **Emerging Properties in Unified Multimodal Pretraining (BAGEL)** — arXiv:2505.14683.
4. **Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens (Mirage)** — arXiv:2506.17218.
5. **BLIP-2** — arXiv:2301.12597.
6. **Fine-tuning Image Transformers using Learnable Memory** — arXiv:2203.15243.
7. **Recurrent Memory Transformer** — arXiv:2207.06881.
8. **Uni-Edit: Intelligent Editing Is A General Task For Unified Model Tuning** — arXiv:2605.21487.

---

# 最终核心假设

旧假设：

\[
\text{Structured Reflection}
\rightarrow
\Delta v_T
\rightarrow
\text{latent memory}
\]

新的假设：

\[
\boxed{
(I_0,I_1)
\rightarrow
\Delta M^{pair}
}
\]

然后：

\[
\boxed{
(I_0,e)
\rightarrow
M_{\rm read}
\approx
M_0^{ref}+\Delta M^{pair}
}
\]

最后：

\[
\boxed{
M_{\rm read}
\rightarrow
v_S
\rightarrow
I_1
}
\]

Phase 1 真正要证明：

> **BAGEL 的原生 hidden space 中存在一个可以由 source–target visual pair ground、由 instruction 预测、并由 generation expert 消费的连续 recurrent edit state。**

只有当 Phase 1.3 的 memory swap / zero-memory control 仍显示：

\[
L_{\rm correct}<L_{\rm shuffled}
\]

同时 paired editing 提升、T2I prior 基本保持，才进一步支持：

\[
\boxed{
\text{latent memory 本身，而不仅仅是额外计算深度，参与了 semantic editing。}
\]
