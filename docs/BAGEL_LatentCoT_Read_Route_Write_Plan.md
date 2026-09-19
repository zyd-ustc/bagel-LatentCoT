# BAGEL LatentCoT：Read–Route–Write Loop 研究方案

> **版本**：v2 · 2026-09-19  
> **定位**：BAGEL-7B-MoT 上的轻量后训练与推理期 recurrent reasoning 方案  
> **当前状态**：Phase 0 结构已实现；下一步进入 zero-shot mechanism validation  
> **核心目标**：让 understanding path 在生成过程中读取当前生成状态，并以尽量不破坏 BAGEL 原生先验的方式反哺 generation path，实现比显式“生成→解码→反思→再编辑”更紧凑的语义编辑循环。

---

## 1. 动机：为什么需要重新设计 loop

### 1.1 已有实验现象

目前已有三类稳定观察：

1. **Flow Matching 在早期步骤就逐步确定位置、数量和全局拓扑**。中后期再换 prompt，通常只能改变纹理或局部属性。
2. **中途换条件并非数学上非法，真正问题是 trajectory distribution mismatch**：当前状态来自旧条件轨迹，而新速度场主要在另一条件分布附近训练。
3. **BAGEL 上普通条件注入很弱**：尤其是非文字模态、中间 hidden 注入，常表现为轻微扰动或模糊，而不是稳定的语义编辑。

因此，FlowEdit 可以保留为“纯 flow transport”基线，但它没有真正利用 BAGEL 作为 Unified Multimodal Model 的优势。

### 1.2 BAGEL 为什么比普通编辑模型更难直接注入新条件

BAGEL 编辑时，源图和编辑指令在 denoising 前已经进入 context，并形成各层稳定的 `past_key_values`；当前 noisy target `x_t` 才是每个 flow step 的动态 query。源图同时拥有 VAE / ViT 两类原生表示，因此 native image/text context 很强。

可以粗略写成：

\[
C_{native}=[I_{src}^{VAE}, I_{src}^{ViT}, e_{edit}],
\qquad
x_t \rightarrow Q_{gen}(x_t) \rightarrow K/V(C_{native}).
\]

新增 latent memory 若直接作为新 Value 注入，需要与长期训练形成的 native text/image KV 竞争：

- 权重小：`Δv ≈ 0`，只有微小扰动；
- 权重强：容易破坏 feature statistics，先出现 blur / texture drift，而不是结构性语义编辑。

这与专门训练过编辑条件路径的模型存在本质差异。

| 架构 | 源图 / 编辑条件 | 对本项目的启示 |
|---|---|---|
| InstructPix2Pix 类编辑模型 | image latent + instruction 是训练期原生条件 | 强编辑不是靠临时 hidden injection，而是训练过的条件接口 |
| FLUX.1 Kontext | image + text 以统一 context 参与 flow editing | 稳定多轮编辑依赖模型熟悉的 context 组织方式 |
| Qwen-Image-Edit | 输入图像同时进入 VLM 语义路径与 VAE 外观路径 | 语义控制与外观保持最好由原生表示分工完成 |
| BAGEL | source VAE + ViT + text 形成强 KV；target `x_t` 是动态 query | 新 memory 应尽量做“读取/路由”，而不是承担新的视觉内容空间 |
| HBridge | 选择中层进行异构 expert 交互，浅层/深层更专用 | loop 更值得优先放在 mid-layer，而不是只放很靠后的层 |

### 1.3 研究问题

我们不再把目标定义为“生成一个新的视觉表征并注入 GEN”，而是：

> **让 understanding path 读取当前 generation trajectory，形成紧凑的 latent semantic state，并动态改变 generation path 如何使用它本来就熟悉的原生 source/text context。**

这条主线称为 **Read–Route–Write Loop**。

---

## 2. 原理：Read–Route–Write Loop

### 2.1 BAGEL 的 joint attention 是 loop 的基础

在 BAGEL 的 MoT layer 中：

- text / understanding token 使用 UND QKV / FFN；
- VAE generation token 使用 GEN QKV / FFN；
- 两类 token 最终进入同一个 attention interaction space。

因此 attention 中天然包含：

\[
Q_UK_G^\top \quad (GEN\rightarrow UND),
\qquad
Q_GK_U^\top \quad (UND\rightarrow GEN).
\]

我们不需要新增 cross-attention 模块，只需要控制 **什么时候允许读、什么时候允许写**。

### 2.2 same-depth body loop

当前实现保留：

\[
F_{0:s}
\rightarrow
\left[F_{s:e}\right]^R
\rightarrow
F_{e:L}.
\]

默认已经实现 `[20,28)`，下一步重点测试 `[12,20)` 与 `[16,24)`。

每轮 body 开始时：

\[
h_s[\neg M]=h_{base}[\neg M],
\qquad
h_s[M]=m_r.
\]

即 **只 recycle memory，非 memory token 每轮都回到同一个 native body-entry state**。这样可以避免 whole-backbone recurrent drift，也比原先 `final hidden → layer 0` 更接近 BAGEL 原生 hidden distribution。

### 2.3 新主方案：先 Read，再 Write

当前 zero-shot loop 最大的潜在风险是：第一轮中随机/未 grounded 的 `m_0` 已经能被 GEN 读取，从而在 memory 还没“理解”任何东西之前就污染生成。

因此下一版默认采用两阶段交互：

#### Round 0 — Read only

允许：

\[
Q_MK_G^\top,
\]

让 memory 读取当前 GEN state；暂时禁止：

\[
Q_GK_M^\top.
\]

得到：

\[
m_1 = U(m_0, x_t, C_{native}).
\]

此时 generator 尽量保持 vanilla 行为。

#### Round 1+ — Route / Write

从 `m_1` 开始再允许 GEN 读取 memory：

\[
G(x_t; C_{native}, m_1).
\]

训练阶段优先让 memory 改变 **attention routing**，而不是学习新的 Value 内容空间：

- 首选 UND / GEN 的 **Q-only LoRA**；
- K/V、FFN、Norm、VAE、ViT、`vae2llm/llm2vae` 先全部冻结；
- 若 Q-only 太弱，再开放 `O_GEN`；
- K/V LoRA 只作为后续 ablation，不做默认。

核心原则是：

> **learn how to read / route before learning new content representations.**

### 2.4 编辑 context：默认去掉旧生成 prompt

若存在 source image 原始生成 prompt `c_old`，下一步编辑默认不再让它进入 GEN context：

\[
C_{GEN}=[I_{src}^{VAE}, I_{src}^{ViT}, e_{edit}].
\]

原因是 source image 已经提供了语义与外观信息，而旧 prompt 可能与编辑指令发生持续 KV 竞争，例如 `red car → blue car`。

必要时可以做 asymmetric context ablation：

\[
C_{UND}=[I_{src}, c_{old}, e_{edit}],
\qquad
C_{GEN}=[I_{src}, e_{edit}].
\]

### 2.5 MDP 定义保持不变

外层 flow / SDE transition 才是 action，inner loop 只是 recurrent policy computation：

\[
s_i=(x_{t_i},m_i,t_i,C),
\]

\[
(x_{t_i},m_i)
\xrightarrow{R\;\text{rounds}}
(v_i,m_i^{out})
\xrightarrow{SDE/ODE}
x_{t_{i+1}}.
\]

不新增 Gaussian memory action head。

---

## 3. 实现 Phase 划分

### Phase 0 — 当前已完成：结构正确性

当前仓库已经完成：

- `same_depth` body loop；`full_depth` 作为对照；
- CFG 三分支独立 memory；
- `m_in / m_out` trajectory logging；
- slot-collapse diagnostics；
- `persist=True/False`；
- `K=0` 回退 vanilla BAGEL。

**Phase 0 的任务不是追求最终指标，而是证明 loop 机制真实发生。**

需要额外记录：

\[
\Delta M_r=\frac{\|m_{r+1}-m_r\|}{\|m_r\|},
\]

\[
\Delta G_r=\frac{\|h_{gen}^{(r+1)}-h_{gen}^{(r)}\|}{\|h_{gen}^{(r)}\|},
\]

\[
\Delta v_r=\frac{\|v_r-v_1\|}{\|v_1\|}.
\]

目标是验证：

\[
M\;changes \rightarrow GEN\;changes \rightarrow velocity\;changes.
\]

### Phase 0.5 — Zero-shot safety redesign

不训练参数，优先验证新架构是否减少 blur / OOD：

1. **去掉 old prompt**，只保留 source image + edit instruction；
2. **Round 0 read-only**，禁止未 grounded `m_0 → GEN`；
3. zero-shot 默认先用 `persist=False`，隔离 cross-timestep OOD 累积；
4. body depth 扫描：`[12,20)` / `[16,24)` / `[20,28)`；
5. R 先固定 `2`，不要同时扩大 loop depth。

如果 Phase 0.5 相比当前 loop 明显减少模糊，同时保留或提高 instruction adherence，则进入训练。

### Phase 1 — Native Teacher → Latent Loop Distillation

显式文本 reflection 不作为最终推理链，而作为 **teacher**。

Teacher 使用 BAGEL 熟悉的 native text condition：

\[
[I_{src}, e_{edit}] \rightarrow r_{text} \rightarrow v^{teacher}.
\]

Student 使用 latent loop：

\[
[I_{src}, e_{edit}] \rightarrow m_1 \rightarrow m_2 \rightarrow v^{student}.
\]

推荐训练目标不是直接让 memory 拟合某个视觉表征，而是蒸馏 **行为变化**：

\[
\Delta v^{teacher}=v^{teacher}-v^{base},
\qquad
\Delta v^{student}=v^{student}-v^{base},
\]

\[
\mathcal L_{distill}
=
\|\Delta v^{student}-\operatorname{sg}(\Delta v^{teacher})\|^2.
\]

训练参数第一版：

| 模块 | 默认 |
|---|---|
| UND attention Q LoRA | train |
| GEN attention Q LoRA | train |
| GEN attention O LoRA | optional |
| K/V LoRA | freeze |
| UND / GEN FFN | freeze |
| Norm | freeze |
| VAE / ViT / connector | freeze |
| `vae2llm / llm2vae` | freeze |

这一步的目标是把“文字反思有效”蒸馏成“不生成文字也能产生相似 velocity correction”。

### Phase 2 — Flow-GRPO / OPSD

只有 Phase 1 已证明 latent loop 能产生稳定语义 correction 后再进入 RL。

- **Flow-GRPO**：沿用现有 SDE rollout / log-prob；policy 仍是带 recurrent loop 的 BAGEL；reward 同时考虑 edit instruction adherence 与 source preservation。
- **OPSD**：在当前 on-policy `x_t` 上构造 reward-improved clean target，再换算回 velocity target，解决只有终局 reward 时的 credit assignment。
- 不建议第一版每轮 decode 图像做 process reward。

### Phase 3 — Compute recovery

先证明 loop 的质量价值，再做 NFE distillation。最终目标是：

> 用较少 outer denoising steps + 少量 inner recurrent depth，替代显式多轮 decode / ViT / text-reflection / full regeneration。

---

## 4. 实验安排

### 4.1 Phase 0/0.5：zero-shot 主矩阵

所有实验使用相同 source、instruction、seed、initial noise、CFG 和 NFE。

| ID | old prompt | Round-0 write | body | persist | 目的 |
|---|---|---:|---|---:|---|
| Z0 | vanilla | - | - | - | BAGEL editing baseline |
| Z1 | 保留 | 开 | `[20,28)` | on | 当前 loop |
| Z2 | 去掉 | 开 | `[20,28)` | off | 验证 old prompt / persistence 影响 |
| Z3 | 去掉 | **关** | `[20,28)` | off | 验证 read-first 是否减少 blur |
| Z4 | 去掉 | **关** | `[16,24)` | off | mid-layer 主候选 |
| Z5 | 去掉 | **关** | `[12,20)` | off | 更早语义桥接 |
| Z6 | 去掉 | **关** | `[16,24)` | on | 只在 zero-shot 稳定后测试 persistence |
| C0 | 去掉 | - | full-depth | off | repeated-compute control |

任务优先选择“理解容易判断、生成容易犯结构错误”的编辑：

- object add / remove；
- count；
- spatial relation；
- attribute binding；
- color / material replacement；
- identity / layout preservation 下的局部编辑。

### 4.2 Mechanism diagnostics

除了最终图，必须同步记录：

| 类型 | 指标 |
|---|---|
| Memory | `ΔM_r`、pairwise cosine、effective rank、top singular ratio |
| GEN feedback | `ΔG_r`、GEN hidden cosine |
| Flow | `Δv_r`、velocity norm、`v_r ↔ v_R` distance |
| Image quality | blur / artifact、source preservation |
| Edit semantics | instruction adherence、count / relation / attribute accuracy |
| Cost | transformer layer-equivalent FLOPs、wall-clock、peak VRAM |

机制解释规则：

- `ΔM > 0, ΔG ≈ 0`：UND 在变化，但没有真正写回 GEN；
- `ΔM > 0, ΔG > 0, Δv ≈ 0`：feedback 进入 hidden，但被后层洗掉；
- `ΔM/ΔG/Δv` 都明显：核心 loop mechanism 成立；
- `Δv` 大但图像变糊：写入过强或 memory distribution OOD；优先 read-first / Q-only routing，而不是继续放大注入。

### 4.3 Phase 1：训练 ablation

| ID | 训练参数 | Teacher | 目的 |
|---|---|---|---|
| T0 | GEN-Q LoRA | none | 最保守生成侧适配 |
| T1 | UND-Q + GEN-Q | none | 验证双向 read/write 适配 |
| T2 | UND-Q + GEN-Q | text-reflection `Δv` | **主模型** |
| T3 | T2 + GEN-O | text-reflection `Δv` | Q-only 太弱时增强输出 |
| T4 | T2 + K/V LoRA | text-reflection `Δv` | 只作为容量上界，不作为默认 |
| T5 | T2 + Flow-GRPO | final reward | RL 增益 |
| T6 | T5 + OPSD | on-policy teacher | 中间 credit assignment |

### 4.4 Equal-compute 对照

必须至少包含：

1. inner loop vs 增加 outer denoise steps；
2. UMM memory loop vs gen-only repeated body compute；
3. latent loop vs 显式 `decode → ViT → text reflection → edit`；
4. same-depth vs full-depth recurrence；
5. `[12,20)` / `[16,24)` / `[20,28)` layer placement。

---

## 5. 结果记录模板

### 5.1 单次实验记录

| Field | Value |
|---|---|
| Experiment ID |  |
| Commit |  |
| Source / instruction |  |
| Seed / initial noise |  |
| NFE / CFG |  |
| Body layers |  |
| `K / R` |  |
| old prompt | on / off |
| Round-0 write | on / off |
| persist | on / off |
| Trainable params |  |
| Instruction score |  |
| Source preservation |  |
| Blur / artifact |  |
| `ΔM / ΔG / Δv` |  |
| Memory effective rank |  |
| Runtime / VRAM |  |
| Observation |  |
| Decision | keep / reject / rerun |

### 5.2 阶段结论表

| Question | Evidence | Conclusion |
|---|---|---|
| UND 是否读到当前 GEN？ | `ΔM` + perturb `x_t` mechanism test |  |
| Memory 是否真正反馈 GEN？ | `ΔG` / mask `GEN←UND` |  |
| Feedback 是否改变 flow？ | `Δv` |  |
| read-first 是否减少 blur？ | Z2 vs Z3 |  |
| mid-layer 是否更适合？ | Z3/Z4/Z5 |  |
| old prompt 是否阻碍编辑？ | Z1 vs Z2 |  |
| persistence 是否有益？ | Z4 vs Z6 |  |
| latent loop 是否优于 gen-only repeated compute？ | equal-compute |  |
| text teacher 是否可被 latent loop 蒸馏？ | T1 vs T2 |  |

---

## 6. 成功标准与止损条件

### Phase 0 Go

不要求 zero-shot 就显著超越 BAGEL，但至少应出现：

\[
M\;changes \rightarrow GEN\;changes \rightarrow velocity\;changes,
\]

并且在一部分结构性编辑任务上出现可重复的正例。

### Phase 1 Go

`UND-Q + GEN-Q + text-teacher distillation` 在 equal-compute 下：

- instruction adherence 稳定高于 current loop / gen-only repeated compute；
- source preservation 不明显下降；
- blur / artifact 不增加；
- 增益来自 loop，而不是单纯更多 FLOPs。

### Stop / Pivot

若满足以下任一情况，应停止继续堆 GRPO / OPSD：

- memory 明显变化，但 `ΔG` 长期接近 0；
- Q-only / Q+O 小规模训练后仍无法让 loop 超过 gen-only repeated compute；
- 语义增益始终只能通过大幅破坏 source preservation 获得；
- equal-compute 下显式 text reflection 始终显著领先，latent loop 无法蒸馏其行为。

此时应重新检查 attention routing / context organization，而不是继续扩大 LoRA、K 或 R。

---

## 7. 当前推荐默认配置

```yaml
loop_recycle_mode: same_depth
memory_loop_start_layer: 16   # 与 12 / 20 做 ablation
memory_loop_end_layer: 24
memory_loop_repeat: 2
num_loop_tokens: 8
loop_memory_persist: false    # zero-shot first
remove_old_prompt: true
round0_gen_reads_memory: false
cfg_branch_memory: independent
```

训练进入 Phase 1 后：

```yaml
trainable:
  und_attention_q_lora: true
  gen_attention_q_lora: true
  gen_attention_o_lora: false   # optional if underfit
  k_v_lora: false
  ffn: false
  norm: false
  vae_vit_connectors: false
teacher:
  type: explicit_text_reflection
  target: delta_velocity
```

---

## 8. 一句话论文主线

> **BAGEL 的优势不应只是“一个模型同时理解和生成”，而应允许理解状态在生成过程中持续读取当前 trajectory，并动态重路由原生图像/文本条件。Read–Route–Write Loop 将这种能力限制在局部 same-depth recurrent body 内，以最小参数和最小 prior shift，把显式文本反思压缩成连续 latent reasoning。**

---

## 9. 参考

1. ByteDance-Seed, **BAGEL: Emerging Properties in Unified Multimodal Pretraining**, 2025. https://github.com/ByteDance-Seed/Bagel
2. SenseTime, **Looped MMDiT：从更大参数到更深计算**, 2026. https://www.sensetime.com/cn/news/looped-mm-di-t-scaling
3. Black Forest Labs, **FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space**, 2025. https://arxiv.org/abs/2506.15742
4. Qwen Team, **Qwen-Image-Edit**, 2025. https://qwenlm.github.io/blog/qwen-image-edit/
5. Wang et al., **HBridge: H-Shape Bridging of Heterogeneous Experts for Unified Multimodal Understanding and Generation**, CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Wang_HBridge_H-Shape_Bridging_of_Heterogeneous_Experts_for_Unified_Multimodal_Understanding_CVPR_2026_paper.html
6. Kulikov et al., **FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models**, ICCV 2025. https://arxiv.org/abs/2412.08629
7. **Flow-GRPO / BAGEL Flow-GRPO integration** and **On-Policy Self-Distillation** are retained as Phase 2 training references rather than Phase 0 architecture dependencies.
