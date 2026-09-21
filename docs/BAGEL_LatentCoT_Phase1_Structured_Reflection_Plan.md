# BAGEL LatentCoT Phase 1：Structured Text Reflection → Latent Read–Write Distillation

> **代码基线**：`f1ac543363e65c99cbe8ff0038e0bc4d4ef1002d`
> **Phase 0.5 结论**：training-free Read–Write loop 已证明机制可用，并基本保留 BAGEL 原生图像生成能力；不同 body/persist 配置在 GenEval2 上没有形成足够稳定的统计优势。训练阶段不再继续追 zero-shot 微小波动，而转向学习“正确的结构语义 correction”。
> **Phase 1 主线**：使用 **Structured Text Reflection** 作为 native-text teacher，将显式结构化语义计划产生的 velocity correction 蒸馏到 latent Read–Write loop。
> **Phase 2**：强化学习（Flow-GRPO / reward optimization），仅在 Phase 1 得到稳定 SFT adapter 后启动。

---

## 0. Phase 1 总目标

Phase 1 不训练 BAGEL 重新学会生成图像，而是训练一个局部 recurrent controller：

$$
\text{Frozen BAGEL}
\rightarrow
\boxed{\text{Trainable Read–Write Loop}}
\rightarrow
\text{Frozen BAGEL}.
$$

Teacher 使用 BAGEL 熟悉的原生 text condition：

$$
C_B=[I_{\rm src},e],
$$

$$
C_T=[I_{\rm src},e,r_{\rm struct}],
$$

其中：

- $I_{\rm src}$：source image；
- $e$：edit instruction；
- $r_{\rm struct}$：Structured Text Reflection；
- Base / Teacher **不开 latent loop**；
- Student 只看到 $[I_{\rm src},e]$，但开启 latent Read–Write loop。

定义：

$$
v_B=v_\theta(x_t,t,C_B),
$$

$$
v_T=v_\theta(x_t,t,C_T),
$$

$$
v_S=v_{\theta,\phi}^{loop}(x_t,t,C_B),
$$

其中 $\theta$ 为 frozen BAGEL，$\phi$ 为 loop LoRA。

核心训练目标不是拟合完整 teacher velocity，而是拟合 teacher 相对原始 BAGEL 的 correction：

$$
\Delta v_T=v_T-v_B,
$$

$$
\Delta v_S=v_S-v_B.
$$

最终目标：

$$
\boxed{
\Delta v_S\approx\Delta v_T
}
$$

即：

> **让 latent memory 学会复现 Structured Text Reflection 给 BAGEL 带来的结构语义修正，同时尽量不重学 BAGEL 已有的纹理、风格和成像先验。**

---

# 1. 现有代码的 Bug 与训练阻塞项

## 1.1 P0：当前 GRPO 中的 `base` 实际不是 vanilla BAGEL

当前：

`scripts/train/bagel_loop_grpo_train.py`

加载模型时：

```python
num_loop_tokens = 8
```

随后 Base 和 Loop 都调用：

```python
inferencer.gen_image(...)
```

而 `InterleaveInferencer.gen_image()` 内部：

```python
generation_input = self.model.prepare_vae_latent(...)
```

没有 per-call `num_loop_tokens` override，因此会直接读取：

```python
model.config.num_loop_tokens
```

也就是 K=8。

当前代码中：

```python
base_image, base_latent = inferencer.gen_image(
    ...,
    return_latent=True,
)

rollout = inferencer.gen_image(
    ...,
    return_trajectory=True,
)
```

`return_trajectory=False/True` 只决定是否记录 trajectory，**不决定 loop 是否开启**。

因此当前 GRPO reward 中：

$$
R_{loop}-R_{base}
$$

并不是严格意义上的：

$$
R_{K=8}-R_{K=0}.
$$

### 修复要求

`gen_image()` 必须增加显式参数：

```python
num_loop_tokens: Optional[int] = None
```

并传入：

```python
prepare_vae_latent(..., num_loop_tokens=num_loop_tokens)
prepare_vae_latent_cfg(..., num_loop_tokens=num_loop_tokens)
```

调用时：

```python
# Vanilla / Base / Teacher
num_loop_tokens=0

# Student / Loop
num_loop_tokens=8
```

**禁止通过临时修改 `model.config.num_loop_tokens` 实现三路切换。**

原因：

- 容易污染 CFG branch；
- 分布式训练中存在共享状态风险；
- replay context 难以保证与 rollout 完全一致；
- Base / Teacher / Student 的实验合同不够显式。

---

## 1.2 P0：Phase 1 不能直接调用 `gen_image()` 做 student backward

当前：

```python
@torch.no_grad()
def gen_image(...):
```

所以 Phase 1 不能直接：

```python
loss = distill(gen_image(...))
loss.backward()
```

### 正确方案

训练采用：

```text
no_grad rollout
      ↓
保存 selected states
(x_t, t, m_in)
      ↓
Base velocity replay      no_grad
Teacher velocity replay   no_grad
Student loop replay       grad
      ↓
Δv distillation loss
```

即复用当前 `loop_grpo.py` 的“rollout → exact replay”思想，但 SFT 不需要 SDE log-prob / GRPO ratio。

新增：

```text
qwen_latent_cot/bagel/loop_distill.py
scripts/train/bagel_loop_delta_v_distill.py
```

---

## 1.3 P0：Phase 1 需要显式区分三套 generation input

三条路径 query sequence 不同：

### Base

```text
Past KV: [source image, instruction]
Current: [SOI | X_t | EOI]
K = 0
```

### Teacher

```text
Past KV: [source image, instruction, structured reflection]
Current: [SOI | X_t | EOI]
K = 0
```

### Student

```text
Past KV: [source image, instruction]
Current: [SOI | M×8 | X_t | EOI]
K = 8
```

因此不能复用同一个 packed query tensor，只替换 KV。

建议新增统一 helper：

```python
prepare_velocity_bundle(
    context,
    image_shape,
    *,
    num_loop_tokens,
)
```

输出：

```python
{
    "flow_input": ...,
    "cfg_text_input": ...,
    "cfg_img_input": ...,
    "past_key_values": ...,
}
```

Base/Teacher `num_loop_tokens=0`，Student `num_loop_tokens=8`。

---

## 1.4 P1：GRPO adapter schema 尚未支持新的 Phase 1 adapter

当前 `bagel_loop_grpo_train.py::_validate_adapter_contract()` 只接受旧 v6/v7 schema，例如：

```text
bagel_native_loop_format_adapter_v6
bagel_semantic_state_flow_adapter_v7
bagel_semantic_state_grpo_adapter_v7
```

Phase 1 新 adapter 建议定义：

```json
{
  "schema": "bagel_loop_delta_velocity_adapter_v8",
  "objective": "structured_reflection_delta_velocity_distillation"
}
```

Phase 2 开始前再让 GRPO contract 接受 v8。

Phase 1 本身不要继续沿用旧：

```text
loop_step_0000100.safetensors
```

作为主 initializer。

推荐从 **zero-init LoRA residual** 开始：

- LoRA A：正常初始化；
- LoRA B：0 初始化；
- 初始 student 等价于 Phase 0.5 frozen loop。

---

## 1.5 P1：现有 GRPO 配置与 Phase 1 主方向不一致

当前：

```yaml
memory_loop_start_layer: 16
memory_loop_end_layer: 24
loop_memory_persist: false
```

Phase 1 决定改为：

```yaml
memory_loop_start_layer: 12
memory_loop_end_layer: 20
```

并行训练：

```text
Fresh   persist=False
Persist persist=True
```

因此不可直接复用 `configs/training/loop_grpo.yaml` 作为 SFT 配置。

---

## 1.6 P1：当前训练入口仅支持 CUDA

当前 GRPO trainer 包含：

```python
if not torch.cuda.is_available():
    raise RuntimeError(...)
```

并使用：

```text
device = cuda
backend = nccl
torch.autocast("cuda")
```

如果正式 Phase 1/2 继续部署在 Ascend NPU，该脚本当前是硬阻塞。

建议训练公共层抽象：

```python
accelerator.resolve_device()
accelerator.autocast_for()
accelerator.manual_seed_all()
```

不要在新 SFT trainer 中复制 CUDA-only 写法。

---

## 1.7 非 Bug，但必须保持的现有设计

以下行为当前是正确设计，Phase 1 不修改：

- `loop_memory` / $m_0$ frozen；
- Round 0 strict read；
- non-memory query 在 Read 阶段不能读取 memory；
- Read round 后仅 recycle memory，GEN 恢复到同一个 `h_base`；
- LoRA 仅在 recurrent body `[s,e)` 激活；
- Prefix/Suffix 保持 frozen BAGEL；
- UND-Q LoRA 仅作用于 memory rows；
- GEN-Q LoRA 仅在 Write round 激活；
- K/V、FFN、Norm 默认不训练；
- persist rollout 中跨 timestep 的 `m_in` 使用 detached state，不保留完整外层 trajectory autograd graph。

---

# 2. 新增 SFT 训练逻辑

## 2.1 Phase 1 默认结构

第一版固定：

```yaml
num_loop_tokens: 8
loop_depth: 2
round0_memory_write_enabled: false
loop_recycle_mode: same_depth

memory_loop_start_layer: 12
memory_loop_end_layer: 20

trainable:
  und_q_lora: true
  gen_q_lora: true
  gen_o_lora: false
  k_v_lora: false
  ffn: false
  norm: false
  loop_memory: false
```

也就是：

$$
F_{0:12}
\rightarrow
Read_{12:20}
\rightarrow
Write_{12:20}
\rightarrow
F_{20:L}.
$$

---

## 2.2 Trainable 参数分工

### Read round

只开启 memory-row UND-Q LoRA：

$$
Q_M
=
(W_Q^{UND}+\Delta W_Q^{UND})M.
$$

目的：

$$
\boxed{
\text{learn what memory should read}
}
$$

即学习从：

- source condition；
- edit instruction；
- 当前 $x_t$ / GEN hidden；

中提取真正与编辑有关的信息。

### Write round

开启：

$$
Q_M
=
(W_Q^{UND}+\Delta W_Q^{UND})M
$$

以及：

$$
Q_G
=
(W_Q^{GEN}+\Delta W_Q^{GEN})G.
$$

目的：

$$
\boxed{
\text{learn how GEN should use grounded memory}
}
$$

第一版不修改：

$$
K_M,V_M,K_G,V_G.
$$

因此 memory 仍处于 BAGEL 原生 feature/value space。

---

## 2.3 Structured Text Reflection Teacher

Teacher reflection 不是自由 CoT。

禁止：

```text
First, I should carefully inspect the image...
Then I should think...
```

采用短、结构化、可验证的 final plan。

推荐格式：

```text
EDIT PLAN
Target changes:
- Count: red cube -> 3.
- Relation: red cubes should be left of the blue sphere.

Preserve:
- Keep the blue sphere unchanged.
- Preserve object colors, materials, background, camera, lighting and texture.
```

或者单一属性：

```text
EDIT PLAN
Target changes:
- Replace the car color from red to blue.

Preserve:
- Preserve car identity, shape, position, background, lighting and texture.
```

reflection 必须强调两部分：

$$
\boxed{\text{Target change}}
$$

和：

$$
\boxed{\text{Preserve constraints}}
$$

这比自由文本 reflection 更适合蒸馏“语义 correction 而不是重新生成”。

---

## 2.4 三路 velocity

对完全相同的：

$$
(x_t,t)
$$

计算：

### Base

$$
v_B
=
v_\theta(x_t,t;I_{\rm src},e)
$$

K=0，no-grad。

### Teacher

$$
v_T
=
v_\theta(x_t,t;I_{\rm src},e,r_{\rm struct})
$$

K=0，no-grad。

### Student

$$
v_S
=
v_{\theta,\phi}^{loop}(x_t,t;I_{\rm src},e)
$$

K=8，grad。

定义：

$$
\Delta v_T=v_T-v_B,
$$

$$
\Delta v_S=v_S-v_B.
$$

---

## 2.5 主 Loss

### 2.5.1 Delta velocity regression

推荐使用带 floor 的 normalized Smooth-L1，而不是裸 MSE：

$$
s_T^2
=
\operatorname{mean}(\Delta v_T^2),
$$

$$
d
=
\max(\operatorname{sg}(s_T^2),\tau).
$$

$$
\boxed{
L_{\Delta v}
=
\operatorname{SmoothL1}
\left(
\frac{\Delta v_S}{\sqrt d},
\frac{\operatorname{sg}(\Delta v_T)}{\sqrt d}
\right)
}
$$

这样避免：

- teacher correction 很小时 normalization 爆炸；
- 大 velocity token 完全主导梯度；
- student 被迫拟合完整 BAGEL velocity。

---

### 2.5.2 Direction loss

对非 trivial teacher correction：

$$
L_{\rm dir}
=
1-\cos(
\Delta v_S,
\operatorname{sg}(\Delta v_T)
).
$$

只在：

$$
\|\Delta v_T\|>\tau_{\rm active}
$$

时启用。

目标：

> correction 的语义方向优先正确。

---

### 2.5.3 Overshoot safety

防止 student correction 明显大于 teacher：

$$
L_{\rm over}
=
\left[
\max
\left(
0,
\operatorname{RMS}(\Delta v_S)
-
\gamma\operatorname{RMS}(\Delta v_T)
\right)
\right]^2.
$$

建议：

$$
\gamma=1.5.
$$

---

### 2.5.4 No-op restraint

数据中显式加入 no-op / already-satisfied case。

对于这些样本：

$$
\Delta v_T\approx0.
$$

单独优化：

$$
\boxed{
L_{\rm noop}
=
\operatorname{RMS}(\Delta v_S)^2
}
$$

直接训练：

> 不需要改的时候不要动 BAGEL。

---

### 2.5.5 Phase 1.1 总 Loss

建议第一版：

$$
\boxed{
L
=
L_{\Delta v}
+
\lambda_{\rm dir}L_{\rm dir}
+
\lambda_{\rm over}L_{\rm over}
+
\lambda_{\rm noop}L_{\rm noop}
}
$$

初始化建议：

```yaml
lambda_delta_v: 1.0
lambda_dir: 0.1
lambda_over: 0.05
lambda_noop: 1.0
overshoot_gamma: 1.5
```

不要第一版加入：

- memory hidden matching；
- target CLIP loss；
- image perceptual loss；
- K/V auxiliary loss；
- per-round deep supervision。

先验证单一行为蒸馏链：

$$
\text{reflection}
\rightarrow
\Delta v_T
\rightarrow
\Delta v_S.
$$

---

## 2.6 Optimizer / LoRA 默认安排

建议 Phase 1 初始配置：

```yaml
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0

optimizer: AdamW
learning_rate: 5.0e-6
betas: [0.9, 0.95]
weight_decay: 0.0
max_grad_norm: 1.0

precision:
  base_model: bf16
  lora_master: fp32

warmup_ratio: 0.03
scheduler: cosine
```

学习率建议至少做：

```text
3e-6 / 5e-6 / 1e-5
```

小规模 smoke，默认先 5e-6。

---

## 2.7 Rollout → Replay 训练

不保存完整 30–50 step autograd graph。

### Stage A：rollout

`torch.no_grad()` 下：

```text
source + instruction
        ↓
base/student-compatible editing trajectory
        ↓
selected timestep states
```

记录：

```python
{
    "x_t": ...,
    "t": ...,
    "m_in": ...,          # persist branch only
    "sample_id": ...,
}
```

### Stage B：replay

每个 state 独立：

```text
Base replay     K=0, no-grad
Teacher replay  K=0, no-grad
Student replay  K=8, grad
```

只 Student 构图。

---

## 2.8 timestep sampling

Phase 1 目标优先是：

- count；
- position；
- relation；
- binding；
- addition/deletion。

因此训练 state 不均匀采样。

建议第一版：

```text
60% : high-noise / early flow
30% : middle
10% : late
```

具体使用 BAGEL 实际 shifted schedule 的 **step index bucket**，不要直接假设线性 t。

例如 30-step rollout：

```text
early : step 0–11
mid   : step 12–21
late  : step 22–28
```

每个 sample 选 2–4 个 state replay。

---

## 2.9 Fresh / Persist 双路

### Fresh

```yaml
loop_memory_persist: false
```

每个 timestep 从 frozen $m_0$ 开始：

$$
m_0
\rightarrow
m_t^{read}
\rightarrow
m_t^{write}.
$$

代表：

$$
\boxed{\text{per-step latent reflection}}
$$

### Persist

```yaml
loop_memory_persist: true
```

上一 timestep：

$$
m_{t_i}^{out}
$$

作为下一 timestep：

$$
m_{t_{i+1}}^{in}.
$$

但训练 replay 时：

$$
m_{t_i}^{out}
\overset{detach}{\longrightarrow}
m_{t_{i+1}}^{in}.
$$

Phase 1 不做跨整个 denoising trajectory 的 BPTT。

Fresh / Persist：

- 同数据；
- 同 batch 顺序；
- 同 teacher；
- 同 seed；
- 同 LoRA initialization；
- 同 optimizer；
- 只改变 persistence。

---

# 3. 数据要求与构造思路

## 3.1 最小数据 schema

推荐 JSONL：

```json
{
  "id": "edit_000001",
  "source_image": "images/edit_000001_source.png",
  "instruction": "Add one red cube to the right of the blue sphere.",
  "reflection": "EDIT PLAN\nTarget changes:\n- Count: red cube -> 3.\n- Relation: the added red cube should be right of the blue sphere.\n\nPreserve:\n- Keep the existing objects unchanged.\n- Preserve colors, materials, background, camera, lighting and texture.",
  "edit_type": ["addition", "count", "relation"],
  "target_constraints": [
    "red_cube_count=3",
    "exists(red_cube,right_of,blue_sphere)"
  ],
  "preserve_constraints": [
    "blue_sphere_identity",
    "existing_object_color",
    "material",
    "background",
    "camera",
    "lighting",
    "texture"
  ],
  "is_noop": false,
  "difficulty": 3
}
```

可选：

```json
{
  "target_image": "...",
  "source_prompt": "...",
  "target_prompt": "...",
  "scene_graph_before": {},
  "scene_graph_after": {},
  "reflection_source": "template|llm|human",
  "verified": true
}
```

Phase 1.1 **不要求 target image**。

---

## 3.2 数据任务优先级

| 类别 | 优先级 | 典型任务 |
|---|---:|---|
| Count | P0 | 2→3 objects、remove one |
| Spatial relation | P0 | left/right/above/below/front/behind |
| Attribute binding | P0 | red cube vs blue sphere 属性归属 |
| Addition / deletion | P0 | add/remove object |
| Replacement | P1 | cube→sphere、car→bike |
| Color / material | P1 | red→blue、metal→wood |
| Identity-preserving move | P1 | move object without changing appearance |
| Style / aesthetics | 暂缓 | 不作为 Phase 1 主任务 |

---

## 3.3 数据构造主线：scene graph → source → edit → reflection

最推荐的数据构造方式不是让 LLM 随机写 instruction，而是先构造可验证的 scene graph。

### Before graph

```text
objects:
- red_cube_1
- red_cube_2
- blue_sphere_1

relations:
- red_cube_1 left_of blue_sphere_1
```

### Edit operator

```text
ADD red_cube_3 right_of blue_sphere_1
```

### After graph

自动得到：

```text
red_cube_count = 3
red_cube_3 right_of blue_sphere_1
```

然后模板化生成：

### Instruction

```text
Add one red cube to the right of the blue sphere.
```

### Structured reflection

```text
EDIT PLAN
Target changes:
- Increase the red cube count from 2 to 3.
- Place the added red cube to the right of the blue sphere.

Preserve:
- Keep the two existing red cubes and the blue sphere unchanged.
- Preserve colors, materials, background, camera, lighting and texture.
```

这样 reflection 是由 ground-truth edit graph 得到，而不是依赖 LLM 自由推理。

---

## 3.4 Source image 构造

Phase 1 优先两级数据。

### Tier A：可控合成 / BAGEL 自生成 source

优点：

- count / relation ground truth 准确；
- scene graph 可验证；
- 易生成大规模 hard structural pairs；
- 与 BAGEL 自身视觉 distribution 更接近。

流程：

```text
scene graph
  ↓
source caption
  ↓
BAGEL / controlled renderer
  ↓
VLM verify source
  ↓
edit operator
  ↓
instruction + reflection
```

### Tier B：真实图像语义编辑数据

后续加入：

- 单物体属性修改；
- 多物体关系；
- object addition/deletion；
- preservation-heavy edits。

真实图像必须先过 source scene verification，避免 instruction 与 source 不一致。

---

## 3.5 No-op / restraint 数据

建议占：

$$
15\%-25\%.
$$

两类：

### Already satisfied

Source 已经满足 instruction。

例如：

```text
source already has 3 red cubes
instruction: Make the image contain 3 red cubes.
```

Reflection：

```text
EDIT PLAN
Target changes:
- No structural change is required.

Preserve:
- Preserve the entire image.
```

### Preservation-only

```text
instruction:
Keep the scene unchanged.
```

teacher correction 应接近：

$$
\Delta v_T\approx0.
$$

这类数据专门保护 Phase 0.5 已经表现出的低纹理破坏特性。

---

## 3.6 Structured Reflection 质量规则

每条 reflection 必须满足：

1. 不包含长 CoT；
2. 不描述无法从数据验证的推理；
3. 必须有 `Target changes`；
4. 必须有 `Preserve`；
5. 每条 target change 可映射到结构化 constraint；
6. 不复制 source prompt 中与编辑无关的大量描述；
7. 不要求改变的属性必须尽可能进入 preservation；
8. target 与 instruction 冲突则整条样本丢弃。

推荐限制：

```text
reflection <= 120 English tokens
target bullets <= 4
preserve bullets <= 4
```

---

## 3.7 数据划分

必须按 scene / source image 分组切分，避免同一 source 的多个 edit 泄漏到 val/test。

建议：

```text
Train : 90%
Val   : 5%
Test  : 5%
```

额外保留：

```text
GenEval2 hard T2I
Phase 0.5 full 800
paired semantic editing holdout
```

作为完全不参与训练的数据。

---

# 4. 实验方案

# Phase 1.1 — Fresh Early Loop：证明 Structured Reflection 可蒸馏

## 目标

回答：

$$
\boxed{
\text{显式 structured reflection 的 correction 能否被 latent loop 学到？}
}
$$

配置：

```yaml
body: [12,20)
K: 8
R: 2
persist: false
strict_read: true

train:
  UND-Q memory-row LoRA: true
  GEN-Q write LoRA: true
  GEN-O: false
  K/V: false
```

对照：

```text
B0  Vanilla BAGEL editing
Z3  Zero-shot early loop
T1  Phase1.1 fresh Δv-distill
TT  Explicit Structured Reflection Teacher
```

主要指标：

### Distillation mechanism

$$
\cos(\Delta v_S,\Delta v_T)
$$

$$
E_{\Delta v}
=
\frac{
\|\Delta v_S-\Delta v_T\|
}{
\|\Delta v_T\|+\epsilon
}
$$

### Editing semantics

- count accuracy；
- relation accuracy；
- binding；
- add/delete success。

### Preservation

- non-target object preservation；
- background；
- texture；
- identity；
- LPIPS / VLM preservation judge（仅作为辅助）。

### Go 条件

至少同时看到：

1. $\cos(\Delta v_S,\Delta v_T)$ 随训练明显上升；
2. $E_{\Delta v}$ 下降；
3. T1 > zero-shot early loop；
4. T1 图像质量不明显低于 vanilla；
5. no-op set 上 Pixel MAE 保持低。

---

# Phase 1.2 — Fresh vs Persist 双路训练

## 目标

回答：

$$
\boxed{
\text{跨 timestep latent state 是否比独立 per-step reflection 更有价值？}
}
$$

两条完全匹配：

### T1-F

```text
body [12,20)
persist=False
```

### T1-P

```text
body [12,20)
persist=True
```

其它所有变量一致。

Persist 使用：

```text
forward persist
+
detached temporal m_in replay
```

不做 full BPTT。

重点分析：

- early steps memory rank；
- memory cosine across timestep；
- $\Delta v$ correction consistency；
- count / relation；
- texture drift；
- long-horizon accumulation。

### 结论规则

如果 Persist：

- hard semantics 提升；
- preservation 不降；
- memory 不 collapse；

则保留。

否则 Phase 1 主线回到 Fresh，Persist 仅作为 ablation。

---

# Phase 1.3 — Structured Reflection / 数据配方扩展

## 目标

不是扩模型，而是确认 teacher/data formulation。

做以下 matched ablation：

### Reflection A：Target-only

```text
Target changes only
```

### Reflection B：Target + Preserve

```text
Target changes
+
Preserve constraints
```

### Reflection C：Free-form reflection

只作为负面对照，不作为主 teacher。

预期主方案：

$$
\boxed{
\text{Target + Preserve}
}
$$

如果 B 明显降低纹理/背景漂移，则固定为最终 teacher format。

同时测试：

```text
no-op ratio:
0% / 10% / 20%
```

主候选 20%。

---

# Phase 1.4 — Capacity Escalation，仅在 Q-only underfit 时启动

只有满足：

$$
\Delta M>0,\quad
\Delta G>0,
$$

但：

$$
\|\Delta v_S\|\ll\|\Delta v_T\|
$$

或 distillation error 长期不再下降时，才扩容量。

顺序：

### 1.4A

增加：

```yaml
gen_attention_o_lora: true
```

### 1.4B

若仍不足，再测试：

```yaml
k_v_lora: true
```

K/V 只作为 capacity upper bound，不直接升级为默认。

停止规则：

若增加 O/KV 主要带来：

- Pixel MAE 大幅增大；
- texture drift；
- quality 降低；

则回退 Q-only。

---

# Phase 1.5 — Editing 泛化与 T2I 回归测试

最终 SFT adapter 必须同时过两套 guardrail。

### Editing

验证训练目标：

- count；
- relation；
- binding；
- addition/deletion；
- preservation。

### Native T2I regression

重新跑 Phase 0.5：

```text
hard 128
full 800
```

要求：

> 开启训练后 loop 不应系统性破坏 BAGEL 原生 T2I。

重点不是追求 T2I score 大幅上涨，而是确认：

$$
\boxed{
\text{训练编辑能力没有导致生成 prior collapse}
}
$$

---

# Phase 2 — Reinforcement Learning（暂定）

Phase 2 只接收 Phase 1 已通过的 adapter：

```text
Fresh winner
和/或
Persist winner
```

不从 zero-shot LoRA 直接做 RL。

现有：

```text
scripts/train/bagel_loop_grpo_train.py
qwen_latent_cot/bagel/loop_grpo.py
qwen_latent_cot/bagel/flow_grpo.py
```

继续复用：

- SDE rollout；
- exact replay；
- frozen reference adapter；
- clipped GRPO；
- KL；
- semantic reward；
- quality penalty。

Phase 2 前必须完成：

1. 修复 K=0 Base baseline；
2. 支持 Phase 1 v8 adapter schema；
3. Base/Loop per-call `num_loop_tokens`；
4. 若使用 NPU，完成 accelerator abstraction；
5. reward 从纯 T2I GenEval 配方扩展为：
   - edit semantic success；
   - source preservation；
   - image quality。

Phase 2 不在 Phase 1 同时开发，避免无法区分：

$$
\text{SFT supervision gain}
$$

与：

$$
\text{RL reward optimization gain}.
$$

---

# 5. 推荐代码落地结构

```text
qwen_latent_cot/bagel/
├── loop.py                       # 保持核心 LoRA / gating
├── loop_distill.py               # NEW: Δv loss / replay helpers
├── loop_grpo.py                  # Phase 2
├── inferencer.py                 # 增加 per-call num_loop_tokens
└── modeling/
    └── bagel/
        ├── bagel.py              # 尽量不改 loop architecture
        └── qwen2_navit.py        # 尽量冻结

scripts/train/
├── bagel_loop_delta_v_distill.py # NEW Phase 1
└── bagel_loop_grpo_train.py      # Phase 2

configs/training/
├── loop_delta_v_early_fresh.yaml
├── loop_delta_v_early_persist.yaml
└── loop_grpo.yaml                # Phase 2

experiments/data/
├── phase1_train.jsonl
├── phase1_val.jsonl
└── phase1_test.jsonl
```

---

# 6. 推荐实施顺序

### Step 1：先修 inference/training contract

完成：

```text
per-call num_loop_tokens
Base K=0
Teacher K=0
Student K=8
```

并增加单元测试：

```text
same model instance:
base packed_loop_token_indexes.numel() == 0
teacher packed_loop_token_indexes.numel() == 0
student packed_loop_token_indexes.numel() == 8
```

---

### Step 2：实现 Phase 1.1 最小 trainer

只支持：

```text
fresh
[12,20)
K=8
R=2
Q-only
Δv loss
```

先用 100–500 条数据 overfit。

必须看到：

$$
L_{\Delta v}\downarrow
$$

以及：

$$
\cos(\Delta v_S,\Delta v_T)\uparrow.
$$

如果 500 条数据都无法 overfit，不进入大规模训练，优先查 pipeline。

---

### Step 3：构造结构化数据并做 1k–5k smoke

验证：

- teacher reflection 本身是否改善 edit；
- Student 是否逼近 Teacher；
- no-op 是否压住 drift。

Teacher 若本身无增益，则该样本不适合作为 distillation supervision。

---

### Step 4：Phase 1.1 正式训练

训练 Fresh early loop。

---

### Step 5：复制相同初始条件运行 Phase 1.2 Persist

Fresh/Persist 双路并行。

---

### Step 6：Phase 1.3 teacher/data ablation

固定 architecture，不再扫 body。

---

### Step 7：只有 underfit 才进入 Phase 1.4 capacity expansion

先 GEN-O，后 K/V。

---

### Step 8：Phase 1.5 完整 regression

确认 editing gain 与 BAGEL prior preservation。

---

### Step 9：进入 Phase 2 RL

使用 Phase 1 winner adapter 初始化 Flow-GRPO。

---

# 7. Phase 1 的一句话定义

> **Phase 1 does not teach BAGEL how to generate again. It teaches a small recurrent latent controller to reproduce the structural velocity correction induced by a native structured-text reflection teacher, while leaving BAGEL's pretrained visual prior frozen.**

形式化地：

$$
\boxed{
[I_{\rm src},e,r_{\rm struct}]
\Rightarrow
\Delta v_T
\quad\Longrightarrow\quad
[I_{\rm src},e,M]
\Rightarrow
\Delta v_S
}
$$

目标：

$$
\boxed{
\Delta v_S\rightarrow\Delta v_T
}
$$

而最终推理时不再需要：

$$
r_{\rm struct}.
$$

Structured Text Reflection 只存在于训练 teacher 中，最终模型仍然是：

$$
\boxed{
\text{source + instruction + latent Read–Write loop}
}
$$
