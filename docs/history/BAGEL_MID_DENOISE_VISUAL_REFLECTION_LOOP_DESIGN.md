# BAGEL 单轨迹中途视觉反思循环：完整设计

日期：2026-09-14  
状态：实现前设计冻结（implementation brief，不作论文 novelty 声明）  
代号：Mid-Denoise Visual Reflection Loop（MVR-Loop）

## 0. 决策摘要

本方案把“多轮编辑”的能力压缩进一次完整的文生图轨迹：BAGEL 从同一个初始噪声沿
同一条 shifted-Euler 轨迹去噪；在一个中间 timestep，用当前 depth-1 velocity 估计
clean-image preview；BAGEL 自己的理解侧读取 preview 和原始全局 prompt，生成针对当前
seed 的错误诊断；随后生成侧从未被替换的当前 latent `x_t` 继续去噪。

```text
external input: global prompt + one initial noise
                         |
native depth-1 denoising before t_ref
                         |
        v_draft = F(x_t, t, C_prompt, depth=1)
                         |
          x0_preview = x_t - t * v_draft
                         |
         temporary VAE decode -> native ViT
                         |
UND: [reflection system, global prompt, preview ViT, critique query]
                         |
          autoregressive reflection for this seed
                         |
           +-------------+----------------+
           |                              |
 clean-text feedback                 full-cache feedback
 [prompt, reflection]      [reflection system, prompt, preview, query, reflection]
           |                              |
           +-------------+----------------+
                         |
recompute velocity at the same (x_t, t), then continue one trajectory
                         |
                  final VAE decode
```

硬边界如下：

1. 不重新采样初始噪声，不 re-noise，不启动第二条 diffusion trajectory。
2. preview 是只读分支；生成主状态始终是原来的 BAGEL VAE latent `x_t`。
3. preview 允许临时 VAE decode，但不通过 VAE encoder 写回主生成轨迹。
4. 不新增视觉 latent、projector、expert、VAE、输出 head 或新的数据字段。
5. zero-shot 先使用 BAGEL 原生 checkpoint；训练仍优先只开 attention LoRA。

本设计覆盖 `PLAN.md` 中旧的“text-only reflection before image decoding”方案。旧实验保留为
负对照：它测到的是 prompt expansion，不是对当前生成结果的反思。

## 1. 研究问题与可证伪假设

### RQ1：中间时刻是否已经存在可供理解侧诊断的语义草图？

主假设 H1：对当前 guided flow velocity 构造的 `x0_preview` 在某个中间 timestep 已经包含
可读的对象、数量、属性和空间关系；ViT 能把这些错误传给 BAGEL UND。

竞争假设：

- H1a：preview 太早时不可读，反思退化为重复 prompt。
- H1b：preview 太晚时可读，但后续 trajectory 已没有足够控制能力完成修正。
- H1c：VAE decode 的 preview artifact 会诱导错误诊断。

### RQ2：性能来自显式反思，还是来自把 preview 直接作为图像条件？

主假设 H2：干净的 `[prompt, reflection]` cache 能提升语义准确率，说明文字反思本身有效。

竞争假设：完整 cache 的收益主要来自 preview ViT tokens，相当于单轨迹内的 image
conditioning；autoregressive reflection token 可能没有额外贡献。

### RQ3：反思与同 timestep 内部 depth-2 loop 是否协同？

主假设 H3：reflection 先指出当前语义错误，随后 loop-gated attention 提供额外条件使用
能力；二者组合优于单纯增加重复计算。

竞争假设：当前 depth-2 body 对低噪声 visual hidden 的扰动抵消反思收益，或者 reflection
conditioning 本身已经足够，额外 loop 只降低质量。

## 2. “一次生成”的严格定义

本方案中的“一次生成”定义为：

- 一次初始噪声抽样；
- 一套单调递减 timestep schedule；
- 一个连续更新的 `x_t`；
- 不把 preview latent 替换进生成主状态；
- 只返回一次最终图像。

需要区分两个“解码”概念：

| 操作 | 次数 | 作用 |
|---|---:|---|
| diffusion/flow sampling trajectory | 1 | 真正的文生图生成过程 |
| 临时 VAE decode | 1 | 只给 BAGEL ViT/UND 看，不写回 trajectory |
| 最终 VAE decode | 1 | 返回最终图像 |

因此它不是“先生成图片，再做 img2img”；它是同一去噪轨迹中的一次感知—反馈事件。如果
未来要求严格只有一次 VAE decode，则需切换到 latent-only critic；该路线不作为第一版，
因为 BAGEL 原生理解能力主要通过 ViT tokens 建立，直接把生成 VAE token 当作可读图像
缺乏 zero-shot 证据。

## 3. BAGEL 原生结构映射

方案只复用 BAGEL 已有模块：

| 功能 | BAGEL 原生路径 | 本方案用途 |
|---|---|---|
| 文本条件 | `prepare_prompts -> forward_cache_update_text(mode="und")` | 全局 prompt、critique query、reflection 重读 |
| 图像理解 | `prepare_vit_images -> ViT -> connector -> forward_cache_update_vit(mode="und")` | 读取临时 preview |
| 图像生成 | `vae2llm(x_t) -> MoT(mode="gen") -> llm2vae` | 保持原生 flow velocity |
| 图像解码 | `vae_model.decode` | 临时 preview 与最终输出 |
| 内部 loop | MoT body `[10,18)` 的 loop-gated attention LoRA | 反思后的可选额外计算 |

第一版 critique 明确调用 `update_context_image(preview, vae=False, vit=True)`：理解侧只接收
ViT 语义 tokens，不把 preview 再经 VAE encoder 变成 source-image latent。生成侧继续使用
原来的 `x_t`。

当前 `loop_depth=2` 发生在一次 `_forward_flow` 内部，只重复 Transformer body，不会产生
完整 depth-1 图。因此中途视觉反思必须放在 sampler 层：它跨两个相邻 flow evaluation
之间更新条件 cache，但不重启 sampler。

## 4. 中途 preview 的数学定义

BAGEL 当前 sampler 的 velocity 指向 data-to-noise，逆向生成执行：

```text
x_next = x_t - v_theta(x_t, t, C) * dt
```

在 rectified-flow 线性路径约定下，当前 clean estimate 为：

```text
x0_preview = x_t - t * v_draft
```

其中 `v_draft` 必须是当前实际使用的 guided velocity，而不是 CFG 前的 conditional-only
velocity。这样 preview 表示“如果保持当前局部 flow 方向，模型认为会形成的干净图像”。

触发 step 的执行顺序固定为：

```text
v_draft = forward_flow(x_t, t, C_prompt, depth=1)
x0_preview = x_t - t * v_draft
preview_rgb = VAE.decode(x0_preview)              # read-only
reflection, caches = review(prompt, preview_rgb)
v_feedback = forward_flow(x_t, t, C_feedback, post_depth)
x_next = x_t - v_feedback * dt
```

关键点：`v_draft` 只用于预览，不执行 Euler update；反馈生成后在相同 `(x_t, t)` 上重新计算
`v_feedback`，所以反思从触发 step 当场生效。该 step 多一次 flow evaluation，但没有第二条
trajectory。

### 4.1 reflection timestep 不预设为单点真理

现有 `t>=0.75` loop gate 是为保护低噪声质量而选，不能直接证明 `t=0.75` preview 可供
VLM 诊断。第一轮先保存以下 shifted model-time 的 `x0_preview`：

```text
t_ref candidates = {0.75, 0.60, 0.45, 0.30}
```

当前主线默认采用 `reflection_t=0.75`，并将 depth-2 语义窗口保留到
`post_loop_stop_t=0.35`；这样反思后的同一轨迹仍有足够的构图更新步数。两项都可通过 CLI
覆盖，不把该默认值当作已验证最优点。

选择规则是：选“最早（最大 t）且已达到可用诊断准确率”的 checkpoint，为后续修正留下
最多 trajectory。选择依据必须来自 preview 的对象/数量/属性/位置可读性和 reflection
diagnosis precision/recall，而不是主观挑图。

## 5. Critic context 与反思输出

### 5.1 每个 seed 独立反思

reflection 必须在每个 seed 的 `x0_preview` 产生后单独生成。禁止像旧实验一样只根据 prompt
生成一次并由多个 seed 共用。

### 5.2 Critic token 顺序

完整 critic context 为：

```text
C_critic_in = [system, global prompt, preview ViT, critique query]
C_critic_out = [system, global prompt, preview ViT, critique query,
                AR start token, generated reflection tokens]
```

critique query 固定为：

```text
Audit the preview against the global prompt now.
Return exactly five concise lines:
COUNT: <expected versus observed, or OK>
POSITION: <expected versus observed, or OK>
ATTRIBUTES: <expected versus observed, or OK>
DETAILS: <expected versus observed, or OK>
CORRECTION: <positive imperative instructions that fix all confirmed errors while preserving correct content>
```

全局 prompt 必须作为独立字段先进入 context，不能只嵌在 critique query 中。reflection
默认 deterministic greedy decode，长度上限先设 128 tokens；输出文本、token ids、是否因
长度截断都必须记录。

### 5.3 反思质量的最小契约

合格 reflection 应同时满足：

- 指向当前 preview 中实际存在的错误，而不是只复述目标；
- 区分 missing 与 extra，避免把“目标需要四只”误写成“图中已有四只”；
- 保留全局 prompt 中没有出错的实体和关系，不建议无关改动；
- correction instruction 简短，避免把长篇自然语言当作新的复杂 prompt；
- preview 改变时，同一 prompt 下的 reflection 应随 seed 改变。

## 6. 两条主要反馈传输路径

反思文字在两条路径中都会真实 autoregressive decode 并保存。区别只在后续生成侧读取的
cache。

### 6.1 T1：Clean text re-read（只保留 prompt + generated reflection）

```text
C_post_text = KV_encode([global prompt,
                         "Correction after inspecting the current draft:",
                         decoded reflection text])
```

此路径丢弃 preview ViT 和 critique query 的 KV，仅保留原始 prompt 与反思文字。它是
“理解通过可解释文字赋能生成”的最干净实验：如果它提升语义准确率，收益不能归因于直接
把 preview 当作 image condition。

实现上必须把 reflection decode 成可记录文本，再通过 BAGEL 原生
`prepare_prompts/forward_cache_update_text` 重读。不要从 `C_critic_out` 截取尾部 K/V 后
直接拼到 prompt cache：这些 K/V 已在 preview 和 query 的上下文中计算，并带有对应 RoPE
位置，简单拼接不是等价操作。

### 6.2 T2：Full cache reuse（保留完整视觉诊断 cache）

```text
C_post_full = C_critic_out
            = [reflection system, global prompt, preview ViT,
               critique query, generated reflection]
```

此路径不重新编码 reflection，而是让后续生成 flow 直接读取理解侧形成的完整 cache。它
保留了三类信息：preview 的视觉证据、critique query 的任务定义、image-conditioned
reflection hidden/KV。

该路径可能最强，但它与“只靠反思文字”不同：preview ViT tokens 会直接形成图像条件，
因此更接近同轨迹内的隐式 image editing。若它胜过 T1，不能直接声称提升来自文字推理。

### 6.3 Full-cache 的拆解消融

为判断完整 cache 中到底哪部分有效，定义以下累加式 contexts：

| cache id | 后续条件内容 | 回答的问题 |
|---|---|---|
| `C0_prompt` | `[prompt]` | 原生轨迹基线 |
| `C1_text` | `[prompt, reflection]` | 纯文字反馈是否有效 |
| `C2_visual` | `[prompt, preview ViT]` | 直接视觉条件本身是否有效 |
| `C3_visual_query` | `[prompt, preview ViT, critique query]` | query 是否只是额外 prompt |
| `C4_full` | `[reflection system, prompt, preview ViT, critique query, reflection]` | 完整闭环效果 |

`C4_full - C3_visual_query` 才是“在完整视觉 cache 上新增 generated reflection token”的
边际贡献；`C2_visual - C0_prompt` 则量化直接 image-conditioning 泄漏。

### 6.4 CFG cache 契约

Clean-text 路径继续使用纯 T2I CFG：

```text
conditional:       [prompt, reflection]
text-unconditional: []
cfg_img_scale:      1.0
```

Full-cache 路径使用 BAGEL 原生双条件拆分，不能把两个独立 cache 事后拼接：

```text
conditional:        [reflection system, prompt, preview, query, reflection]
text-removed:       [preview]
image-removed:      [reflection system, prompt, query, reflection]
```

主实验先固定 `cfg_img_scale=1.0`，避免额外放大 preview image condition；随后只把 BAGEL
原生 image-CFG scale 作为敏感性实验。所有 cache 必须按各自 token 顺序从空 context 构建，
并保存 `kv_lens`、RoPE 末位置和 token provenance。

## 7. 反思与 depth-2 loop 的组合

为避免混淆，使用两个独立术语：

- `draft_depth=1`：触发反思前用于生成 preview 的 flow forward 永远是原生 depth 1；
- `post_feedback_depth in {1,2}`：反思后重新计算当前 velocity 和后续步骤所用的 body depth。

旧的 `loop_timestep_threshold` 只支持 `t >= threshold`，不适合“反思发生后才启用 loop”。
新 sampler 应显式支持区间：

```text
pre-reflection:             depth 1
reflection trigger step:    draft depth 1 -> review -> recompute with post depth
post-reflection semantic window [t_stop, t_ref]: selected post depth
late detail window t < t_stop: depth 1
```

第一轮先做两个 scope：

- `trigger_only`：仅触发 step 的重算使用 depth 2，最容易归因；
- `window`：从 `t_ref` 到 `t_stop` 使用 depth 2，测试持续隐式编辑。

`t_stop` 需通过质量/语义敏感性确定，不能沿用旧阈值而不验证。depth 2 继续使用
`[10,18)`、boundary state 与 residual cap `0.05` 作为起点；原生 checkpoint zero-shot 时
不加载任何 adapter，训练实验才加载 loop LoRA。

## 8. 完整对照组

### 8.1 Gate A：实现与轨迹不变量

| id | 条件 | 必须满足 |
|---|---|---|
| `A0_native` | 原始 `generate_image` | reference |
| `A1_stepwise_no_hook` | refactor 后的逐步 sampler，无 hook | 与 A0 latent/image parity |
| `A2_preview_noop` | 计算 preview 和 reflection，但丢弃全部反馈 | 与 A1 相同；证明只读分支未污染 `x_t` |
| `A3_recompute_prompt` | 触发 step 用相同 prompt 重算一次 | compute-matched control |

text decode 若使用 sampling，必须使用独立 RNG；zero-shot Gate A 固定 greedy decode，保证
preview 分支不改变 diffusion/SDE RNG 状态。

### 8.2 Gate B：cache 内容拆解，post depth 固定为 1

固定同一个 `x_t`、同一 reflection、同一后续 schedule：

| id | post cache | 目的 |
|---|---|---|
| `B0_prompt` | C0 | 原生继续轨迹 |
| `B1_clean_text` | C1 | 纯文字反思 |
| `B2_visual` | C2 | preview 直接条件 |
| `B3_visual_query` | C3 | preview + critique prompt，无反思答案 |
| `B4_full_cache` | C4 | preview + query + generated reflection |

这组首先回答“full cache 的优势是不是主要来自直接看到 preview”。只有 B1 或
`B4-B3` 为正，才能支持 reflection 本身有用。

### 8.3 Gate C：reflection × loop 主交互矩阵

只保留最关键的三类 feedback transport，与 post depth 做 3×2：

| feedback | post depth 1 | post depth 2 |
|---|---|---|
| none | `C0_D1` | `C0_D2` |
| clean text | `CT_D1` | `CT_D2` |
| full cache | `FC_D1` | `FC_D2` |

差分解释：

```text
CT_D1 - C0_D1: reflection-only effect
C0_D2 - C0_D1: loop-only effect
CT_D2 - CT_D1: loop marginal effect under clean reflection
FC_D1 - CT_D1: full visual-cache effect beyond text reflection
FC_D2 - FC_D1: loop marginal effect under full cache
(CT_D2-C0_D2) - (CT_D1-C0_D1): clean reflection × loop interaction
```

所有分支共享同 prompt、seed、initial noise、触发前 `x_t`、timestep schedule 和最终图像
尺寸。分支只能从触发点复制一次 `x_t`，禁止分别重跑前半段后再宣称 paired。

### 8.4 Gate D：必要的上界与反事实

- `oracle_error_text`：由 GT atom 与 preview judge 生成简短正确纠错文本，测试生成侧是否
  仍有可修正能力；它只作为上界，不作为训练数据格式变更。
- `shuffled_preview`：同 prompt 下打乱 seed 的 preview；若 reflection 仍不变，说明 critic
  没有真正看图。
- `shuffled_reflection`：把另一 seed 的 reflection 写入；验证反馈是否与当前视觉状态匹配。
- `more_nfe`：不反思但增加等量 flow compute；排除收益只是计算量增加。

## 9. 评测与成功标准

### 9.1 评测集不变

继续使用现有高 atomicity GenEval2 prompt pool 和 held-out protocol，不增加新的数据格式。
重点分别报告 counting、attribute、position、object、verb；不能只报 aggregate mean。

### 9.2 指标

| 层级 | 指标 |
|---|---|
| preview 可读性 | preview 的 Soft-TIFA/GenEval per-atom score |
| critic grounding | error precision、error recall、seed sensitivity、shuffled-preview drop |
| 最终语义 | per-atom Soft-TIFA、log-GM、all-correct rate |
| 图像质量 | DiNa/FLUX RM 质量 guardrail，外加 artifact 检查 |
| 轨迹影响 | final latent MAD/cos、触发 step velocity delta、后续 delta 曲线 |
| 效率 | 额外 NFE、ViT latency、AR token 数、KV 显存、总 latency |

### 9.3 Zero-shot 通过门槛

第一版不以小样本最终分数直接宣称成功。进入训练前至少满足：

1. `A1` 与 `A0` 达到 BF16 数值容差内 parity，`A2` 不改变最终 latent。
2. reflection 相对 shuffled-preview 有显著更高的错误匹配率，并且不同 seed 不再共用同文。
3. `B1_clean_text` 或 `B4_full_cache-B3_visual_query` 至少出现稳定的正语义差分。
4. 最佳反馈分支的独立质量指标不低于预设 tolerance，且不存在明显复制/重影 artifact。
5. 相对 `more_nfe` control，语义增益不能仅由额外计算解释。

## 10. 代码落点

当前首版实现已落到：

```text
qwen_latent_cot/bagel/modeling/bagel/bagel.py
qwen_latent_cot/bagel/inferencer.py
scripts/evaluate/bagel_mid_denoise_visual_reflection.py
scripts/evaluate/run_bagel_mid_denoise_visual_reflection.sh
```

其中评测脚本默认输出 Gate A、B、C 所需的 9 个 frozen-checkpoint 分支；critic 现在按
`COUNT → POSITION → ATTRIBUTES → DETAILS → CORRECTION` 固定审计顺序，视觉 CFG 主线默认
为 `cfg_img_scale=1.0`，并在共享 trigger 保存 feedback/depth 的 flow delta。正式执行前仍须先跑
`native_d1` 对 `prompt_d1` 的单 seed parity gate。实现和命令已经准备，但不把未执行的
parity、preview 可读性或语义收益写成已验证结论。

### 10.1 `modeling/bagel/bagel.py`

把现有 `generate_image` 的单体循环机械拆成可暂停的原生 sampler primitive：

```python
state = prepare_image_sampler(...)
v_t = predict_velocity(state, contexts, loop_config)
state = euler_step(state, v_t)
latents = finish_image_sampler(state)
```

`generate_image(...)` 保留为兼容 wrapper，逐步调用 primitive。无 reflection 时执行顺序、
dtype、CFG 和 Euler 更新必须与原实现一致。语义 review 逻辑不放进 BAGEL model；model 只
暴露可暂停/继续的数值采样接口。

### 10.2 `qwen_latent_cot/bagel/inferencer.py`

新增高层编排：

```python
gen_image_with_mid_reflection(
    global_prompt,
    reflection_t,
    feedback_mode,          # none | clean_text | full_cache
    post_feedback_depth,
    post_loop_scope,
    ...,
)
```

并新增：

- `decode_clean_preview(x_t, v_t, t, image_shape)`；
- `build_visual_critic_context(global_prompt, preview)`；
- `generate_text_trace(context)`；
- `build_clean_reflection_context(global_prompt, reflection)`；
- `build_full_feedback_cfg_contexts(trace)`。

### 10.3 文本生成 trace

当前 `gen_text()` deep-copy context，并只返回字符串；完整 cache 被丢弃。新增返回对象：

```python
@dataclass
class TextGenerationTrace:
    text: str
    token_ids: Tensor
    context_before: GenContext
    context_after: GenContext
    stopped_on_eos: bool
    truncated: bool
```

底层 `generate_text` 需选择性返回更新后的 `past_key_values`、`key_values_lens`、RoPE 末位置。
`context_after` 必须对应真实 AR forward 产生的 KV；禁止通过重新编码文本伪造
`full_cache`。

### 10.4 新评测脚本

建议新增：

```text
scripts/evaluate/bagel_mid_denoise_reflection_probe.py
scripts/evaluate/bagel_mid_denoise_feedback_ablation.py
scripts/evaluate/render_mid_denoise_feedback_ablation.py
scripts/evaluate/run_bagel_mid_denoise_feedback_ablation.sh
```

先做原生 checkpoint zero-shot；不覆盖旧的 text-only reflection 脚本，避免历史结果含义
被静默改变。

## 11. 输出与审计格式

每个 prompt×seed 至少保存：

```text
prompt.txt
seed_<seed>/x0_preview_t<time>.safetensors
seed_<seed>/preview_t<time>.png
seed_<seed>/reflection.txt
seed_<seed>/reflection_tokens.json
seed_<seed>/cache_manifest.json
seed_<seed>/<variant>.png
seed_<seed>/<variant>_latent.safetensors
report.json
index.html
```

`cache_manifest.json` 必须记录每个 context 是否包含：global prompt、preview ViT、critique
query、reflection tokens、VAE source-image tokens；同时记录 `kv_lens`、末端 RoPE、CFG
drop branches。这样可阻止“标称 clean text，实际仍带 preview cache”的实验污染。

## 12. 训练路线

### Phase 0：原生 zero-shot

不加载 SFT/RL adapter。先判断三件事：preview 是否可读、UND 是否能指出真实错误、反馈后
剩余 trajectory 是否仍有纠错能力。任一项失败都先修正机制，不启动 RL。

### Phase 1：format-only SFT

只有反思格式不稳定时，才对 UND attention LoRA 做短 SFT，使输出遵循固定诊断格式；它
不承担最终性能提升，也不训练新视觉 latent。生成基座、DiT/GEN base、VAE、ViT、MLP、
norm 继续冻结。

### Phase 2：RL

首轮冻结 reflection policy，仅训练反思后启用的 GEN attention loop LoRA。每个 rollout
从同一触发点复制 latent，reward 使用最终图像相对原生 continuation 的 paired delta：

```text
semantic_delta = R_sem(feedback) - R_sem(native continuation)
quality_delta  = R_quality(feedback) - R_quality(native continuation)
objective      = semantic_delta - wq * relu(-tolerance - quality_delta)
```

若 zero-shot 证明 critic 常漏报真实错误，再考虑对 UND attention LoRA 做 token-level RL；
此时 reflection token log-prob 和 image-transition log-prob 必须分开记录并分别加 KL，不能
只训练生成侧却声称 critic 学会了反思。

## 13. 主要风险与止损条件

| 风险 | 可观察信号 | 缓解/止损 |
|---|---|---|
| preview 不可读 | reflection 与 shuffled-preview 几乎相同 | 后移 `t_ref`；仍失败则停止视觉闭环 |
| feedback 太晚 | oracle correction 也改不动最终结果 | 前移 `t_ref` 或扩大 post-feedback window |
| full cache 复制草图错误 | identity 高但语义错误保持 | 降低/关闭 image CFG；优先 clean text |
| reflection 只是 prompt expansion | 不提 preview 独有错误 | 强化 critique contract；做 seed/shuffle 测试 |
| depth2 破坏质量 | CT_D2/FC_D2 质量低于 D1 | 缩成 trigger-only 或放弃 post depth2 |
| KV 路径混淆 | cache manifest 与 variant 不符 | 测试逐 token provenance，实验结果作废 |
| 计算收益混淆 | more-NFE 与 feedback 同等提升 | 不宣称理解反馈有效 |

最强反对意见是：`full_cache` 可能只是把 preview 当作 source image 做编辑，而非“理解赋能
生成”。本设计用 `C2_visual`、`C3_visual_query` 和 `C1_text` 明确拆开该混淆；只有文字
路径或 reflection token 的边际贡献为正，才能支持反思叙事。

## 14. 实施顺序

1. 机械拆分 sampler，并完成 `A0/A1/A2` parity。
2. 实现单 seed、多 `t_ref` 的 preview+reflection probe，先看 UND 是否真的诊断当前图。
3. 实现 `C0-C4` cache provenance 与 Gate B zero-shot。
4. 完成 3×2 Gate C、自动评分和 HTML 组图。
5. 只有 zero-shot gate 通过后，再接入 format SFT 与 paired RL。

## 15. 相关工作边界

BAGEL 本身提供统一理解、生成和自由图像操作能力，因此本方案优先复用其 ViT/UND 与
MoT/GEN 原生路径，而不是增加外部 critic 或新表示空间。显式 iterative refinement 已显示
“看成图—反馈—再生成”能够改善复杂组合，但其多次生成流程不能直接证明中途单轨迹反馈
有效。本设计要验证的正是这条尚未被本项目实验证实的边界：中间 clean estimate 是否足以
触发有用诊断，以及诊断能否在不 re-noise 的剩余轨迹中改变结果。

参考：

- [BAGEL: Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683)
- [Iterative Refinement Improves Compositional Image Generation](https://arxiv.org/abs/2601.15286)
- `artifacts/idea/bagel_loop_supervision_20260911/literature_survey.md`
- `artifacts/experiment/zero_shot_mid_denoise_prompt_switch_20260909/summary.md`

## 16. P0：直接理解表征桥接

KV-only 实验只能证明反思改变了条件流，不能证明 GEN 的循环体读取到了一个可执行的
理解状态。P0 因此保留 `generate_text` 最后一个非 EOS token 的 UND hidden，作为
`s_ref`，并只在 depth 2 的第二次 body 入口更新原生 BOI/EOI text token：

```text
s_ref = final UND reflection hidden
b     = GEN BOI/EOI hidden at layer 10 entry
b'    = bounded_residual_merge(b, broadcast(s_ref), scale=0.20)
GEN    = layers [10,18) repeated from {VAE entry unchanged, BOI/EOI=b'}
```

这不是新 latent：`x_t`、VAE token、噪声轨迹和 scheduler 均不变。`s_ref` 在每个
反思后的 suffix step 只读复用，不跨 step 累加。默认参数为 `None`，所以已有推理、SFT
和 RL 路径保持原行为。第一阶段只用原生 checkpoint 做 forced-count zero-shot；如果仍
不能改变数量，则先否定“无训练直接桥接即可执行语义编辑”，不接训练。
