UMM-T2ILoop
Anchored Native Loop + Memory Scratchpad
面向 BAGEL/UMM 的后训练循环生成架构设计与实验方案
Design status: Proposed implementation spec
核心原则：保留原始 T2I 先验，将 memory 从“主 reasoning state”降级为安全 scratchpad，逐步把 recurrence 迁移到 GEN hidden。
Reference basis: Looped Diffusion Transformer (arXiv:2609.40305v1) + 当前 training-free memory-loop 实验观察

0. Executive Summary / 设计结论
本文建议将当前 UMM-T2I 的 training-free memory loop 从“memory-only recurrent state”升级为 Anchored GEN Loop + Memory Scratchpad。核心不是删除 memory，也不是直接复制 Looped-DiT 的 full native loop，而是利用原始 BAGEL/UMM T2I 路径作为不可破坏的 anchor，在同一个 diffusion timestep 内增加受控的 GEN hidden refinement。
设计目标是同时满足三件事：第一，loop 关闭时严格退化为原始 BAGEL；第二，不修改原始 prompt KV cache 与 sampler state x_t；第三，允许额外 recurrent compute 直接作用于分布式 visual hidden，而 memory 只承担低带宽、低风险的全局 workspace / UND↔GEN 交换通道。
决策项 | 推荐方案 | 原因
Prompt KV | 全程 read-only / frozen anchor | 此前实验已表明对 prompt KV 的强干预会显著破坏生成质量；后训练阶段不应先改 condition manifold。
x_t | 同一 diffusion timestep 内固定 | 内部 loop 只增加 hidden compute，不修改 sampler trajectory；避免把 inner loop 与 denoising step 混在一起。
GEN hidden | 成为主要 recurrent working state | 空间、数量、关系、属性绑定等视觉约束是 distributed representation，不应全部压入 K=8/16 的 memory bottleneck。
Memory | 保留，但降级为 scratchpad | 当前唯一能在质量基本不掉的前提下触发语义编辑的安全 intervention topology，应作为脚手架保留。
Loop transition | Anchored re-entry，而非直接 H_e→H_s recycle | 预训练 BAGEL 未训练过 native recurrence，直接把 body exit 塞回 body entry 存在明显 depth-distribution mismatch。
训练目标 | Direct Flow Deep Supervision | 每一轮都直接对原始生成目标负责；第一阶段不使用 final-loop distillation，避免过早把各 loop 拉成相同状态。
Update control | loop-only zero-init gate | 只控制 extra loop，不改变 base pass；逐步学习“哪里需要改、改多少”。

最终建议的系统可以概括为：
Frozen anchors:  P_KV^0, x_t, G_0
Recurrent states:  ΔG_r, M_r
G̃_r = G_0 + A_φ(ΔG_{r-1}, M_{r-1})
(G_r, M_r) = B_θ(G̃_r, M_{r-1}; P_KV^0)
G_out^r = G_base + α_r · (G_r − G_base)
v_r = F_{e:L}(G_out^r),   L = L_flow^R + λ_DS · Mean_{r<R} L_flow^r
1. 动机：为什么不继续只做 Memory，也不直接做 Native Full Loop
1.1 当前 training-free memory loop 已经证明了什么
当前实验最有价值的结论并不是“memory 内容已经被证明具有正确语义”，而是已经找到了一条在 pretrained UMM 上相对安全的 recurrent intervention path：保留原始 prompt condition 与主生成路径，在中间插入少量 memory tokens，通过重复读取 prompt KV 与 body 内 attention 交互，可以产生数量、位置等可见语义变化，同时整体图像质量不出现 prompt-KV mask / direct condition rewrite 那样的大幅坍塌。
同时，normal / shuffled / frozen / zero 等对照已经提示：当前 effect 很大一部分可能来自额外 token 带来的 attention routing、denominator、K/V position 与 recurrent compute topology，而不一定来自 memory token 本身承载了精确的 semantic content。因此，下一步不应继续默认“memory 就是 reasoning state”，而应把它视为一种低带宽、安全的介入接口。
1.2 Memory-only 的结构性瓶颈
若把全部 refinement 都压到 K=8/16 个 memory slots，模型需要在少量 token 中同时表达 object identity、count、spatial relation、attribute binding、layout correction、当前 x_t 的局部结构以及下一轮修改方向。这是非常强的 bottleneck。更关键的是，图像的空间与对象关系天然分布在大量 GEN/image tokens 中；仅让 memory persistent，而 GEN state 每轮被重置或不保留，会阻断 distributed visual state 的自然迭代。
1.3 为什么不能直接照搬 Looped-DiT 的 native loop
Looped-DiT 的 shared middle blocks 从训练开始就被优化为 recurrent operator：h^(r+1)=B(h^r)，并通过 Deep Supervision 让多个 recurrent depths 都直接对同一个 clean-image / flow target 负责。论文还发现 naïve looping 会出现 performance saturation / degradation，并伴随可线性解码的空间信息下降，因此额外设计 Self-Modulating Attention 来约束 repeated write。
BAGEL/UMM 的情况不同：当前 backbone 是 pretrained non-loop model。原始 body B=F_{s:e} 学到的是“layer-s distribution → layer-e distribution”，而不是“layer-e output → 再次进入 layer-s”。若直接执行 G_{r+1}=B(G_r)，会把 depth mismatch、residual scale drift 与真实 recurrent refinement 混在一起。后训练阶段首先要保护 prior，而不是假设现有 block 已经是 native recurrent operator。
1.4 设计目标
保住原始 T2I prior：loop off 时输出必须与原始 BAGEL 数值一致或在浮点误差范围内一致。
不直接修改 prompt KV 与 x_t：二者作为 frozen anchors；所有新增行为都通过额外 hidden computation 完成。
把 recurrence 从 memory-only 扩展到 GEN hidden：让分布式视觉表示自己迭代，而不是把所有信息压进少量 memory slots。
Memory 保留为安全 scaffold：可作为 UND↔GEN workspace，但系统必须允许 K=0，不能在理论上依赖 memory 才能运行。
训练由浅入深：先只训练 adapter / gate / memory projection，再小学习率放开 loop body；不一开始破坏 backbone。
所有实验必须可解释：先做 topology 对照，再做训练；semantic gain 与 base quality retention 必须同时报告。
1.5 非目标（第一版明确不做）
不让 prompt KV 在 loop 内动态演化。
不在 inner loop 内更新 x_t；sampler state 只在外层 denoising step 更新一次。
不直接采用 full hidden recycle（H_e 直接塞回 H_s）作为默认主方案。
不先引入复杂 RL / trajectory distillation / 人为 margin loss；先证明 recurrent topology 与 direct flow supervision 本身有效。
不要求 memory token 必须可解释为显式 textual CoT；它首先是计算与信息交换通道。
2. 架构原理与符号定义
2.1 原始 UMM-T2I 前向
将当前 T2I forward 抽象为三段：prefix F_{1:s}、可循环 body B=F_{s:e}、suffix F_{e:L}。在 diffusion timestep t，prompt 条件记为 P，sampler state 记为 x_t。
(C_0, G_0) = F_{1:s}(P, x_t)
G_base = B(G_0; C_0)
v_base = F_{e:L}(G_base; C_0)
其中 C_0 表示在 loop entry 可供 body 使用的原始 condition / prompt representation（实现中可以对应固定 prompt KV cache 或 UND-side condition states），G_0 表示 GEN/image token 在 loop entry 深度的 hidden state。
2.2 三类状态必须严格区分
状态 | 是否在 inner loop 内更新 | 作用
x_t | 否 | 外层 diffusion/sampling state；一个 timestep 内固定。
C_0 / P_KV^0 | 否 | 预训练 condition prior；read-only anchor。
G_r | 是 | 主要 visual working state；承担 distributed constraint refinement。
M_r | 可选，是 | global scratchpad / UND↔GEN workspace；低带宽辅助状态。
ΔG_r | 是 | 相对于 base path 的 visual correction，用于安全 re-entry 与 residual merge。

2.3 四个核心 invariant
Base equivalence：R=0 或所有 loop gate=0 时，完整输出严格等价于原始 BAGEL T2I。
Condition immutability：extra loop 只能读取 P_KV^0，不能覆写或持久化新的 prompt KV。
Sampler immutability：inner loop 不改 x_t；只有最终 v_t 进入原 sampler 更新。
Anchored recurrence：每轮以原始 loop-entry G_0 为 anchor 注入 correction，而不是 free-running 地把前一轮 body exit 直接当作下一轮 entry。

3. 总体架构：Anchored GEN Loop + Memory Scratchpad
                       Frozen Prompt / UND Condition                                P_KV^0                                  │                                  │ read-only                                  ▼P, x_t ──► F_{1:s} ──► (C_0, G_0) ────────────────────────────────┐                           │                                       │ anchor                           │ normal base pass                      │                           ▼                                       │                       B = F_{s:e}                                 │                           │                                       │                           ▼                                       │                        G_base                                     │                           │                                       │              ┌────────────┴──────────────┐                        │              │      EXTRA LOOP r         │                        │              │                           │                        │              │  ΔG_{r-1}, M_{r-1}        │                        │              │          │                │                        │              │          ▼                │                        │              │   Re-entry Adapter A_phi  │                        │              │          │                │                        │              │          └────► G_0 + correction ◄────────────────┘              │                         │              │                         ▼              │             shared body B (UND + GEN)              │                         │              │              ┌──────────┴──────────┐              │              ▼                     ▼              │             G_r                   M_r              │              │                     │              │              └──────────┬──────────┘              │                         ▼              └──────────── next loop / exit ────────────                 G_out^r = G_base + alpha_r (G_r - G_base)                                  │                                  ▼                              F_{e:L}                                  │                                  ▼                                 v_r

3.1 模块 A：Frozen Anchor Extractor
职责：运行完全原始的 prefix F_{1:s}，得到 C_0 与 G_0，并构造后续 loop 需要的只读 prompt KV/cache。该模块不新增参数。
实现要求：
保留当前 forward_inference 中原始 prompt/text condition 的缓存方式。
extra loop 开始后禁止对 condition tokens 做 persistent hidden update；即使 body 内计算产生临时 UND hidden，也不能替换下一轮所读的 anchor prompt KV。
CFG 的 cond / text-removed / img-removed（如仍使用三分支 CFG）必须各自保存独立 anchor cache，绝不能跨分支共享 recurrent state。
3.2 模块 B：Base Pass（不可破坏的原始路径）
G_base = B(G_0; P_KV^0)
第一次 body 执行必须走原 BAGEL 逻辑，不启用 loop-only gate，不注入 re-entry adapter，不改变 attention 公式。Base pass 提供两个东西：原始可用生成状态 G_base，以及所有 loop correction 的参考点。
强制单元测试：在 `enable_t2i_loop=False`、`loop_depth=0`、`loop_output_alpha=0` 三种关闭方式下，velocity 与当前 baseline 的最大绝对误差应处于浮点可接受范围。
3.3 模块 C：Re-entry Adapter A_φ
目的：解决 pretrained non-loop body 的 depth-distribution mismatch。不要把上一轮 G_r 直接当作下一轮 layer-s hidden，而是只把“上一轮相对于 base 的 correction”投影回 loop-entry manifold。
ΔG_{r-1} = G_{r-1} − G_base
δ_r = A_φ(ΔG_{r-1}, M_{r-1})
G̃_r = G_0 + δ_r
第一版推荐 A_φ 结构：RMSNorm(ΔG) → low-rank Linear(d→d_a→d)；memory 存在时可将 pooled/projected M 作为加性或 FiLM 条件。A_φ 的最后一层 zero-init，使训练初始时 δ_r≈0，extra loop 从原始 G_0 附近开始。
子模块 | 推荐默认 | 可选升级
GEN re-entry | RMSNorm + Linear(d→d/8→d), last layer zero-init | LoRA-style low-rank projection / token-wise FiLM
Memory injection | mean/attention pool(M) → Linear → additive bias | cross-attention from GEN queries to M
位置处理 | 继承 G_0 的原 position ids / RoPE 语义 | 不新增虚构 depth positional embedding，除非后续消融证明需要

3.4 模块 D：GEN recurrent state
GEN hidden 是下一版 UMM-T2ILoop 的主 recurrent state。它承担空间布局、对象数量、属性绑定、关系等 distributed visual constraints。每一轮 body 都可以更新 GEN tokens，但下一轮不是直接 free-run，而是通过 ΔG→A_φ→G_0 anchored re-entry。
这种设计兼顾两点：一方面让 image/GEN representation 真正具有 recurrent capacity；另一方面把每轮输入限制在 pretrained loop-entry manifold 周围，避免不断累计 drift。
3.5 模块 E：Memory Scratchpad M_r
Memory 保留，但定义改变：它不是“必须包含全部 edit gap 的 semantic bottleneck”，而是可选的 global workspace。系统必须支持 K=0、K=8、K=16 等设置，以实验回答 memory 的真实价值。
在 UMM 中，最自然的使用方式是把 M 作为 UND↔GEN shared workspace：UND expert 读取 prompt 与当前 visual state，更新 M；GEN expert 再读取 M 形成 distributed correction。
M_{r+1} = UND(M_r; P_KV^0, G̃_r)
G_r = GEN(G̃_r; P_KV^0, M_{r+1})
第一版不要求实现严格的先 UND 后 GEN 两阶段调度；如果当前 block 内 expert routing 已经自然完成 UND/GEN 交互，可先保持现有路由，仅确保 M 是 persistent state 而 prompt anchor 非 persistent。
3.6 模块 F：Loop-only Write Gate
Looped-DiT 的关键发现之一是 repeated attention write 会逐渐侵蚀局部/空间信息。对于 post-training UMM，第一版更适合采用保守的 zero-init / near-zero loop gate，而不是立即把原始 attention 改成 XSA。
ΔH_loop = g_{r,l,t} · ΔH_body
g_{r,l,t} = sigmoid(b_g + f_g(RMSNorm(H)))
base pass 中强制 g=1 且走原实现；仅 extra loop 使用新增 gate。为了让初始行为接近 no-op，可以把 extra-loop gate 的 bias 初始化为负值，使 g≈0.01~0.05。第一阶段甚至可以只用每层一个 scalar gate；验证后再升级为 token/head-dependent gate。
3.7 模块 G：Residual Output Merge
ΔG_r^out = G_r − G_base
G_out^r = G_base + α_r · ΔG_r^out
不要让 G_r 无条件完全替换 G_base。通过 α_r 让原始 T2I path 永远存在。α 可以是全局标量、per-loop 标量或 timestep-conditioned 标量；第一版推荐 per-loop scalar，初始化为 0。
这意味着初始 checkpoint 在数值上仍是原始 BAGEL；训练学习的是“是否以及多大程度相信 extra recurrent correction”，而不是从第一步就强迫模型切换到新路径。
3.8 模块 H：Shared Suffix / Functional Readout
v_r = F_{e:L}(G_out^r; P_KV^0)
所有 loop depth 复用同一个原始 suffix。它既是最终输出头，也是最重要的 functional probe：如果某个 recurrent state 只有 hidden-space 变化但经过 shared suffix 后没有形成有用 velocity correction，则不应被解释为有效 reasoning。
3.9 CFG 分支处理
若当前 BAGEL 使用 cond / text-removed / image-removed 三分支 CFG，推荐每个分支独立执行同构 loop，并分别维护 G_0、G_base、ΔG_r、M_r、gate state。CFG 组合发生在每个分支得到最终 velocity 之后。禁止 cond 分支 evolved memory 与另一个分支 frozen memory 混用，否则 recurrent depth 不一致会污染对照。
4. Loop 架构与执行算法
4.1 推荐默认模式
配置 | 默认值（建议起点） | 说明
loop_start_layer s | 沿用当前机制验证较稳定的中后层位置 | 优先保持已有 training-free loop body，减少变量。
loop_end_layer e | 沿用当前 body end | 确保 suffix 保持原模型。
extra_loop_depth R | 2 或 3 | 先验证 1→2→3 的 gain curve；不要一开始追求 8/16 loops。
memory_slots K | 8（同时必须支持 0） | 8 作为当前安全 scaffold；K=0 是必要对照。
re-entry adapter rank | d/8 或 d/16 | 足够表达 correction，但保持低容量。
output α | 0 初始化，可学习 | 保证初始模型=base。
loop gate | extra-loop only，near-zero init | 保护 pretrained residual stream。
prompt KV | 固定 | 第一版不可训练、不可 loop 更新。
x_t | 固定 | 一个 timestep 内只读。

4.2 Training-free / inference 伪代码
def forward_t2i_loop(P, x_t, cfg, R, use_memory=True):    # 1) Frozen anchor extraction    C0, G0, prompt_kv0 = prefix(P, x_t)        # original F_{1:s}    # 2) Exact original base path    G_base = body(G0, prompt_kv0, loop_mode=False)    if R == 0:        return suffix(G_base, prompt_kv0)    # 3) Initialize recurrent states    dG = zeros_like(G_base)    M  = init_memory(C0, G0) if use_memory else None    velocities = []    loop_logs = []    for r in range(1, R + 1):        # anchored re-entry; training-free version can initially use        # identity/hand-crafted small residual mapping for topology tests        delta_entry = reentry_adapter(dG, M)        G_entry = G0 + delta_entry        # shared body; prompt KV remains read-only        G_r, M_r, stats = body(            G_entry,            prompt_kv0,            memory=M,            loop_mode=True,       # enables loop-only gate / logging        )        dG = G_r - G_base        M = M_r        # safe output merge        G_out = G_base + alpha[r] * dG        v_r = suffix(G_out, prompt_kv0)        velocities.append(v_r)        loop_logs.append(stats)    return velocities[-1], velocities, loop_logs

4.3 三种必须同时支持的 loop state 模式
模式 | Persistent state | 目的
A. Memory-only | M_r；GEN 不作为 persistent correction | 复现当前方案，作为安全基线。
B. GEN-only | ΔG_r；K=0 | 直接回答 distributed GEN recurrence 是否足以产生稳定 semantic gain。
C. GEN+Memory | ΔG_r + M_r | 推荐主方案；验证 memory 是否提供额外 global workspace。

这三个模式共享相同 prompt anchor、body、suffix、sampler、seed 与 loop layers。只有 persistent recurrent state 不同。这样才能回答 memory 的“必要性”，而不是比较不同系统。
4.4 不推荐的 Direct Native Recycle（仅作为负对照）
G_{r+1} = B(G_r; P_KV^0)
可以保留为对照组，但不作为主路线。若它质量下降而 Anchored GEN Loop 保持质量，说明 re-entry anchor 确实解决了 post-training depth mismatch。若两者都稳定，则后续可考虑简化架构，逐步向真正 native recurrence 迁移。
5. 实现模块与代码改动边界
下面给出模块级实现规范。文件名以当前 BAGEL/LatentCoT 代码组织为参考；如果现有实现仍集中在 `qwen2_navit.forward_inference`，建议先把 loop 逻辑从 forward 中拆成独立 helper，避免后续训练与推理两套代码漂移。
模块 | 建议接口 | 输入 | 输出/状态 | 是否可训练
LoopConfig | dataclass / argparse config | R, s/e, K, mode, α, gate, logging flags | 统一配置 | 否
AnchorState | build_loop_anchor(...) | prompt, x_t, branch | C0, G0, prompt_kv0 | 否
BaseBodyPass | run_base_body(...) | G0, prompt_kv0 | G_base | 沿用 backbone
ReentryAdapter | A_phi(dG, M, t, r) | dG, M, timestep/loop id | delta_entry | 是
MemoryWorkspace | init/update_memory(...) | C0/G0/M/body states | M_r | 可选训练
LoopGate | apply_loop_gate(...) | body update + state | gated update | 是
LoopBodyRunner | run_loop_iteration(...) | G_entry, M, anchors | G_r, M_r, stats | 共享 backbone
OutputMerger | merge_with_base(...) | G_base, G_r, α_r | G_out | α 可训练
LoopReadout | decode_loop_state(...) | G_out, prompt_kv0 | v_r | 共享 suffix
LoopLogger | record_loop_stats(...) | states/updates/attn summary | per-step records | 否

5.1 建议新增配置项
enable_t2i_loop: bool = Falseloop_mode: str = "gen_memory_anchored"   # memory_only | gen_only | gen_memory_anchored | direct_nativeloop_start_layer: int = ...loop_end_layer: int = ...loop_depth: int = 2memory_slots: int = 8# anchors / recurrencefreeze_prompt_kv_in_loop: bool = Trueanchor_gen_reentry: bool = Truereentry_adapter_type: str = "low_rank"reentry_rank: int = ...# write controlloop_gate_type: str = "scalar_per_layer" # later: token/headloop_gate_init: float = 0.02loop_gate_base_pass: bool = False# output mergeloop_output_alpha_mode: str = "per_loop"loop_output_alpha_init: float = 0.0# deep supervisionloop_deep_supervision: bool = Falseloop_ds_weight: float = 0.5loop_ds_scheme: str = "final_plus_mean"# logginglog_loop_velocity_delta: bool = Truelog_loop_write_ratio: bool = Truelog_loop_state_norms: bool = Truelog_loop_attn_summary: bool = False

5.2 推荐开发顺序
先重构现有 memory-loop runner：把 anchor extraction、body iteration、suffix readout、CFG branch state 明确分开，不改行为。
新增 GEN-only persistent correction 与 K=0 模式；此时 ReentryAdapter 可先用固定/identity-like 小残差映射，仅做 topology 验证。
新增 Base residual merge，确保 α=0 时等价 baseline。
新增 loop-only scalar gate 与 write-ratio logging。
完成 A/B/C 三模式 training-free 对照后，再把 ReentryAdapter、gate、α 设为可训练。
最后才考虑 LoRA / 小学习率解冻 body；不要在 topology 未验证时直接训 backbone。
5.3 必须通过的正确性测试
测试 | 检查项 | 通过标准
Base equivalence | loop disabled / R=0 / α=0 | velocity 数值与 baseline 一致；图像逐像素一致或仅有可解释浮点误差。
Prompt anchor | 每轮 prompt KV hash/norm | extra loop 前后不发生 persistent 改写。
x_t anchor | 每轮输入 sampler state | inner loop 内完全相同。
CFG isolation | 三分支 state ids / tensors | 分支之间不共享 M、ΔG 或 gate hidden。
Depth consistency | 所有 recurrent state 都来自同一 body endpoint | 禁止把不同 layer depth 的 hidden 直接做 recurrent delta。
K=0 path | GEN-only 可完整运行 | 不依赖 dummy memory token。
Gradient scope | 训练阶段 requires_grad map | 第一阶段只有 adapter/gate/α/memory params 可训练。

6. Training-free 推理验证：先证明 topology，再训练
第一阶段验证的目标不是追求最终 benchmark SOTA，而是回答两个因果问题：① distributed GEN recurrence 是否比 memory-only 更接近稳定 semantic refinement；② memory 在 GEN recurrence 已存在时是否仍提供额外价值。所有比较必须使用同一 checkpoint、同一 prompt、同一 seed、同一 sampler trajectory 与同一 body layers。
6.1 最小四组主对照
组别 | Prompt KV | GEN recurrent | Memory recurrent | Anchor | 用途
Base | frozen | × | × | — | 原始质量与语义基线
Current MemLoop | frozen | × | ✓ | ✓ | 当前唯一已观察到语义编辑且质量稳定的方案
Anchored GEN-only | frozen | ✓ | × | ✓ | 检验 visual state recurrence 的必要性
Anchored GEN+Mem | frozen | ✓ | ✓ | ✓ | 推荐主方案；检验 memory 的增益是否超出 topology effect

6.2 可选负对照
Direct Native Recycle：G_{r+1}=B(G_r)，验证 depth mismatch 是否导致质量/表示漂移。
Dynamic Prompt KV：只在小规模诊断中保留，作为“直接动 condition manifold”的高风险对照，不再作为主路线。
Zero / shuffled memory：仅在 GEN+Memory 内做，判断 memory semantic content 是否真正影响 final correction。
Frozen memory：M 固定但保留 token topology，分离“memory 内容”与“额外 token routing”效应。
6.3 核心指标：只保留能解释机制的少数指标
指标 | 定义 | 回答的问题
Semantic task score | GenEval2-hard / TIIF spatial / 自建结构 prompt 子集 | 是否真的改善数量、空间、属性、关系，而不只是纹理变化？
Base quality retention | Aesthetic/quality proxy + 人工 pairwise + invalid rate | 语义 gain 是否以破坏原生图像质量为代价？
Velocity delta ratio | ρ_v(r,t)=||v_r−v_base|| / ||v_base|| | loop 实际上改变了多少生成动力学？
Velocity direction consistency | cos(Δv_r, Δv_{r+1}) 或与 oracle/teacher correction 的 cosine | correction 是否形成稳定方向，而不是每轮乱跳？
GEN write ratio | ρ_G(r,l,t)=||ΔG_update||/||G_anchor|| | repeated write 是否逐轮收敛，还是持续强扰动？
Loop gain curve | score(R=0,1,2,3,4) | 是否存在“更多内部计算→更好”的规律，而不是只在某一固定 R 偶然有效？

6.4 推荐的 timestep 分析
不要只做整条轨迹平均。至少将 denoising trajectory 分为 early / middle / late 三段，分别报告 semantic score proxy、ρ_v 与 ρ_G。原因是数量/全局布局通常更依赖早中期，而后期更容易表现为纹理/细节变化；如果 loop 只在 late timesteps 产生高 ρ_v，却不改善结构任务，说明它更像 appearance perturbation 而不是 semantic refinement。
6.5 Training-free 阶段的进入训练门槛
建议满足以下条件再进入 post-training：
Anchored GEN-only 或 GEN+Mem 至少在一个结构性 benchmark 上显著优于 Base，同时 quality retention 与 Current MemLoop 同级。
R=1→2→3 至少在部分 prompt 上呈现可解释的 progressive correction，而不是只有随机纹理变化。
ρ_G 不随 loop 深度持续增大；理想情况下在后续 loop 有衰减趋势。
Direct Native Recycle 若明显更差，则进一步确认 anchored re-entry 的必要性。
GEN+Mem 相比 GEN-only 的 gain 必须可重复，否则 memory 暂不作为训练主对象。
7. Post-training 方案
训练目标不是让 BAGEL “重新学一次 T2I”，而是让新增 recurrent operator 在不破坏原模型的情况下学会产生小而有用的 visual correction。训练按参数侵入程度分三阶段，只有前一阶段成立后才进入下一阶段。
7.1 Stage 1：只训练新增模块（推荐首轮）
参数组 | 状态 | 初始化 | 学习率关系
Backbone F_{1:L} | freeze | 原 checkpoint | 0
Prompt/UND condition path | freeze | 原 checkpoint | 0
ReentryAdapter A_φ | train | 最后一层 zero-init | 1.0× base LR
Loop-only gate | train | near-zero output | 1.0× base LR
Output α_r | train | 0 | 0.5~1.0× base LR
Memory init/projection | train if K>0 | 沿用当前可行初始化或 small init | 1.0× base LR

这样训练初始点严格接近原模型：A_φ≈0、α≈0、loop gate≈0。模型需要通过梯度“主动打开” recurrent correction，而不是训练一开始就承担大分布迁移。
7.2 Direct Flow Deep Supervision
借鉴 Looped-DiT 的核心思想，但不做 self-distillation：每个 recurrent depth 经过同一个原始 suffix 得到 v_r，并直接对同一个 flow-matching target v* 监督。
L_r = ||v_r − v*||_2^2
L_total = L_R + λ_DS · (1/(R−1)) Σ_{r=1}^{R−1} L_r
推荐第一版 λ_DS=0.25~0.5 作为起点；主目标仍是 final loop，intermediate supervision 负责保证 trajectory 中每一轮都有 functional meaning。不要一开始使用 L_distill=||v_r−sg(v_R)||²，因为这会鼓励各 loop 尽快相同，反而削弱我们要研究的 recurrent refinement。
7.3 是否加入“收敛”正则
第一轮训练不建议加入强制 monotonic / margin loss。可以记录而不优化以下量：||ΔG_r||、||v_r−v_base||、cos(Δv_r,Δv_{r+1})。只有观察到明显 exploding/oscillation 后，才考虑非常轻的 update penalty，例如对 extra-loop gate 或 adapter output 做 norm regularization，而不是人为规定“每轮必须更好”。
7.4 Stage 2：让 Memory 成为真正的 UMM workspace
Stage 1 先回答 GEN recurrence 能否工作；若 GEN+Mem 明显优于 GEN-only，再强化 M 的功能。此时目标不是让 M 拟合某个手工 semantic label，而是让它作为 UND↔GEN 的共享 workspace 提供可被生成侧利用的 global correction。
可训练模块包括：memory query/init、UND-side memory projection、GEN-side memory read projection、memory-to-reentry conditioning。仍保持 prompt KV frozen。
建议重点用简单结构数据：object/count/spatial/action/attribute binding，避免首轮混入文字渲染、风格、文化身份与复杂替换，因为这些任务会让 memory 学到的“reflection”与视觉结构 correction 混杂。
7.5 Stage 3：小学习率适配 loop body
只有 Stage 1/2 已经证明 stable loop-depth gain 后，再允许 shared body B 适应 recurrent usage。优先 LoRA 或 selective unfreeze，而不是全量 backbone fine-tune。
lr_body ≈ 0.05~0.1 × lr_adapter
推荐只对 loop body 的 GEN-related attention/output projection 或少量层做 LoRA；UND/prompt anchor 继续冻结。训练时同时保留 R=0 / base reconstruction batch 或显式 base-path consistency 检查，防止 body 更新后破坏 no-loop 能力。

7.6 训练数据构成
数据桶 | 优先级 | 用途
普通 T2I semantic prompts | 高 | 保持常规生成能力，避免 loop 只对 hard prompts 过拟合。
结构 hard prompts：count/spatial/relation/action/attribute | 最高 | 直接训练 loop 的 visual constraint refinement。
No-op / easy prompts | 中高 | 训练 gate 学会“不需要改时少写”，保护 base quality。
复杂文字/风格/文化实体 | 低（首轮排除） | 避免把非结构性难题混进 early mechanism validation。

7.7 Loop depth curriculum
建议不要一开始固定只训 R=4。更安全的做法是从 R∈{1,2} 开始，确认模型能利用 extra step 后再加入 R=3/4；或在 batch 中随机采样 R∈{1,2,3}。这样可以减少“第 r 轮被隐式绑定为固定功能层”的风险，并更接近真正 elastic recurrent operator。
8. 训练效果验证与对比
训练完成后不能只看 final benchmark。必须同时验证：base prior 是否保留、loop depth 是否可调、每一轮是否产生 functional refinement、memory 是否真的提供因果增益。
8.1 主对比矩阵
模型/方案 | 训练 | GEN recurrence | Memory | Prompt KV 动态 | 用途
Base BAGEL | 原始 | × | × | × | 原生能力基线
Training-free MemLoop | 无 | × | ✓ | × | 当前安全 semantic-edit 基线
Training-free Anchored GEN | 无 | ✓ | 0 | × | topology 作用
Stage1 GEN-only | adapter/gate/α | ✓ | 0 | × | 验证可训练 visual refiner
Stage1 GEN+Mem | adapter/gate/α+M | ✓ | ✓ | × | 推荐主模型
Direct Native Recycle | 可选 | ✓ | 可选 | × | 验证 anchor/re-entry 必要性
Dynamic Prompt KV | 对照 | 可选 | 可选 | ✓ | 高风险 condition rewrite 对照

8.2 三类结果必须同时成立
维度 | 希望看到的结果 | 失败意味着什么
Semantic gain | hard structural tasks 提升，尤其 count/spatial/relation；定性出现缺物体补全、关系纠正等 progressive correction | 如果只有纹理变化，loop 仍未形成 semantic refinement operator。
Quality retention | easy/general prompts 与 base 基本持平；invalid / artifact 不上升 | 如果 hard score 提升但质量下降，说明 correction 侵入过强或 write 未受控。
Elastic compute | R=0 保留 base，R=1→2→3 在 hard set 上总体不下降且存在边际增益 | 若仅固定 R 有效，则更像特殊深度适配，不是可扩展 loop。

8.3 Memory 因果性验证
在 Stage1 GEN+Mem 上做以下四组，其他完全相同：correct M、shuffled-across-sample M、zero M、frozen M。真正证明“memory 内容有用”的标准不是这些组都能产生变化，而是 correct M 在 semantic score / velocity alignment / final quality 上稳定优于 topology-preserving controls。
若 correct≈shuffled≈frozen，则应接受结论：memory 主要是 routing/workspace topology，而不是 semantic carrier。此时可以继续保留 memory 作为安全低秩通道，但文档叙事应从“memory reasoning”改成“auxiliary recurrent workspace”。
8.4 Loop-depth trajectory 验证
对每个 prompt 保存 r=0,1,2,3 的中间图与 velocity。重点统计：
任务 score 随 r 的变化；
ρ_v(r,t) 与 ρ_G(r,l,t)；
当前轮相对上一轮修复了多少 initially-wrong constraints（repair）；
当前轮破坏了多少 initially-correct constraints（damage）；
同一 prompt 的 correction 是否在后续 loop 变小而不是持续扩大。
可借鉴“Repair / Damage”思想：如果 loop 真的是 refinement，理想情况不是单纯 final score 高，而是 Repair 明显高于 Damage，并且随着 r 增大 update magnitude 逐渐衰减。
8.5 推荐的最小验收表
实验 | 必须报告
Base vs Stage1 GEN-only vs Stage1 GEN+Mem | GenEval2-hard / TIIF-Spatial 或同类结构集；通用质量；平均 inference FLOPs。
Loop depth R=0..4 | semantic score、quality、ρ_v、ρ_G。
Memory correct/shuffled/zero/frozen | semantic score + final image pairwise + velocity delta。
Direct recycle vs anchored re-entry | quality retention + write ratio + loop-depth stability。
Timestep bins | early/mid/late 的 ρ_v、semantic change rate。

9. 风险、失败模式与对应处理
风险 | 可观察信号 | 处理
GEN recurrence 破坏 pretrained manifold | easy prompts 质量下降；ρ_G 持续变大 | 降低 α / gate；缩小 adapter rank；只在更后层 loop；保持 body frozen。
loop 只产生纹理变化 | ρ_v 高但结构 score 不变 | 把 loop 窗口前移或增加 GEN persistent capacity；检查 prompt/GEN attention routing。
memory 仍是 topology-only | correct≈shuffled/frozen | 接受 workspace 叙事；不再强训 semantic memory；重点训练 GEN correction。
early loop 有效、late loop 变坏 | Repair↓ Damage↑，write ratio 不衰减 | 学习 loop gate / early exit；不要强制固定最大 R。
adapter 学成大幅重编码器 | δ_entry norm 过高 | norm penalty / rank 降低 / α 上限；加强 no-op/easy 数据。
body 微调后 base 能力退化 | R=0 也下降 | 回退到 adapter-only；body 使用更小 LR/LoRA；加入 base consistency batch。

10. 推荐里程碑（按因果问题推进）
阶段 | 要回答的问题 | 实现范围 | Go / No-Go
M0: 重构 | 现有 loop 能否统一成可切换 runner？ | 无行为改动；拆 anchor/body/suffix/logger | Base equivalence 通过。
M1: Training-free GEN state | GEN recurrence 是否有价值？ | GEN-only + anchored re-entry + α 手动值 | 至少一个结构任务有稳定 semantic gain，质量不明显下降。
M2: GEN+Memory | memory 在 GEN recurrence 上是否有额外价值？ | K=0 vs K=8/16；correct/shuffled controls | correct GEN+Mem 若无额外 gain，不扩大 memory 训练。
M3: Stage1 post-train | 新增模块能否学出稳定 loop？ | 只训 adapter/gate/α/M | R=0 保持 base；R>0 hard set 提升；write 不爆。
M4: Stage2 workspace | UMM 的 UND↔GEN workspace 是否有效？ | 训练 memory read/write projections | correct M 对 controls 出现可重复优势。
M5: Body adaptation | 能否进一步接近 native loop？ | LoRA/小 LR 适配 B | loop-depth elasticity 提升且 base 不退化。

11. 最终推荐：UMM-T2ILoop 的理论核心应该是什么
UMM-T2ILoop 不应被定义为“一个更聪明的 memory writer”。更合适的定义是：
A pretrained-prior-preserving recurrent visual refinement operator
其核心状态是 distributed GEN representation；其稳定性来自 frozen anchors、anchored re-entry 与 loop-only write control；其 UMM 特性来自可选的 UND↔GEN shared workspace。Memory 在这个体系中不是理论必要条件，而是当前最有价值的工程脚手架：它提供安全的低带宽 intervention path，并可能在后续训练中发展为真正的 global reasoning workspace。
因此，下一版不建议继续把全部精力投入 memory-only，也不建议直接做 full native recycle。最合理的主线是先实现 Anchored GEN+Memory Loop，在严格 frozen prompt KV / fixed x_t / exact base path 的约束下验证 distributed recurrence；确认有效后，再通过 Direct Flow Deep Supervision 与小规模参数适配把它逐步训练成真正的 native-like recurrent operator。
Appendix A. 关键公式汇总
(C_0, G_0) = F_{1:s}(P, x_t)
G_base = B(G_0; P_KV^0)
ΔG_{r-1} = G_{r-1} − G_base
G̃_r = G_0 + A_φ(ΔG_{r-1}, M_{r-1})
(G_r, M_r) = B(G̃_r, M_{r-1}; P_KV^0)
G_out^r = G_base + α_r(G_r − G_base)
v_r = F_{e:L}(G_out^r; P_KV^0)
L_r = ||v_r − v*||²
L_total = L_R + λ_DS · Mean_{r<R}(L_r)
ρ_v(r,t) = ||v_r − v_base|| / ||v_base||
ρ_G(r,l,t) = ||ΔG_update(r,l,t)|| / ||G_anchor(l,t)||
Appendix B. 与 Looped-DiT 的对应关系
Looped-DiT | UMM-T2ILoop 推荐映射 | 关键差异
h^(r+1)=B(h^r) | G̃_r=G_0+A_φ(ΔG_{r-1},M_{r-1}) → B | 我们是 post-training，先用 anchor 解决 depth-distribution mismatch。
text + image hidden 一起 recurrent | prompt KV frozen；GEN + optional M recurrent | 优先保护 pretrained condition prior。
Deep Supervision on every loop | shared suffix 对每个 G_out^r 做 direct flow loss | 直接借鉴，且不先做 final-loop distillation。
Self-Modulating Attention / XSA | extra-loop zero-init write gate | 先采用更保守、可退化为原模型的控制。
从头/联合训练 recurrent operator | 分阶段：adapter/gate → memory workspace → body LoRA | 降低 post-training 破坏风险。

References
[1] Chng et al., “Looped Diffusion Transformer,” arXiv:2609.40305v1, 2026. https://arxiv.org/html/2609.40305v1
[2] OpenSenseNova, Looped-DiT codebase. https://github.com/OpenSenseNova/Looped-DiT