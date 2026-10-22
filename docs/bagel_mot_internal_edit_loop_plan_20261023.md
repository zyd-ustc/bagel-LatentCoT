# BAGEL MoT：单条文生图轨迹内的连续理解与编辑

日期：2026-10-23。状态：已按用户后续授权实现新分支，完整 NPU 前后向 pilot 已通过；8000 步运行状态见[RUN.md](mot_edit_8000_20261023/RUN.md)。

## 1. 决定与研究问题

**保留原生 BAGEL 文生图，在去噪中途加入“UND 看当前预测 → 连续语义反馈 → GEN 编辑预测”的两轮内循环。UND 输出连续状态，GEN 复用已训练的 `llm2vae` 输出速度。整个系统沿用原生输出头。**

研究问题是：统一多模态模型能否把完整图像上的“看图、指出问题、编辑”压缩到一次文生图的内部计算，并由 UND 的状态理解带来语义改善？

固定以下任务条件：

- 用户输入仍为原始 prompt `P` 和 seed。
- 原生 GEN 始终保留完整 `P`，负责基础去噪。
- 编辑分支观察当前预测图像的 latent；其目标信息经过 UND 的连续反馈进入。
- UND 读取 GEN hidden。训练和推理的 reader 都采用这条入口，推理不引入 ViT 或外部图像。
- 每个内循环固定外层 `x_t,t`。完成编辑后，才推进一次外层去噪。
- 内循环传递 latent 和连续 token；最终才做一次 VAE 图像解码。
- 不新增 caption head、velocity head、hidden-to-hidden 输出投影或零初始化 residual head。已有 `lm_head` 服务于读取能力训练，已有 `llm2vae` 服务于图像预测。

**这次保留连续 KV 作为反馈载体。历史已经用过这一载体；新的核心变化是编辑对象、分支职责和循环闭合方式。** 原来的 GEN 同时读完整目标和可选 Memory；本方案让基础 GEN 继续正常生图，独立的逻辑 GEN 编辑器读取预测图像和 UND 反馈，再由原生生成接口输出。

## 2. 历史经验如何进入本方案

| 已有证据 | 对本方案的具体影响 |
|---|---|
| C2-observe 的 GEN-hidden caption 在同 prompt 来源交换中，12/12 条描述随来源变化；人工观察指出 `t>0.75` 后描述模板化 | 继承“目标无关的观察”训练方式；首轮反馈放在中等噪声节点。reader 重新对齐本方案实际使用的 clean-proposal GEN 特征 |
| 早期 GEN 即使只拿到 `An image.`，G-TC 与 G-swap-TC 仍很接近；训练 reader/consumer response 后，局部预测才出现明显响应 | 去掉编辑器的原始 prompt 捷径只解决输入职责。训练仍需把 UND 的状态表达与 GEN 的读取接口连起来 |
| 累计 1100 步 response 训练的局部预测中，用户观察到 G-TC 接近 G-text；实验没有完成整条采样轨迹 | 保留 native attention 与逐层反馈接口经验；最终判断用完整采样图像 |
| Stage A 曾同时把外层 `γ`、memory gate 和 reentry gate 置零；零点 cosine 梯度又过大 | 新方案无这组串联门控，无 cosine loss。训练直接通过原生 GEN 输出计算 MSE |
| Step4000 Consumer 隔离：Memory 与回灌关闭后，Consumer ON 仍有异常纹理，OFF 在已检查案例中恢复画风 | 基础 GEN 参数与缓存保持独立；编辑训练只更新专用适配器。原生采样不经过编辑适配器 |
| 照片 5000 步训练中 loss 下降，同时出现对象消失与彩色重复纹理 | 数据质量、回归误差和语义编辑分别记录。当前方案先修复能力接口，不把换数据当作接口问题的替代解释 |
| guidance 干预发现纠错系统性抵消 `v_g-v_c`；去掉反向分量恢复对象和质感 | 保留当前 guidance-safe 的目标及部署约定；新 GEN 输出先换算为条件残差，再进入同一投影 |
| guidance-safe Step2000 六榜未形成整体收益 | 保留该 checkpoint 为历史结果，新的实验从原生生成权重与明确的 reader 初始化开始 |
| Correction Training 在预测空间重复编辑，并继承生成器的预测接口 | 循环对象采用明确的预测 latent，输出复用 GEN 原生速度头；训练包含实际要执行的两轮编辑 |

六榜的完整同分母比较为：GenEval，base 84.47% / corrected 83.62%；GenEval++，42.50% / 40.00%；DPG，71.5158 / 71.4672；WISE，32.00% / 30.80%。这些是原 27B 内部评分协议下的结果。CoRe、CompBench++ 两轮有效题集不同，完整记录保留各自分母。

历史依据见[研究记录](research_record_20261022.md)、[Consumer 隔离](loop_ablation/QUICK_RESULTS.md)、[guidance 方向实验](overnight_20261022/GUIDANCE_PROBE.md)。历史 caption 见 Git `e539415`、`2c1c05f`；Memory response 见 [`08b7cce` 的实验记录](https://github.com/zyd-ustc/bagel-LatentCoT/blob/08b7cce/docs/experiments.md)。

## 3. 总体流程：基础生成、观察、理解、编辑

```text
用户 prompt P + seed
          │
          ▼
原生 BAGEL：D(x_t,t,P) 与 D(x_t,t,∅)
          │ 原生 CFG
          ▼
当前预测 X⁰ = x_t - t·v_g
          │
          │  在选定时间点，固定 x_t,t，执行两轮：
          │
          ├─ ① 冻结 GEN 观察前向：编码当前 Xʳ，提取逐层图像 KV
          │       │
          │       ▼
          │  ② UND 观察：读取实际图像状态，得到 Oʳ（目标无关）
          │       │
          │       ├──────── 原始 P 的 UND KV
          │       ▼
          │  ③ UND 反馈：比较 Oʳ 与 P，得到连续反馈 Fʳ
          │       │
          │       ▼
          │  ④ GEN 编辑：E(Xʳ,t,Fʳ) → 原生 llm2vae → wʳ⁺¹
          │       │
          │  条件残差换算 + guidance-safe
          │       ▼
          └──── 新预测 Xʳ⁺¹；下一轮重新观察这个实际应用后的预测
          │
          ▼
使用第二轮速度推进 x_t → x_t′
          │
          ▼
继续同一条去噪轨迹 → 最终 VAE decode
```

这里的观察前向是模型内部计算，输入是已有 latent，直接绕过 VAE encode。它与编辑前向分别执行，目的是让下一轮 UND 真正看到已应用修正后的图像状态。

### 3.1 五个角色，共享原生能力，分开适配器与缓存

| 角色 | 输入 | 输出 | 参数状态 |
|---|---|---|---|
| 基础生成器 `D` | `x_t,t,P` 或空条件 | 原生条件/无条件速度 | 原生权重冻结 |
| 图像观察前向 `V` | 当前预测 `X`，固定中性文字，图像时间 `0` | 28 层图像 KV | 原生 GEN 冻结 |
| UND 观察器 `U_obs` | 图像 KV，16 个中性观察 slot | 逐层观察状态 `O` | 先训练原生 UND 投影的适配器，正式编辑阶段冻结 |
| UND 反馈器 `U_fb` | `O`、完整 `P`、16 个中性反馈 slot | 逐层连续反馈 `F` | 独立 UND 适配器可训练 |
| GEN 编辑器 `E` | `X,t,F` | 完整条件参考速度 `w` | 独立 GEN 适配器可训练；原生输出头冻结 |

“独立 GEN”指独立计算角色和可训练参数。首版共享只读的原生 GEN 权重，通过私有 LoRA 实现编辑参数分离；基础生成器始终关闭这些 LoRA。这样保留生成预训练能力，也保留原生对照的身份。

UND 两个角色同样共享只读基础权重，使用各自适配器。适配器明确属于角色，禁止用进程全局替换参数的方式切换模式。

### 3.2 图像观察前向为什么取时间 0

`X` 表示预测的干净 latent。观察前向复用 BAGEL 原生 clean-VAE-context 的入口语义：

\[
h_{\rm image}=\operatorname{vae2llm}(X)+e_{\rm time}(0)+e_{\rm 2D}.
\]

图像 token 仍走 GEN 专家，提取它们进入各层 attention 的原生 K/V。固定中性文字采用 `An image.`，来源中排除其文字 KV。观察器因此读取由图像 latent 产生的 GEN 状态。

时间 0 表示输入的类型是预测图像，并不表示这张预测已经正确。实际外层噪声时间 `t` 仍交给编辑器的原生时间入口。高噪声预测本身缺少可辨认内容的问题，通过反馈节点选择处理。

历史 C2 reader 读过不同噪声时间下的 GEN 特征；本方案改成 clean-proposal 观察前向。因此在这个确切入口上先做 caption 对齐，保留历史证明过的训练思路，不把不同入口视为相同分布。

## 4. 利用 MoT 的原生接口，不增加输出头

### 4.1 相同维度提供接口，专家参数负责各自分布

本项目 BAGEL 为 28 层、hidden size 3584。UND 与 GEN 具有匹配的 attention 张量布局，但各自使用原生的归一化、Q/K/V/O 和 MLP。

在第 `l` 层：

- 连续反馈 slot 使用 UND 的 `input_layernorm`、`k_proj`、`v_proj`、`k_norm`。
- 图像编辑 token 使用 GEN 的 `input_layernorm_moe_gen`、`q_proj_moe_gen`、`q_norm_moe_gen`。
- 图像 attention 输出继续走 `o_proj_moe_gen` 与 GEN MLP。
- 最终图像 hidden 经过 `norm_moe_gen` 和已加载的 `llm2vae`。

令 `F_l` 为反馈 token **进入第 l 层之前**的状态，`H_l` 为编辑图像 token 的对应状态：

\[
K^F_l=\operatorname{RoPE}(\operatorname{kNorm}_{U,l}(W^K_{U,l}\operatorname{LN}_{U,l}(F_l))),
\]
\[
V^F_l=W^V_{U,l}\operatorname{LN}_{U,l}(F_l),\qquad
Q^E_l=\operatorname{RoPE}(\operatorname{qNorm}_{G,l}(W^Q_{G,l}\operatorname{LN}_{G,l}(H_l))).
\]

编辑 attention 在一次原生 softmax 中读取反馈与图像自身：

\[
A_l=\operatorname{Attention}\bigl(Q^E_l,[K^F_l;K^E_l],[V^F_l;V^E_l]\bigr).
\]

这沿用 MoT 中“GEN query 读取 UND key/value”的跨专家计算方式。新增学习集中在既有投影的 LoRA 上。相同维度省去桥接输出头；反馈 token 与自然文本 token 的分布差别由训练对齐。

### 4.2 反馈通过原生条件 attention 进入编辑器

基础 GEN 的原始 prompt 条件保持原样。编辑器是附加的独立角色，它的条件区由 `F` 构成，允许固定格式 token，不携带逐样本原始 prompt KV。

编辑器还会从 `X` 看见已经画出来的目标内容。这是编辑任务需要保留的输入。隔离的是“再读一遍完整 P，绕过当前观察直接重画”的条件通道。

连续反馈有原生 attention 的逐层计算，但没有额外的逐层 residual 分支、独立 Memory softmax、反馈缩放门或入口 hidden 叠加。原生 Transformer 自身的 residual 结构保持不变。

### 4.3 UND 先观察，再结合目标形成反馈

`U_obs` 的 16 个固定中性 slot 读取 GEN 图像来源。它们没有逐样本目标文字输入。`U_fb` 的另外 16 个 slot 读取观察状态和目标文字，训练其形成可供编辑使用的连续反馈。

因此，文字目标与图像证据的汇合位置明确位于 UND。观察状态、目标文字和反馈状态的缓存分别存储。编辑器仅接收反馈 slot 的 KV。

这个结构给出清楚的信息路径；状态是否实际改变编辑行为，仍由同目标、不同图像状态下的反馈交换结果回答。

### 4.4 层与位置约定

1. 全部 bank 保存 28 层、逐样本、未旋转的 K/V。记录 token 角色和有效长度。
2. 消费时，在接收序列的原生位置规则下，对 key 应用一次 RoPE；value 保持不变。
3. 图像 query 按 BAGEL 现有 `packed_query_position_ids` 生成 RoPE，2D 空间位置使用现有 `latent_pos_embed`。图像 token 的一维位置规则沿用原生实现，不另造逐 patch 的一维递增规则。
4. 基础生成器的位置与原生代码完全一致。观察器、反馈器和编辑器分别持有自己的序列布局与缓存。
5. 使用同深度的 layer-input KV。第 28 层 UND 输出不回灌第 1 层 GEN；每轮都从新预测 latent 的原生入口开始。

历史 `MemoryWriter` 在同层 block **之后**重新投影导出 KV，其 block residual adapter 和 time-conditioned query adapter 也与本方案的原生投影 LoRA 不同。本方案改用原生 prefill 的 layer-input 语义，从原生 UND 权重建立新 reader，并在阶段 A 对齐这个确切入口。旧 C2 权重保留作历史读取能力的参考，不把它的 adapter 张量直接改名加载进新模块。

### 4.5 输出直接沿用生成器

512px 下，VAE latent 经 `2×2` patch 化形成 1024 个 token，每个输出 64 个 latent 标量：

\[
w=\operatorname{llm2vae}\bigl(\operatorname{norm}_{G}(H_{28})\bigr).
\]

`llm2vae` 使用发布权重并冻结，整个方案没有重新初始化它。编辑输入从 noisy latent 改为预测 latent，这项适配由 GEN 内部 LoRA 完成。

编辑入口明确为 `vae2llm(X) + time_embedder(t) + latent_pos_embed`，使用实际外层时间 `t`。观察入口的时间为 0，编辑入口的时间为 `t`，二者分别描述“这张预测是什么”与“此时应怎样更新”。

## 5. 循环与 guidance-safe 的完整数学定义

### 5.1 时间与固定参考

BAGEL 使用 `t=1` 表示噪声、`t=0` 表示干净图像。训练图文对为 `(P,x_0)`，噪声为 `ε`：

\[
x_t=(1-t)x_0+t\epsilon,\qquad v^*=\epsilon-x_0.
\]

每个外层步先运行原生条件/无条件前向，得到 `v_c`、`v_u`，再按当前 BAGEL 的 CFG 与 rescale 实现得到 `v_g`。固定：

\[
g=v_g-v_c,\qquad X^0=x_t-tv_g.
\]

一个内循环期间，`x_t,t,v_c,v_u,v_g,g` 均不重新定义。尤其不把上一轮的修正速度改称新的 `v_c`。

### 5.2 投影含义

对每张图独立计算：

\[
\mathcal H_g(d)=d-\min\left(\frac{\langle d,g\rangle}{\max(\langle g,g\rangle,10^{-12})},0\right)g.
\]

投影保留与 guidance 正交的部分和同向部分，移除反向平行部分。`g=0` 时结果为 `d`。点积与平方和在 FP32 中逐图计算，避免一个 batch 的样本共享投影系数。

这里保留的是 guidance 的方向；正向过强、正交方向画错对象、纹理失真仍是实际图像要回答的问题。

### 5.3 每一轮计算

对于 `r=0,1`：

\[
S^r=V(X^r,0),\quad O^r=U_{obs}(S^r),\quad F^r=U_{fb}(O^r,P),
\]
\[
w^{r+1}=E(X^r,t,F^r),
\]
\[
d^{r+1}=w^{r+1}-v_c,\quad
\Delta^{r+1}=\mathcal H_g(d^{r+1}),
\]
\[
\widetilde v^{r+1}=v_g+\Delta^{r+1},\qquad
X^{r+1}=x_t-t\widetilde v^{r+1}.
\]

`w` 是原生头输出的完整速度量，`d` 才是用来部署的条件残差。第二轮重新观察 `X¹`，它已经包含实际应用的 guidance-safe 结果。

每轮输出都相对于固定参考 `v_c` 解释。两轮修正不会被相加为 `v_g+Δ¹+Δ²`。第二轮接收第一轮预测并给出新的完整编辑结果。

外层只更新一次：

\[
x_{t'}=x_t-(t-t')\widetilde v^2.
\]

实现时复用当前 sampler 对原生 BF16 基础更新与 FP32 修正更新的分离方式，避免顺手改变原生舍入路径。

### 5.4 为什么增加观察前向

编辑器本轮 hidden 对应其输入 `Xʳ` 与内部计算。最终部署的 `Xʳ⁺¹` 还经过速度换算及 guidance 投影。把原来的 hidden 原样交给下一轮，会让观察内容与实际预测错位。

首版明确重新编码实际 `Xʳ⁺¹`。这样“看修改后的图，再给下一轮反馈”在数据流上闭合。以后优化这次前向时，保持这个接口含义。

## 6. 训练：先接回理解能力，再训练两轮编辑

### 6.1 阶段 A：用原生语言输出建立观察能力

训练入口与正式推理一致：干净训练 latent `x₀` → 冻结 GEN clean-context 前向 → `U_obs` → 冻结原生 UND 语言解码器及 `lm_head` → caption CE。

- 来源前向只使用固定中性文字。目标 caption 仅出现在语言解码的监督位置，按原生因果 mask 训练。
- 更新 `U_obs` 的原生 UND Q/K/V/O LoRA；冻结 GEN、VAE、基础 UND、归一化和所有原生输出头。
- 沿用 16 个观察 slot；slot 初值来自固定中性 token 的原生 embedding，不新增可训练输出查询头。
- 新 reader 从原生 UND 初始化。历史 C2-observe 的原路径为 `/root/outputs/visual_observe_full_20261012`；其模块接口与本方案不同，历史结果用于选择训练方式和解释来源读取现象。
- reader 在真实图像 latent 上学习描述。生成 proposal 的语义读取直接查看其预测内容与描述，保留它相对真实图像的输入分布差别。

使用现有失败样例中的 4 个 prompt、每个 2 个 seed，读取实际中间 proposal。用已有语言头解码一次观察描述，检查是否随图像内容变化。这里要看到的是图中已有对象、数量和关系，而非复述原始目标。描述解码用于本阶段检查，正式采样去掉这段文字生成。

### 6.2 阶段 B：两轮展开，训练连续反馈和 GEN 编辑

冻结完成对齐的 `U_obs`。更新 `U_fb` 的 UND Q/K/V/O LoRA 与 `E` 的 GEN Q/K/V/O LoRA，其他参数冻结。首版不增加 MLP adapter、bridge 或输出参数。

采用同一个明确的训练目标：

\[
d^*=\mathcal H_g(v^*-v_c),\qquad w^*=v_c+d^*,
\]
\[
\mathcal L=\frac{1}{2}\sum_{r=1}^{2}\operatorname{MSE}(w^r,w^*).
\]

该目标与现有 guidance-safe 的条件残差目标相同，只把新零头输出残差，改为原生生成头输出完整参考速度。比较基准仍为 `zero_correction_mse=MSE(0,d*)`。物理误差 `MSE(v_g+Δ,v*)` 单独记录。

两轮使用同一组反馈/编辑适配器。每轮的 proposal 输入停止梯度；第 1 轮输出仍由其自身 loss 训练，第 2 轮从第 1 轮实际产生的 proposal 开始。这样训练覆盖实际的重复应用，同时避免跨轮反向图累积。

每个训练样本执行如下顺序：先由真实图文对构造 `x_t`，计算冻结的 `v_c/v_g/w*`；完成第 1 轮后对半权重 loss 反向传播；保存 detach 的 `X¹` 并释放本轮计算图；完成第 2 轮并对另一半 loss 反向传播；最后统一做一次梯度裁剪和 optimizer step。两轮之间参数保持同一版本。

梯度路径为：

```text
本轮 MSE → 冻结 llm2vae 的输入梯度 → GEN 编辑 LoRA
                                 → 反馈 KV → UND 反馈 LoRA
```

冻结原生输出头只冻结其参数，编辑前向仍保留 autograd。`U_fb` 与 `E` 的本轮前向均不放入 `no_grad`。

阶段 B 只使用上述 MSE，caption CE 在阶段 A 完成。没有 cosine、shuffled margin 或强制逐轮改善的附加损失。目标相同不等于每次编辑自然改善；第二轮结果与第一轮的实际语义差别是该方案的直接研究结果。

### 6.3 首版训练配置与执行边界

以下默认值已落入 `configs/mot_edit_8000.json`。用户后续授权新分支实现和 64 卡 8000 步，运行配方同步如下。

| 项目 | 首版选择 |
|---|---|
| 底座 | 原生 `Bagel-7B-MoT`；不加载旧速度纠错头或旧 Consumer LoRA |
| 图像尺寸 | 512px |
| reader 来源 | 原生 GEN clean-context，图像时间 0 |
| slot | 观察 16，反馈 16 |
| 新适配器 | 原生 Q/K/V/O LoRA，rank 32；A 随机、B 置零的标准 LoRA 初始化 |
| 优化器 | AdamW，lr `1e-5`，betas `(0.9,0.999)`，weight decay `0`，clip norm `1` |
| 精度 | 基础计算 BF16；loss、投影和修正合成 FP32 |
| 正式编辑训练轮数 | `R_train=2`，与首版推理一致 |
| 时间采样 | 从下一节三个实际采样节点均匀抽取 |
| 数据 | 用户指定 `/data/250010019/unimoe_r3/qwen_image_vllm_20261019/pairs/pairs-20261022.jsonl`；606433 对，训练 594454、验证 6058、测试 5921，按标准化 supplied prompt 分组 |
| batch / steps / 卡数 | global batch64，64 ranks；总 8000 步 = 观察对齐 1000 + 两轮编辑 7000 |

rank 与学习率是方案起点，不是已测出的最优配置。正式启动记录中同时固定阶段 A/B 的步数、有效 batch 和数据版本。

## 7. 首版反馈节点与计算开销

保留当前 50 个时间节点、49 次 Euler 更新、shift 3、CFG 4。选择最接近 `t=0.70/0.55/0.40` 的现有节点：

| 0-based 索引 | 面向日志的第几步 | 实际 t | 下一 t | 内循环轮数 |
|---:|---:|---:|---:|---:|
| 28 | 29 | 0.6923076923 | 0.6741573034 | 2 |
| 35 | 36 | 0.5454545455 | 0.5200000000 | 2 |
| 40 | 41 | 0.4029850746 | 0.3692307692 | 2 |

三个节点分别给较早的结构调整、中段关系调整和后段细化提供反馈机会。这是结合历史高噪声读取失败选出的工作区间，具体语义效果由本次方案的图像决定。

基础采样在 CFG 开启时约有 `49×2=98` 次 GEN 图像前向。每次编辑轮增加一次图像观察和一次 GEN 编辑，因此增加 `3×2×2=12` 次完整 GEN 图像前向，以及 12 次小 slot UND 前向。

完整 GEN 前向次数由 98 增至 110，增幅约 12.2%。这只是次数比，不是实测时延。UND attention、缓存处理、NPU kernel 和通信另计。最终只执行一次 VAE decode。

训练也要计入两轮图像前向。缓存保留当前轮的来源与反馈，逐轮 backward/释放；跨轮 proposal detach。冻结基础权重可共享，适配器与 cache 分开，避免复制数套完整 BAGEL。

## 8. NPU 落地与代码边界

### 8.1 原生接口复用

- `qwen_latent_cot/bagel/modeling/bagel/bagel.py`：复用 VAE patch 规则、`vae2llm`、时间/2D 位置 embedding、`llm2vae`。从已有 latent 开始构造 clean-context，跳过图像 encode。
- `qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py`：复用 UND/GEN 的原生投影、归一化、MLP 和 packed attention；增加显式角色适配器入口及逐层来源导出。
- `qwen_latent_cot/correction/objective.py`：参考已部署的 guidance-safe 语义；新实现支持逐图投影，保留旧 checkpoint 的入口含义。
- `qwen_latent_cot/correction/sampling.py`：参考原生时间网格与分离更新路径。新循环采用独立入口，旧结果继续可复现。

### 8.2 新模块的职责

建议新增 `qwen_latent_cot/mot_edit/`：

| 文件 | 单一职责 |
|---|---|
| `roles.py` | 只读原生权重、角色适配器及冻结关系 |
| `state.py` | clean-proposal 观察、逐层 KV、样本和位置布局 |
| `feedback.py` | UND 观察/反馈及原生 caption 训练入口 |
| `editor.py` | 原生 GEN 编辑与已有 `llm2vae` 输出 |
| `objective.py` | 完整速度目标、逐图投影与两轮 loss |
| `sampling.py` | 三节点、每节点两轮、单条外层轨迹 |
| `training.py` | 阶段 A/B、参数组、checkpoint |

对应新增配置、训练入口与单图/图册采样入口。首轮实现不迁移 SenseNova，不删除用于历史复现的旧 checkpoint loader。

### 8.3 NPU 的具体约束

1. 延用项目已工作的 torch-npu 与 attention 后端，不升级共享环境。
2. 图像/反馈 token 在同一个原生 attention 中计算；packed 样本边界明确，图像段双向、文字前缀按原生规则处理。
3. Q/K norm 和 RoPE 的计算精度沿用原生路径；长向量投影与 MSE 使用 FP32。
4. 基础生成、观察、反馈和编辑使用独立 cache 对象。一次内循环结束释放其临时 bank。
5. 先完成单卡完整尺寸的一次两轮前后向，得到显存与吞吐，再选多卡 batch。沿用现有分布式能力，不先承诺原实验的 batch 256。

## 9. 完成实现后先回答三个具体问题

### 9.1 图像状态有没有被读出来

使用阶段 A 的 4 prompt × 2 seed 中间预测，显示图像、观察描述和目标。固定目标、交换图像来源，检查描述中的对象与关系是否随来源变化。

### 9.2 反馈是否改变了正确的编辑动作

在相同 `P,t`、不同 seed 的两条状态间交换 `F`。保持每条状态的 `X` 不变，分别运行自己的反馈和另一条状态的反馈。展示下一轮预测及最终图像，指出改了哪个对象、数量、属性或关系。

这个对照固定了目标，区分图像状态反馈与目标重编码。若两条反馈只复述同一目标，它们对编辑的作用会接近；这给出直接的下一处问题，而非用“存在梯度”代替状态依赖。

### 9.3 多一轮是否带来更好的最终图像

复用已有 32 条高难 prompt 和 seed，展示原生、1 轮、2 轮完整采样结果。记录每题语义改对/改错的事实及风格变化。1 轮与 2 轮共享同一份两轮训练权重。

这三项直接对应读取、状态依赖和迭代收益。工程检查仅覆盖原生 `K=0` 路径、角色缓存隔离、两轮更新公式及本轮梯度到达反馈/编辑适配器。首轮不追加六榜大规模采样。

## 10. 与 Correction Training 的关系，以及 UND 的贡献

共同点是把模型预测作为下一次输入，固定外层状态做内循环，并沿用生成模型已有的预测接口。

本方案把其 corrector 的职责拆为两项：UND 根据当前图像和目标形成语义反馈，GEN 把反馈转成图像更新。编辑器保留已训练的生成专家和速度头；观察与反馈使用已训练的理解专家及语言输出能力。

与 Correction Training 默认联合更新 denoiser/corrector 的设置相比，首版固定基础生成器，训练专用适配器，且直接展开两轮训练。训练覆盖了本方案的两轮输入，但原生真实采样轨迹与训练图文对加噪状态仍有差别。最终图册用于观察这种差别如何影响语义和画风。

**UND 的贡献应落在“理解当前状态并形成针对性修改”上。** 输出速度的能力由 GEN 承接；原始目标仍完整保留在基础生成与 UND 比较过程里。匹配维度、原生跨专家 attention 和复用输出头，让这两个预训练能力以较少的新接口连接起来。

本方案的主要研究假设是：让 UND 在可读的中间状态上完成观察与目标比较，并让 GEN 对明确的预测图像执行编辑，会比 UND 从头学习速度残差更有效地保留画质并修正语义。历史 caption、Memory response 和 guidance 实验分别支持其中的接口选择；端到端收益由新的两轮实现给出。

## 11. 资料与实现依据

- [跨实验研究记录](research_record_20261022.md)：BAGEL、SenseNova、历史结果与六榜。
- [历史 Memory / response 记录](https://github.com/zyd-ustc/bagel-LatentCoT/blob/08b7cce/docs/experiments.md)。
- [历史 caption 来源人工记录](https://github.com/zyd-ustc/bagel-LatentCoT/commit/2c1c05f29e53ea1b80f24b31cd814c8e5f585b37)。
- [Stage A v2](https://github.com/zyd-ustc/bagel-LatentCoT/blob/dad8a08/docs/stage_a_v2_canvas_reentry.md)：旧 canvas、Memory 与门控设计。
- [Correction Training 论文核对](correction_training_paper_notes.md)、[代码核对](correction_training_code_notes.md)。
- [Correction Training 论文](https://arxiv.org/html/2610.14136v2)及[固定版本循环代码](https://github.com/sunovivid/correction-training/blob/475e21464485c8c3b38730131d071042ad6481f1/src/models/correction.py)。
- 本地原生实现：`qwen_latent_cot/bagel/modeling/bagel/{bagel.py,qwen2_navit.py}`。本方案核对了 `forward_cache_update_vae`、`Qwen2MoTAttention` 与 GEN 速度输出路径。

用户已追加实现与运行授权。新实现、pilot 证据和固定 run 位置见[运行记录](mot_edit_8000_20261023/RUN.md)；历史训练保持停止，新任务不恢复旧纠错或启动纯 T2I SFT。
