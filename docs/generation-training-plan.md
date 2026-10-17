# Memory 条件下的生成训练方案

> 2026-10-17 更新：首轮执行方案见 [理解—生成交互 SFT → RL](generation-loop-sft-rl-plan.md)。当前主线为 Observe → caption-image 交互 SFT → RL；UND＋第二遍 GEN 承担修正功能，memory 不要求具有文字反馈语义。SFT 同时训练 writer UND LoRA 与 consumer GEN LoRA，RL 首轮冻结 writer；本文保留为前期设计记录，以新方案为准。

日期：2026-10-16。状态：代码与资料审阅后的方案，尚未实现或启动生成训练。

## 1. 决策

**主线转向生成侧：固定 C2 UND，训练 GEN 消费 memory；短程 flow-SFT 是有预算上限的适配试验，Flow-GRPO 是终图优化主线。旧注意力响应蒸馏默认关闭。**

存在分布变化，不等于必须先完成长程 SFT。RL 本身也能适应新条件。选择取决于当前 memory 是否提供有效方向、GEN 是否能利用它，以及终图奖励能否区分同组候选。先完成小规模校准；如果直接 RL 的条件成立，可以跳过 SFT。不要继续以 UND loss 收敛作为启动生成侧训练的前置条件。

本次依据是本地 docs、实现与官方论文/代码。用户报告两个 UND 实验均有进步；本地 Observe 文档有量化结果，C2 有 step 2000 checkpoint 和生成配置，但没有完整正式 C2 曲线与完整 benchmark 汇总。本方案不声称已验证 benchmark 提点，也未读取远端实时运行状态。资料清单只作为检索线索，其中的工作名称、方法和 BAGEL 关系需逐项核实。

## 2. 当前实现已经证明什么

| 证据 | 当前事实 | 对方案的含义 |
|---|---|---|
| [experiment.md](experiment.md) | Observe 113 条固定验证 NLL 从 2.5865（1000 步）降到 2.0381（19750 步）；Source-off 差距扩大 | 支持读取能力进展；不等于生成质量提升 |
| [visual_objectives.py](../qwen_latent_cot/diagnostics/visual_objectives.py)，`training_objective` | C2-feedback 只执行来源 GEN、memory 写入和文字 CE，直接返回 | 没有优化第二遍 GEN 出口，也没有终图奖励 |
| [visual.py](../qwen_latent_cot/diagnostics/visual.py)，`VisualStudent` | 当前只登记原生 UND 参数 | 现有 C3 flow loss 也主要把梯度传回 UND；开启 flow 不等于开始训练 GEN |
| [runtime.py](../qwen_latent_cot/diagnostics/runtime.py) | 初始化冻结模型与 VAE，UND 后续显式解冻 | 新阶段必须单独登记 GEN 参数和优化器 |
| [flow_training.py](../qwen_latent_cot/diagnostics/flow_training.py) | `conditional_gen` 已绕开原生 no_grad，复用推理 memory 注入；支持激活重计算 | 可以复用生成监督，不必重写整个 BAGEL |
| [visual_record.py](../qwen_latent_cot/diagnostics/visual_record.py) | format 4 保存 UND；拒绝历史 residual writer/teacher 初始化 | 旧 C1 checkpoint 不能直接当新版教师接入；GEN checkpoint 格式需要扩展 |
| [shallow_sampling.py](../qwen_latent_cot/diagnostics/shallow_sampling.py) | `sample` 为 no_grad，使用确定性 Euler，无转移 log-probability | 当前入口不能直接套文本 GRPO |

文档存在版本差异：`inference-tests.md` 的旧 writer/query、C1 初始化描述已被 `native_und_v1` 和 format 4 实现替代；`experiment.md` 中“正式 C2 未启动”是早期准备记录，后文已记录 step 2000。实现审计按当前代码，运行状态按带时间的具体记录。

## 3. Memory 生图有哪些分布适应问题

### 3.1 噪声与状态来源：明确存在错位

Observe/C2 训练使用干净图加噪，t∈[0.05,0.25]。部署配置为 50 个节点、shift=3、early20。由 [schedule.py](../qwen_latent_cot/diagnostics/schedule.py) 的公式：

`u_i=1-i/49; t_i=3*u_i/(1+2*u_i)`。

前 20 次更新在 i=0…19 计算速度，t 从 1 降至约 0.825688；第 20 次更新到达约 0.813084。**部署时启用 memory 的区域与已有文字监督区域完全不重叠。**

此外，插值训练状态与模型自身采样轨迹不同。即使把 SFT 的 t 扩展到高噪声，也没有自动解决真实轨迹迁移。正确目标图的 flow 监督可以覆盖高噪声，但不能把端点修正标签直接当作高噪声当前状态事实。t=1 时尤其不能要求恢复某个指定目标图的独有细节。

### 3.2 条件接口：有适配理由，尚无“必须 SFT”的实证

原生 GEN 消费文本/图像条件。现在加入的是逐层 UND hidden 投影所得的 16 槽 KV，并移动图像 query 的 RoPE 位置。中性文字初始化提供接口起点，但不会使训练后的连续 KV 自动等价于原生文本。

C2 的文字 CE 使 memory 能被 UND 读出，不保证 GEN 能正确使用它。历史 C1 响应训练提供过小规模局部预测图改善，但不证明新版高噪声动态 memory 已适配。

还需区分两个原因：memory 内容错误，与正确 memory 被 GEN 错用。只训练 GEN 可能学会忽略错误 memory。若最终提高了分数，却无法优于去除/交换 memory 的结果，应记录为 GEN 提点，不能归因为反馈闭环。

### 3.3 参数与比较基线

UND 注意力也用于原生 prompt 编码。因此加载 C2 后的无 memory Base，不等于原版 BAGEL。所有比较至少保留原版 BAGEL、C2 无 memory、C2+memory，防止把 prompt 编码变化归为反馈收益。

## 4. 最小校准与分支决策

先固定 64 个独立开发 prompt、每题 4 seed，覆盖数量、属性绑定、空间关系和较长组合要求。使用现有 early20/L28/R2、512、CFG=4、shift=3、49 次 Euler 更新。开发 prompt 不取正式榜单题目。

核心对照为原版 BAGEL、C2 无 memory、C2+memory。另选其中 16 个有明确错误/保持标签的 prompt，测同布局屏蔽 memory、同目标异状态 memory 交换；必要时增加人工核对的原生文字修正。不要把尚未校准的旧 C1 Text-memory 当上限。

记录终图任务分、质量、正确事实损伤、同 prompt 多 seed 的奖励方差。布局/交换对照只用于定位，不扩展成全部组合矩阵。

| 可观察结果 | 决策 |
|---|---|
| Memory 有可重复的方向性收益，质量稳定；同组奖励有差异 | 直接进入 Flow-GRPO，SFT 可跳过 |
| 正确文字能纠错，但 memory 收益弱或造成退化 | 做短程 memory 条件 flow-SFT；不恢复长程 attention 蒸馏 |
| 正确/交换 memory 效果相近，反馈方向也不可靠 | 优先补少量真实早期轨迹反馈；RL 可先做 GEN-only 基线，暂不声称 memory 有效 |
| 奖励几乎恒定或明显偏爱伪影 | 修正训练 prompt 难度/奖励；延长 SFT 不能解决奖励问题 |

小样本结果用于选择路线，不作为正式 benchmark 提点结论。零初始化 LoRA 下，新增路径必须复现当前 C2+memory 输出。

## 5. 短程生成 SFT：适配消费接口

### 5.1 首轮可训练范围

固定一个明确的 C2 checkpoint，初始可用已有 step 2000；若改用后续权重，重新记录 parent/hash 和开发集结果。冻结 UND、VAE、词嵌入、速度头和 GEN 主权重。

首轮仅在 GEN 全 28 层 `self_attn.{q,k,v,o}_proj_moe_gen` 上加入 LoRA（低秩可训练增量），建议 rank=16、alpha=16、dropout=0。此为工程起点，未验证最优；瓶颈确在表达能力时再扩展 GEN MLP，不先展开大规模超参搜索。

**LoRA 仅在 memory 消费的第二遍条件 GEN 启用。** 第一遍来源 GEN、无条件 CFG 分支和窗口外 GEN 使用冻结权重。这一作用范围在训练、rollout、重算和部署中一致。

这样在固定 x_t、t、P 下，memory 不依赖待更新 LoRA 参数，可以复用现有来源停止梯度逻辑；保留 C2 的输入接口，并把首轮改动限定到消费能力。以后若将 LoRA 扩展到第一遍来源 GEN，必须重新评估 UND 来源分布，并处理 memory 对参数的间接依赖。

### 5.2 数据和目标

使用现有可读 Generation 视图中的高质量图文配对，以组合遵循为重点。先抽 8k–16k 条，按来源和目标长度分层；优先完整明确的目标 caption，不要求新建百万级数据集。Echo-4o 可借鉴数据策略，但当前数据文档列有不可读原图来源，不默认它们可训练。

对正确目标图 B 编码为 x0，抽样 epsilon，构造 x_t=(1-t)x0+t*epsilon。第一遍冻结 GEN 产生来源，冻结 UND 写 memory，第二遍带 LoRA GEN 预测速度。损失：

`L_flow = mean((v_theta(x_t,t,P,M) - (epsilon-x0))²)`。

t 采样先沿用现有 Generation 的 60% early20、20% 中段、20% 后段。高噪声只做配对 flow；不附加“当前图已经满足目标”的文字标签。SFT 中全节点可用于学习接口，部署仍固定 early20；最终选择必须看完整部署轨迹。

编辑 A→B 中，不能从 A 加噪却用 `epsilon-B` 作为该状态的原生 flow target。首轮直接使用 B 的 Generation 视图；没有配对目标的真实 trajectory 不伪造 flow target。其优化交给真实反馈教师或终图 RL。

首轮 `L_attention=0, L_teacher=0, L_text=0`。UND 冻结后，其文字 loss 没有训练 GEN 的作用。若出现正确内容受损，可在已确认保持的状态上加冻结参考的条件速度保持项；它不是替代终图奖励的主目标。

### 5.3 预算与停止条件

建议全局 batch=16、学习率 1e-5、先 500 步，按 100/250/500 步检查开发集；有终图改善才扩到 1000 步。记录真实 NPU 时间与出图成本，不沿用 UND 的每步耗时预测生成训练。

500 步后 flow loss 下降但开发集没有改善，不自动延长到 20k。若 memory 已稳定可消费，转 RL；若始终不依赖 memory，回看内容与接口。SFT 是有限的适配试验。

## 6. Flow-GRPO：以终图作为优化目标

### 6.1 借用实现与策略定义

首选 [Flow-GRPO 官方 BAGEL 训练入口](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_bagel.py)。它已有 GEN LoRA 路径；本地需要适配 Ascend、memory 和参数作用范围，不直接替换现有 BAGEL 内核。该代码的 GPU 配置不能视为已通过 NPU 验证。

策略动作定义为随机采样的下一图像 latent，而不是 memory token。连续 memory 是确定性条件，不具备文本 token 的 categorical logprob。

对选中的随机转移，记 `pi_theta(x_s | x_t,P)=Normal(mu_theta, sigma² I)`，其中均值由本项目 R2 的最终 CFG 速度和选定随机采样公式决定；sigma 由固定调度确定。必须使用同一转移公式计算采样与训练 logprob，不能把 ODE 速度 MSE 命名为 logprob。

首轮选官方随机微分方程（SDE）分支，检查本项目 t 从 1 到 0、v=epsilon-x0 的符号。官方 BAGEL 的 CPS 分支使用负平方误差分数，不能机械照搬并称为精确转移 logprob。不要只在现有 Euler 后随意加噪声，再宣称保持原模型边缘分布。

### 6.2 首轮配置

- 沿用 SFT 的消费侧 LoRA 和冻结 UND；直接 RL 分支从零 LoRA 开始。
- 同一 prompt 分组采样 G=8 张，组内独立随机转移；可共享初始噪声以集中比较随机窗口产生的改进。记录 seed 策略。
- 首轮每个 rollout batch 取 8 个 prompt，共 64 张；设备不足则顺序累计，跨 rank 聚合完整组。
- 保留 49 次生成更新，在 early20 的内部节点 i=1…19 选择 1–2 个随机转移做训练，其余沿用确定性采样。避开 t=1/0 的奇异或零方差端点。
- 使用全部 memory early20 出图，终图奖励分配给选中的随机转移；这是窗口式策略优化的代理目标，不声称精确求导整条混合 ODE/SDE 轨迹的终图期望。
- 每批旧策略数据只做 1 轮更新，先减少过期样本影响。学习率建议 1e-5，clip 起点 1e-3；实际数值需按 BAGEL/Ascend 的 logprob 与 ratio 尺度校准。
- 先 100 个 rollout batch（6400 张训练终图）作为试验预算，每 25 批看开发集；用真实吞吐估算后续成本。

组内优势 `A_i=(r_i-mean(r))/(std(r)+eps)`，奖励相同的组不产生策略梯度。对固定旧策略转移，`rho=exp(logpi_new-logpi_old)`；最大化标准 clipped surrogate，并采用固定初始化参考策略的转移 KL 约束。KL 系数先以限制开发集退化为准校准，不同时叠加多种蒸馏目标。

Gaussian 的逐维 log-density 求和才是严格的联合 logprob；首轮建议沿用官方 SDE 的逐维平均分数作为缩放的代理比率，显式记录该约定；KL 单独按固定方差 Gaussian 计算并说明逐维平均尺度。不能将其写成精确联合似然比，也不能与求和版本混用 clip/KL 超参。

**关键检查：** 参数未更新时，重算 logprob 与 rollout 一致，ratio 接近 1；消费侧 LoRA 得到有限非零梯度；UND 与来源 GEN 不变；CFG、RoPE、节点、memory、精度和 batch 组织一致。激活重计算必须恢复同样的 LoRA 启用范围，不能只依赖前向外层临时开关。

如果未来让第一遍 GEN 也训练，重算必须使用当前参数重建来源和 memory；仅把旧 memory 当外生条件，会改变所优化的策略。冻结 UND 参数不等于可以切断穿过 UND 输入的全部梯度。

### 6.3 奖励与 benchmark

首轮只集中做组合遵循：对象、数量、属性绑定、空间关系。使用独立训练 prompt 及其原子问题，复用已有检测/VQA 能力。需要概率型细粒度分来减少全零组；图像质量作为单独约束或经校准的小权重分量，避免仅靠整体 CLIP 相似度。

先固定一种可运行的任务奖励，再观察组内方差；不同时接六个昂贵裁判。已有 GenEval2 Qwen3-VL-8B 是候选，但文档仅确认 CPU 评分与 NPU 导入检查，正式 NPU 吞吐必须实测。奖励可异步批量计算，裁判尽量不与生成抢同一设备。

正式 GenEval/GenEval2/DPG/DSG 的题目和答案不用于训练采样或权重选择。使用同类独立训练题，保留一个不同裁判或人工盲审切片检查奖励投机。报告零方差组比例、质量与多样性，不能只报告训练奖励。

RL 训练采用随机采样；最终 benchmark 继续用既有确定性 Euler 配置。分别报告随机训练采样与确定性部署表现，防止把随机采样本身的差异当作权重收益。

## 7. 旧注意力响应蒸馏怎么处理

**默认不保留为主损失；保留代码作诊断或必要时的小权重辅助。**

[response.py](../qwen_latent_cot/diagnostics/response.py) 在固定、停止梯度的 query/common KV 上匹配 attention response，读出点为输出投影前；它不是完整生成速度，也不是 attention map KL。旧实验只看了少量层/探针，教师来自文字 caption。

它可以建立“文字条件如何影响局部注意力”的接口，但不负责最终对象数量、关系和图像质量。新版 memory 表达动态修正，旧 caption 压缩教师未必对应相同任务。过强约束会把训练锁定到教师行为，妨碍超过教师。

只有正确文字教师在同状态终图上确实有效，而直接 flow/RL 显示接口优化困难时，才尝试短程辅助蒸馏。须使用新版冻结教师、相同 prompt/t/布局与真实 query，并与不加蒸馏比较。优先蒸馏完整条件速度而非恢复旧探针训练。若 GEN 正在更新，教师必须是独立冻结参数；不能在同一可训练 GEN 上临时 no_grad 就把它当固定教师。

## 8. 最小实验比较和晋级规则

| 实验 | 初始化 | 训练 | 回答的问题 |
|---|---|---|---|
| B0/B1/B2 | 原版 / C2 无 memory / C2+memory | 无 | 当前基线与 memory 增益 |
| G1 | B2 | 500 步 flow-SFT | 接口适配是否带来终图收益 |
| G2 | B2 | 直接 Flow-GRPO | 是否可跳过 SFT |
| G3 | G1 | 同预算 Flow-GRPO | SFT 是否提高 RL 的最终收益或效率 |

先完成 B2 的小校准并启动最有依据的一条路线；G2/G3 只做有限匹配预算比较，不要求所有大实验同时运行。至少报告累计 NPU 小时、训练终图数和评分耗时；G3 的成本必须包含 SFT。

在独立开发集选择模型，正式榜单只比较固定候选。主指标配对到 prompt/seed；置信区间按 prompt 聚合重采样，不把同题多 seed 当独立问题。要求任务分改善且质量/正确事实保持没有明确退化；结果不确定时扩大评测，不用训练 loss 代替证据。

为了判断收益是否来自 memory，最佳模型补做 memory 屏蔽和同目标状态交换。若需要声称“memory 优于普通 GEN RL”，再增加同等训练成本的无 memory GEN-LoRA RL；单纯屏蔽并不能代替这个训练对照。

## 9. 实现任务与复用边界

| 顺序 | 修改位置 | 具体工作 |
|---|---|---|
| 1 | 新生成阶段参数配置、`visual.py`、`visual_record.py` | 支持冻结 UND+GEN LoRA；新 checkpoint 保存 UND parent/hash、LoRA、启用范围、优化器；兼容读取 C2 format 4 |
| 2 | `flow_training.py`、`loop_training.py`、`native.py` | 保留 R2 注入；按调用启停消费侧 LoRA；激活重计算重入相同上下文 |
| 3 | 生成 SFT 独立入口、现有 corpus/NoisePolicy | 使用 Generation 正确目标 flow；从 optimizer 到保存文件明确只训练 GEN |
| 4 | 新 stochastic rollout 与 logprob 模块 | 移植 BAGEL Flow-GRPO 的转移；保留本项目 CFG/布局；保存 prompt、x_t/x_s、t/s、old logprob、seed、策略版本 |
| 5 | 新 GRPO trainer、现有分布式与日志工具 | 按 prompt 聚合奖励；LoRA current/old/reference 管理；ratio/KL/方差/成本记录 |
| 6 | `memory_benchmark.py` 与生成入口 | 加载 GEN adapter；继续输出可配对图像与官方评分；附原版、C2、SFT、RL 身份 |

必要验证只围绕新增行为：零 LoRA 复现；作用范围与梯度；保存/恢复；随机采样与 logprob 重算一致；同组跨 rank 聚合；小规模 NPU 一轮真实训练和终图。已有 UND 130 项测试通过的记录不能替代新生成/RL 路径验证。

本轮只形成训练方案，没有改动已有训练代码、启动脚本或正在运行的实验。

## 10. 外部依据

详见 [BAGEL 生成优化工作核验](generation-prior-work.md)。最短实现依赖为 [Flow-GRPO](https://github.com/yifan123/flow_grpo)；窗口式优化与 SDE 转换参考其[官方说明](https://github.com/yifan123/flow_grpo)及[算法文档](https://github.com/verl-project/verl-omni/blob/main/docs/algo/flowgrpo.md)。LoRA 模块与实际优化器可对照 [BAGEL 训练代码](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_bagel.py)。

这些工作说明可采用的优化方法，不证明本项目 memory 已经具备相同的分布、语义或效果。
