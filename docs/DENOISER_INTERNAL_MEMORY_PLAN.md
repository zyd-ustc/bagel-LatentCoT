> 实现状态（2026-10-05）：`main` 已实现免训练算子、对照、原生一致性测试和离线语义/质量评测。以下研究协议保留原设计口径。工程结果不等于独立确认；训练与 E4 原生语义 probe 尚未实现。执行入口及证据见 README 和本次工程报告。

# Denoiser 内部语义循环：机制、测试、训练与证据方案

日期：2026-10-05。状态：研究与实施方案；新算子尚未实现，本方案的实验尚未执行。

本文件承接用户最新约束，重新定义下一阶段研究路线。它不把旧 Anchored adapter 的结果视为新方案的验证，也不修改旧实验记录。配套文件：DENOISER_INTERNAL_MEMORY_PROTOCOL.yaml 和 DENOISER_INTERNAL_MEMORY_CHECKLIST.md。

## 1. 目标与研究边界

最终任务始终是文生图。输入是 prompt 和初始噪声；输出是图像。推理时在一次 denoiser 调用内部增加受控计算，修复对象数量、属性绑定、空间关系等语义错误，并尽量保留已经正确的内容和画面质量。

Memory 的目标是承载 UND 从当前生成状态中加工出的、GEN 可以使用的信息。这里的信息增量可以是对已有信息的整理，不要求引入外部新事实；但必须证明它超出了重复 prompt、增加 token 数或增加 GEN 计算次数的效果。

项目约束：

- 推理主线不生成反思文本，不将中间图像交给 ViT，不调用外部评审，不进行外部图像编辑或多候选择优。
- ViT、属性标注和文字解释只用于离线监督、诊断或结果评估；不得进入正式推理的输入。
- 不默认添加 re-entry adapter、零输出投影、输出混合 α、额外可学习 gate 或强化学习阶段。
- 不依赖 GEN SFT 改善画质，不以延长训练挽救没有免训练有效编辑信号的算子。
- 理解侧格式/隐式适配属于后续条件分支；其任务是稳定、压缩或规范已经观察到的编辑能力。
- “原生 UND 看懂 noisy GEN”“8/16 个 slot 足以保存修正信息”都属于假设，不属于 BAGEL 已证实的能力。

本文给出的窗口、样本数、预算和准入阈值均为本项目的拟定实验协议，不是论文结论。正式运行前冻结协议及数据哈希；不能根据测试结果放宽阈值。

## 2. 来源、可用思想与证据边界

理论与机制依据限定为用户指定的 BAGEL、两篇循环论文、SLVR 和 UNO。当前仓库只用于定位后续实施接口，不作为新机制已经成立的证据。

| 来源 | 已核对的事实 | 本项目借鉴 | 不能据此声称 |
| --- | --- | --- | --- |
| BAGEL 源码 | 原生 GEN 读取 prompt KV；文本/图像可以通过原生路径写入上下文。训练掩码隔离生成噪声块，块外 token 不能直接读取该块。 | 原生专家、按层 K/V、位置与掩码、flow 目标。 | 额外 UND slots 已经学会解释 noisy GEN。 |
| Training-Free Hidden-State Refinement，2608.29160 | 同一层重复计算残差，单次强度为 1/N；主要冻结实验使用 RAE。GenEval2 表 14 中原模型/循环/Loop Guidance 为 0.1801/0.1736/0.1441；VAE 迁移没有一致收益。 | 先控制同层累计更新强度，再研究重复计算。 | 对 BAGEL 的计数/关系修复有现成保证。 |
| Looped Diffusion Transformer，2609.40305 | 图像与文本状态共同经过共享 body；中间输出接受相同目标监督。主实验是经过预训练与微调的循环模型。 | 联合状态演化、直接监督不同深度。 | 冻结 BAGEL、只训练理解侧具有同样效果。 |
| SLVR，本地代码与 2605.19342v2 | 使用视觉与属性语义监督；代码包含语义投影，并训练语言模型。源码的训练输入含目标特征注入和下一位置预测。 | Memory 的内容应明确到对象属性、关系；同一状态应支持多项语义判断。 | 视觉特征相似等于有效生成编辑；其教师强制训练等于部署时自回归/连续状态效果。 |
| UNO，2605.05781v1，本地仅有论文 | 冻结 UND，通过 caption 与视觉监督更新 GEN；屏蔽监督 token 对 prompt 的直接读取，并采用重新描述。 | 理解信号的组织方式、监督泄漏防范、区分语言与视觉目标。 | UNO 验证了“冻结 GEN、只训练 UND”；普通 SFT 在所有情况下均无效。 |

UNO 表 4 中普通 SFT 本身也有指标改善。因此，本项目“不走 GEN SFT”来自用户的实验经验与路线约束，而非声称文献证明它普遍无效。UNO 的 GenEval2 主表使用 self-CoT，不能与本项目无显式 CoT 的数字直接比较。

SLVR 的语义投影和 UNO 的 metaqueries 不是本方案默认组件。只借鉴监督思想，避免把不匹配的表示空间靠任意新增 head 拼接起来。

## 3. 核心假设与最低证据

| 编号 | 可证伪假设 | 最低证据 | 不能替代它的现象 |
| --- | --- | --- | --- |
| H1 | 冻结 BAGEL 的内部重复计算存在有效语义编辑区间。 | 同 prompt、噪声下，最终图像的语义修复超过破坏，且开销合格。 | velocity 改变、单张好图、loss 降低。 |
| H2 | UND Memory 能获得与当前生成状态有关的内容。 | 同 prompt 不同生成状态下，对实际可见属性的判断随证据变化；在 prompt 与实际图像不一致的案例中不能只复述 prompt。 | Memory 范数、注意力质量、有效秩非零。 |
| H3 | GEN 会利用 Memory 的更新内容。 | 动态 Memory 比静态 Memory、关闭 Memory 读取更有效，比较保持其余计算与位置约定一致。 | Memory 读入后图片发生变化。 |
| H4 | 内容更新带来的收益超过纯 GEN 循环及等时延原生采样。 | 独立确认集上的配对语义和质量结果、真实延迟。 | 不同 FLOPs 或不同 CFG 下的总分比较。 |
| H5 | 理解侧适配改善接口稳定性，而非改变生成主干能力。 | GEN/原始 prompt 路径权重不变；训练后有效编辑率或预算效率提高，R0 保持原生行为。 | 理解 QA 更准或监督 loss 更低。 |

先验证 H1，再验证 H2/H3，最后验证 H4。H5 仅在训练准入成立后研究。若只有 H1 成立，应将结果称为 GEN 内部细化，不能宣称 Memory 方法成功。

## 4. 首个候选算子：同层联合细化与临时 Memory KV

以下是拟实现的最小候选，不是已有论文的直接复现。首轮只比较纯 GEN 与带 Memory 的同层算子，暂不同时改变外部采样器、CFG 强度和输出读出。

### 4.1 状态、深度和生命周期

- t：原生 flow 时间；一次内部循环期间 x_t 与 t 固定。
- s=i/(S-1)：从噪声到图像的采样进度。s 不能直接替代 BAGEL 的 shifted timestep。
- l：模型层号。初始候选层区间为 [16,24)，左闭右开，共 8 层。
- N：每个选中层的总 GEN 计算次数；N=1 为原生路径。额外次数 R=N-1。禁止复用旧 runner 的 R 而不转换语义。
- G_l^r：当前图像查询序列在第 l 层的第 r 次状态；包含原生 SOI/EOI 边界，保持原生 UND/GEN 路由。
- M_l^r：第 l 层的局部 Memory 状态。按层保存，仅在该层的内部迭代中更新。
- P_l：原生 prompt KV，整个 denoiser 调用中只读。

每个 timestep 从当前 x_t 构建原生输入。Memory 不跨 timestep 携带，不将上一层的最终 M 直接作为所有层的 KV，也不把某个 body 末端状态送回较浅层。后续若研究跨层/跨 timestep Memory，必须作为单独实验，不能悄悄改变本协议。

此版本的 Memory 是按层的临时 KV bank；它不是一个跨全网共享的单一 K×D 状态。实际存储成本必须包括所有选中层。

~~~mermaid
flowchart LR
    P["Prompt native prefill"] --> KV["只读 Prompt KV"]
    X["当前 x_t 与 t"] --> A["原生 prefix"]
    A --> B["选中层：GEN / UND Memory 同层细化"]
    KV --> B
    B --> M["更新 Memory KV"]
    M -->|"下一次层内迭代读取"| B
    B -->|"所有选中层完成"| C["原生 suffix 与 velocity"]
    C --> S["外部采样器推进一次"]
~~~

### 4.2 非零且无新增参数的初始化

prefill 时额外保留选中层入口的原生 prompt hidden states。去掉模板/特殊 token 后，按文本序列位置均匀选取最多 K=16 个不同 token，作为各层 M_l^0。短 prompt 使用实际 token 数，不复制补齐；无可用文本 token 时 M 为空。

这是确定性、无语义 oracle 的工程起点，不声称均匀选取最佳。它避免把多个 slot 初始化成同一 boundary embedding，也避免新增随机投影。若原生表示本身退化，诊断须如实报告。

首版保留被选 token 的原始 RoPE position。Memory 的物理 KV 存放位置在 P 后，但不把“存放顺序”误当作“必须重新编号”。GEN 的 position 全程保持原生值，以免位置移动成为收益来源。若以后采用连续新位置，必须连同配套对照作为另一候选版本。

### 4.3 更新与一轮延迟

令 η=1/N。F_l^G 表示该层原生图像查询更新，F_l^U 表示该层原生 UND 更新，包括各自的 norm、attention、MLP 和 residual。

对 r=0,…,N-1：

1. 从当前 G_l^r、M_l^r 计算该层原生 Q/K/V；同一轮的读取均基于更新前的状态。
2. G 在 r=0 只读取 P 和当前 G；r≥1 才读取 P、当前 G 和 M_l^r。
3. 若下一轮仍需 Memory，M 读取 P、当前 G、当前 M，通过 UND 更新。
4. 两种状态都按 η 累计该层残差。最后一轮无需计算无消费者的 Memory query/MLP。

数学约定：

    G_l^(r+1) = G_l^r + η [F_l^G(G_l^r; P_l, read(M_l^r)) - G_l^r]
    M_l^(r+1) = M_l^r + η [F_l^U(M_l^r; P_l, G_l^r, M_l^r) - M_l^r]

第一轮 read(M)=空；其余轮为当前 M 的原生 UND K/V。初始化 Memory 不直接进入第一轮 GEN。Memory 只有在读取过当前生成状态后，才进入下一轮 GEN。

N 次完成后将 G_l^N 交给下一层；该层 Memory 不再更新。N=1 直接走原生层，不分配 Memory、不改变位置、不调用 writer。

这是把“同层残差细化”与“理解/生成状态交互”结合起来的新假设。归一化系数是按 N 定义的固定更新规则，不是可学习 gate；它不保证该系统稳定或语义正确。

### 4.4 attention 与缓存接口

| Query | Prompt KV | 当前 GEN KV | 当前 Memory KV | 教师/标签 |
| --- | --- | --- | --- | --- |
| 原 prompt | 使用原缓存，不重算 | 不读 | 不读 | 不读 |
| GEN，第 0 轮 | 读 | 读 | 不读 | 不读 |
| GEN，后续轮 | 读 | 读 | 读 | 不读 |
| Memory writer | 读 | 读 | 读 | 不读 |

Memory 的 K/V 使用该层 UND input norm、K/V projection、K norm 与 RoPE 生成。GEN 保留该层原生 GEN Q/K/V/O 和 MLP。不能在读取路径再加一个 Memory projection 或输出 α。

P 的 NaiveCache 不变。Memory 作为 attention 层的临时 overlay 追加；各层单独计算 packed KV indexes 和 lengths。未启用 Memory 的层没有这些条目，不能把同一个加 K 的全局长度广播给所有层。

同一轮中 Memory 只能作为一份 K/V 出现，不能同时在 cache 与 query sequence 中重复计入。实现可以合并 G/M query，也可以分开执行，但必须数值验证相同掩码下两种路径等价。优先复用 G 的投影，避免 writer 额外执行整套 GEN MLP。

NaViT 保持 packed token 表示和 per-sample offsets；不能要求 batch 内图像 token 数相等。CFG 各分支拥有独立 P/M，不能从 conditional branch 把 Memory 复制给去条件分支。无文本内容的分支不人为填入 conditional Memory。

所有实际执行的 CFG 分支采用相同的 GEN 循环窗口、N 与进度规则；分支之间只有自身条件及由其产生的 Memory 不同。不能只循环 conditional 分支却将结果当作排除了 CFG 改动的比较。

### 4.5 开销控制

首个候选只在 s∈[0,0.5]、[16,24) 层运行，N=2，K≤16。前期最多扩展到 N=3；N=4 属于通过后才做的边界测试。

以 32 层、8 层循环、约一半采样步为例，忽略 Memory 与不同层成本，N=2 的层计算比约为 1+0.5×8/32=1.125。这只是估算；writer、额外 attention、kernel 调度与缓存成本必须实测。

拟定部署门槛：相对原生 Base，端到端生成延迟中位数≤1.35×，p95≤1.50×，峰值显存≤1.20×。这不是当前实测能力。若真实开销超限，先缩窗口或启用步数；不通过缩分辨率、减采样步或削弱基线来伪装合格。

普通推理只运行一次最终 suffix/readout；诊断、训练的多次读出另记开销。

## 5. 免训练测试顺序与停止条件

所有新 arm 名称都是协议 ID，尚不是现有脚本已支持的 CLI mode。

| ID | 测试 | 改变的唯一核心因素 | 输出与决策 |
| --- | --- | --- | --- |
| E0 | 原生接口正确性 | 新 runner 的 bypass 与 N=1 | 与 refs/Bagel 的同权重同输入 velocity 对齐；先排除实现差异。 |
| E1 | 纯 GEN 同层细化 | 原生层改为 N=2，Memory 为空 | 是否存在免训练有效编辑信号；没有则只允许预定的一次窗口调整。 |
| E2 | 动态 Memory | 在 E1 上加入上述 UND writer 与 KV read | 是否修复更多语义错误；最终图像必须优于 Base，不能只优于较差的 E1。 |
| E3 | 内容用途 | 与 E2 比较静态 Memory及关闭读取 | 区分动态内容、条件重复、额外计算；不默认做跨样本 shuffle。 |
| E4 | 内容是否来自实际状态 | 同 prompt 不同初始噪声的状态诊断 | Memory 对实际计数/关系的判断能否区分不同状态；只作诊断，不将探针准确率当生成收益。 |
| E5 | 独立确认与预算 | 固定唯一候选，扩大到未使用 prompts 和 seeds | 判定准入、失败或证据不足；含等时延原生采样比较。 |
| E6 | 边界 | 通过 E5 后，N=3/4、长宽比及较高分辨率 | 有效深度、失效区间、成本边界；不得反向用于选择 E5 最佳配置。 |

E0 必测：N=1、Memory 写但 GEN 不读、单样本/packed 多样本、不同长宽、CFG 分支隔离、每层缓存长度、缓存无污染。BF16 parity 阈值需按原生重复运行与 kernel 差异校准，另用小规模高精度 attention 验证 mask；不能为迁就错误而任意放宽。

E3 的 static-M 固定为 M_l^0，后续读取时序与动态版相同；writer 可以继续计算并丢弃结果，以匹配开销。no-read 同样运行 writer，但移除 GEN 对 M 的读取；它应对齐同配置 GEN-only 的数学路径。两种对照分别回答“内容更新有用吗”和“读取有用吗”，无需穷举全部控制组。

E4 的重点是 prompt 要求与实际生成不一致的样本。标签写“实际有两个对象”，不能因为 prompt 要求三个就标成三个。对高噪声时刻无法可靠判断的内容标 unknown。辅助 native UND 问答读取 Memory 的探针只能离线运行；探针自身若无法通过有已知内容的 native KV 校准，就不能用其失败否定 Memory。

### 5.1 数据与预算

| 阶段 | 拟定规模 | 用途 |
| --- | --- | --- |
| 工程检查 | 8 prompts×1 seed，覆盖不同长宽 | 正确性，不报告质量结论。 |
| 开发筛选 | 64 structural + 32 ordinary/easy，seeds 0、1 | 选择一个候选；历史 hard128/easy16 只能作为开发资料。 |
| 独立确认 | 256 structural + 128 ordinary/easy，seeds 11、23、37 | 配对确认；与开发、历史评测、训练数据去重。 |
| 证据不足的唯一扩展 | 追加独立 256 structural + 128 ordinary/easy，同 3 seeds | 配置不变；到此仍不确定则不进入训练。 |

使用实际可取得的 GenEval2 与 spatial 题目及 ordinary prompts，保留原题、答案和来源 ID。数据尚未绑定；冻结实际 manifest 后才能执行。未获得独立确认集时，不得用历史数据重命名成 holdout。

开发阶段预定最多 4 个算法配置：默认 E1/E2；若无信号，追加同样 N=2、s 窗口但层区间改为 [12,20) 的 E1/E2。禁止扩展成窗口×深度×slot×CFG 的无界搜索。E3 只对最有希望的动态 Memory 候选执行。E5 比较 Base、胜出 E1、胜出 E2、static-M 和等时延原生采样；no-read 的接口等价性在 E0/E3 留证。

E5 首批为 384 prompts×3 seeds×5 arms=5760 张图；若执行一次等规模扩展，总计最多 11520 张确认图，不含开发和边界实验。运行前用真实单图时延估算 GPU 小时；本文不虚构耗时。等时延 Base 的采样步数只在开发集标定，确认阶段误差目标为±5%；无法满足时报告实际差值，不称严格等预算。

训练准入需要 E2 的正向收益及 E3 对动态内容的支持。仅 E1 有效时，保留纯 GEN 结果，停止 Memory 的训练分支。

### 5.2 指标与统计

主指标为最终图像的官方语义分数和约束修复。保留原始 atom 分数，不只输出均值。

- Repair：Base 失败、候选通过的约束数。
- Damage：Base 通过、候选失败的约束数。
- Net repair=(Repair-Damage)/全部配对约束数。
- 同时报 Repair/Base失败数、Damage/Base成功数；两者分母不同，不能直接相减代替 Net repair。
- prompt 全约束通过率、官方 soft-TIFA/GM 单独报告；不混用 atom 与 prompt 口径。
- 质量：固定 judge 的独立画面质量代理、invalid、普通/easy 保留率；与语义问题分开。
- bootstrap 按 prompt 聚类重采样，保留同 prompt 的全部 seeds 与 atoms；确认集报告 95% CI。
- 解码损坏按预注册规则计失败；scorer 崩溃/缺结果属于评测故障，修复后重评，不能偷偷丢样本。

训练 teacher 与确认 scorer 尽量采用独立来源；同源时明确记录风险。确认结果另以固定随机种子抽取 128 对图像，隐藏 arm 名称做语义及质量核查，覆盖成功、失败和无变化，保留抽样规则。未完成独立核查时，结论只能称“自动评测支持”，不能直接等同于人工质量偏好。核查结果不能反过来挑选确认集配置。

E5 的拟定门槛：

1. 动态 Memory 相对 Base 的 Net repair 点估计≥2 个百分点，95% CI 下界>0；语义 GM 的配对差下界也>0。
2. 相对 E1 和 static-M 的 Net repair 配对差均为正，95% CI 下界>0，才支持 Memory 内容贡献。否则仅称候选、不能训练。
3. ordinary/easy 的全约束通过率差，95% CI 下界≥-2 个百分点；质量代理在 [0,1] 尺度上的差下界≥-0.02。
4. invalid 增量的 95% CI 上界≤1 个百分点；成本满足 §4.5。
5. 与等时延原生采样比较不出现明确劣势。若等时延 Base 已更好，则不宣称循环有预算优势。

这些是 go/no-go 阈值，不是现有效果。CI 跨界属于证据不足，不属于通过；仅允许一次预定扩展。数值门槛若因业务需求调整，必须在确认集运行前记录新版本。

目标不是每个样本随 N 单调改善。报告修复、破坏及深度边界；不以“某一深度最好”替代跨样本证据。

## 6. 理解侧适配：严格受准入约束

### 6.1 为什么训练、训练哪些参数

只有免训练内部循环的有效编辑及 Memory 内容贡献成立，才进入本阶段。它用于改善不稳定的 Memory 表达或相同效果所需的开销，不承担从零创造 GEN 编辑能力的任务。

固定原始 GEN、VAE、ViT、原始 prompt prefill 与边界 token 的执行参数。适配选中层中用于 Memory writer 的原生 UND 参数。

为避免更新 UND 后连原始 prompt 编码都改变，拟采用 writer 专用参数副本，逐项复制原生 UND norm、attention、MLP，初始数值完全一致。它不是新增随机/零初始化 adapter，但确实增加了参数与显存，必须如实计入预算。若副本开销超限，先收缩 writer 层数并重走免训练确认，不默认改用 LoRA 掩盖问题。

不能只冻结 GEN 权重后把整个 GEN forward 放进 no_grad：图像损失仍需穿过固定 GEN 的计算图回传到 writer。原始 prompt prefill 可冻结并缓存，不能误 detach 有效 Memory 路径。

训练前后对固定参数做哈希校验；writer 停用时应保持原始 R0。单独更新用于评审的 teacher 会让尺度漂移，因而 teacher 固定。

### 6.2 监督数据必须区分“要求”与“观察”

每条训练记录至少保存：

    sample_id, split, prompt, target_image, bucket
    native_model_hash, rollout_seed, schedule, layer_window, N, probe_step
    desired_constraints
    observed_constraints, observation_confidence, observation_source
    teacher_visual_target, teacher_caption_or_attributes
    provenance, duplicate_group_id

desired_constraints 来自文本意图；observed_constraints 来自图像证据；两者不能互相充当标签。

ViT 对目标干净图像的特征可用于“目标表示”监督，但不等于当前 noisy GEN 已经包含的可见事实。当前状态的标签来自实际中间估计图或固定续采样诊断，并记录具体来源；未来续采样图的标签不能表述为当前已经看见的对象。低置信时刻不强制给“观察正确”的标签。

同一图像/区域的计数、属性、空间关系分别形成问题，增加联合正确率检查，借鉴 SLVR 的多属性思路。改写描述必须依据图像而不是同义改写 prompt，否则仍可能形成复制捷径。训练与确认按 image/prompt/来源组去重。

拟定桶配比：ordinary 40%、structural 40%、easy/noop 20%。easy/noop 须有 Base 已满足约束的证据，不能仅凭 prompt 短或数据来源推断。比例为项目起点；记录实际抽样占比和分桶指标。

### 6.3 多深度监督：保留原生 flow 定义

BAGEL 定义：

    raw_t ~ Normal(0,1)
    u = sigmoid(raw_t)
    t = shift*u / (1+(shift-1)*u)
    x_t = (1-t)*x_0 + t*epsilon
    v_target = epsilon-x_0

shift 使用所绑定原生训练配置，不能把论文 2 的相反时间约定、noise scale 或像素空间目标搬过来。

对同一 x_0、epsilon、t、condition，选择多个 N，分别完成整个合法网络路径，再用同一 velocity target 监督：

    L_depth = sum_N w_N * MSE(v_t^(N), epsilon-x_0)

重要区别：本方案逐层循环，不是 Looped-DiT 的整个 B stage 循环。N=2 完成的整段窗口不能直接当成 N=3 路径的中间出口。因此，多深度训练需要共享 prefix 后分别运行各 N 的窗口，再经过 suffix；不能在任意层的某个小循环后直接接 suffix 冒充合法读出。这是借鉴 deep supervision 的目标思想，不是复现其计算图或训练成本。

N=1 无 Memory 参数梯度，只用于原生对照。训练深度从已通过免训练筛选的 N 中选择：首轮只用 N=2；只有 N=3 也显示有效时才纳入。若使用 N=2、3，两项 loss 权重归一化为 1/2、1/2。N=4 始终先标记为 unseen，不能预先宣称 elastic-depth 能力。

### 6.4 语义与视觉监督按需求加入

默认首个训练配置为 L_depth + 语义内容监督。语义监督通过固定原生 UND/LM head 的训练期辅助读取，要求 Memory 支持实际属性问答或非逐字复制的描述；该辅助读取不进入推理。

辅助 query 对原 prompt 的直接读取应屏蔽，只读取所需 Memory 与问题/先前答案 token。即使如此，Memory 本身仍可能转存 prompt，所以必须使用“实际图像与 prompt 不一致”的标签及多噪声样本检查间接泄漏。GEN 和 Memory writer 不得读取答案 token、教师视觉特征或目标描述。

只在语义信号不足且存在坐标/维度明确对应时加入视觉监督。不能直接把 K=16 的非空间 Memory 与 256 个 ViT patch 做逐元素 MSE，也不能为凑维度任意加投影。可选方案是在训练期用原生 UND 辅助 queries 从 Memory 读取，预测 BAGEL 原生视觉 connector 后的特征；需先验证 query 网格、维度、归一化和位置一一对应。该模块及其监督用途单独标注，不进入生成读取通路。

对所有 teacher targets 停止梯度。Loss 权重依据开发 batch 的梯度尺度校准后固定，并记录各项对 writer 的梯度；不照抄 UNO 的 λ，因为其更新方向和目标结构不同。

必要的比较只有：免训练候选、相同预算的 depth-only 理解侧适配、加入语义监督的理解侧适配。视觉监督仅在明确证据缺口时增加。先不引入 M-GRPO、奖励模型或学习型退出器。

### 6.5 有界训练计划与退出

在 E5 通过后，先进行最多 200 个 optimizer steps 的接口试验；64 条训练记录，另有独立 64 条开发记录。每 50 steps 检查梯度路径、内容判断、实际 T2I repair/damage 和普通/easy 保留率。这个阶段只验证适配有无作用，不能报告泛化成功。

起始优化设定：AdamW、writer learning rate 1e-5、weight decay 0、20-step warmup、grad clip 1、global batch 32。这些是拟定试验起点，必须检查显存和每步 token 数，不复制 UNO 的完整训练配方。

试验通过后，最多 1000 steps，最多 8192 条去重且监督可信的配对记录，维持 40/40/20 桶配比；每 200 steps 在开发集检查。与 200-step 试验合计上限 1200 steps，不自动延长。使用全新独立确认 prompts 评价训练后结果，避免反复查看 E5 的确认集调参。

训练通过标准仍是图像级语义/质量/成本，而非 loss。训练后若只能改善辅助 QA 或视觉特征回归，而图像修复没有提高、开销也没有降低，停止此适配路线。若质量受损或 writer 对 prompt 复制捷径加重，回退到免训练版本。

保留原生 condition-dropout 的配置来源；去条件分支根据自己的输入构建 Memory。训练图与部署图一致，不向学生注入教师特征后再在部署时移除。SLVR 的教师强制特征注入不能不加区分地搬入本训练主路径。

## 7. 证据记录、归因与决策

每次运行保存：代码 commit、未提交 diff/hash、原生权重 hash、writer checkpoint/hash、配置 hash、prompt/图片/初始噪声 hash、seed、CFG、实际 N/K/窗口、采样步、位置规则、mask 版本、GPU/精度/kernel、图片、原始 scorer 输出、耗时与峰值显存。

Memory 诊断记录 effective rank、mean pairwise cosine、slot std、每轮 update ratio、attention read mass。它们用于发现塌缩/失效，不参与质量成功门槛。

报告区分四类证据：

1. 接口证据：谁能读取谁、K/V 是否真的接入、梯度是否到 writer。
2. 内容证据：Memory 是否区分当前状态、是否只是复述目标。
3. 干预证据：动态内容的读取是否改变了正确的语义结果。
4. 产品证据：最终 T2I 的质量、语义、开销是否共同达到要求。

辅助图、单步 x0 estimate 和 probe 不能替代最终完整采样结果。个例展示须绑定全部结果列表，包含 Repair、Damage、无变化案例，不能只挑成功图。

| 结果 | 决策 |
| --- | --- |
| parity/mask 失败 | 修接口，不解释模型能力。 |
| 预定冻结候选均无有效编辑 | 停止该算子；不启动训练。 |
| 纯 GEN 有效，Memory 无增益 | 保留 GEN 细化结论，停止 Memory 训练分支。 |
| Memory 可读、内容判断较好，但 GEN 不修复 | 写入内容尚未成为有用增量，不凭理解分数启动训练。 |
| 最终图像改善，但静态 Memory 一样有效 | 只能归因为条件重复或额外计算，不能声称 Memory 反馈。 |
| 免训练 Memory 有效且预算合格 | 允许有界理解侧适配，目标为可靠性或成本。 |
| 适配后 loss 降但图像不改善 | 停止适配，保留免训练结果。 |
| 超预算或质量门槛失败 | 不作为可部署方案；不能仅用结构分数宣布成功。 |

## 8. 实施任务与执行环境

现有 anchored_loop.py 的 α/gate/adapter 语义不适合直接换名复用。新算子应独立实现，旧 Legacy/Anchored 路径保留为历史可复现对象。

| 工作包 | 拟实施位置/产物 | 验收 |
| --- | --- | --- |
| W0 | 绑定 native 权重、prompt manifests、scorer 与环境 | provenance 完整，独立数据可用。 |
| W1 | 新增 layerwise 内部循环；扩展 navit_loop.py 的临时 KV overlay | E0；packed、RoPE、专家分流与缓存正确。 |
| W2 | 扩展 scripts/evaluate/t2i_loop_matrix.py，新增独立 mode | N/K/窗口/统计独立；无旧 α、gate 污染。 |
| W3 | 复用并核对 evaluation/loop_results.py 的评分与配对逻辑 | prompt-cluster CI、Repair/Damage、质量/成本一致。 |
| W4 | 完成 E1–E5 与决策报告 | 明确 go / stop / inconclusive。 |
| W5 | 仅准入后，新增理解侧 writer 训练入口 | 冻结参数证明、teacher 掩码、正确多深度计算图。 |
| W6 | 条件性训练与最终独立验证 | 模型收益与成本证据齐全。 |

H200 的预期执行方式为 8 卡各一个独立生成 worker，按 prompt/seed shard 分配，不默认使用 8 卡 DDP 推理。每张卡顺序处理同一 shard 的各 arm；噪声由 prompt_id 与 seed 决定，不能依赖 worker 编号或执行顺序。

生成完成并释放模型后再运行 scorer/teacher，避免显存竞争。计时固定硬件、分辨率、batch、精度与 CFG；先 warmup 3 次，再测至少 20 次，包含最终 VAE decode，不含模型加载和磁盘写图。另报纯 denoiser 时间与吞吐。普通计时关闭 probes、save-readouts 与逐轮 suffix。

这里没有提供新算子的执行命令，因为对应 mode/训练入口尚未实现。配套 YAML 是待实现 runner 的实验合同，不是可以交给现有脚本直接运行的配置。不得用当前 anchored 模式的命令声称已经执行本方案。

## 9. 来源索引

本机根目录：/Users/zyd/Documents/LCoT-codex。

- BAGEL：refs/Bagel，commit 056b5fd51a88c1eb4547318609e25d40080fcf87。
  - inferencer.py:40、62、263：上下文更新与原生 planning cache。
  - modeling/bagel/bagel.py:101、232、267、757：训练目标、位置、文本 prefill、GEN flow。
  - modeling/bagel/qwen2_navit.py:499、559、757：专家 Q/K/V、缓存合并、native residual。
  - data/data_utils.py:13、data/dataset_base.py:449：噪声块掩码。
- [Training-Free Hidden-State Refinement](https://arxiv.org/pdf/2608.29160)：完整正文/附录；Eq.3、Algorithm 1、Table 5、Table 14。
- [Looped Diffusion Transformer](https://arxiv.org/pdf/2609.40305)：完整正文/附录；§2、§2.1、Appendix A.1/A.3。
- SLVR：refs/slvr，commit 660320ea0cedf90d9564ebd4ee7302126234203f。
  - [论文](https://arxiv.org/pdf/2605.19342)：v2，§4.2–4.3、Table 4。
  - scripts/finetune_slvr_stage1_7b_viscot.sh：freeze_llm=False、slvr_text_head=True。
  - src/train/monkey_patch_forward_slvr.py:554、657：目标特征注入与下一位置回归。
  - src/trainer/slvr_trainer.py:263：CE 与视觉/语义 loss 的实际组合。
  - src/model/qwen_slvr_model.py:603：latent decoding。
  - 论文 PDF SHA256：6232c978e3f3d6465ad05adfb494bfc9919a8bdae7ba882d3decd60fbbc670d9。
- [UNO 本地论文](../../refs/UNO/2605.05781v1.pdf)：§3.3、Table 4/6/7/9、Appendix A/D。refs/UNO 中没有代码，未声称审计其实现。
  - SHA256：167eae8f64235517918cd39d305922881c70f399ca6e537a25261492e6b6f81c。
- 本项目实施接口核对基于 commit 303703174182abda7c352f42c40a9c954e6045c1；未读取其他路线实验作为机制依据。

## 10. 当前完成状态

已完成：来源核对、机制与掩码定义、免训练测试次序、数据/统计/成本协议、训练准入与退出条件、实施清单。

尚未完成：新算子实现、H200 parity、任何本方案质量评测、任何本方案训练。当前不能宣称该内部 Memory 循环已产生语义增益。
