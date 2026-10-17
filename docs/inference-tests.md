# Inference Tests

更新：2026-10-15。本文根据当前训练与推理代码制定测试方案。入口状态来自代码核对；新增 benchmark 测试尚未执行。

[README](../README.md) · [原理](design.md) · [训练与历史实验](experiment.md)

## 1. 各阶段分别回答什么问题

测试沿训练顺序推进：先验证文字反馈接口，再验证状态读取、目标比较、多次反馈，最后验证生成侧适配。每阶段保留上一阶段的固定测试集，以观察能力变化。

| 阶段 | 训练作用 | 必要推理测试 | 主要结果 |
|---|---|---|---|
| C0 | 校准原生 UND 文字读出 | 原生自由复述 | 读出接口是否保留事实 |
| 历史 C1 / 文字教师 | 将文字压入 16 槽，并建立生成响应 | 正确文字与自动反思的 Text / Text-memory 对照 | 反思质量、压缩损失、GEN 消费效果 |
| C2-observe | 从 GEN KV 获取图像信息 | 分噪声自由描述、关闭与交换来源 | 描述正确性与来源依赖 |
| C2-feedback | 从 P+GEN 形成目标相关修正 | 纠错/保持文字读出、单次反馈终图、完整生成 | 修正是否正确，正确内容是否保留 |
| C3-loop | 根据上一遍 GEN 的新来源再次反馈 | 一次/两次反馈、New/Old/Fixed 对照 | 新反馈是否减少剩余错误 |
| 后续 GEN 适配 | 学会消费可靠 memory | 固定同一 memory，比较适配前后 GEN | 消费能力与端到端终图收益 |

Observe 的主结果是文字读取；Feedback 和 C3 的主结果包含终图。NLL、速度差异、图像变化各自记录，终图事实与质量评价决定生成效果。

## 2. 当前入口与部署前需要补齐的部分

### 2.1 已实现的能力

| 入口 | 当前行为 | 用途 |
|---|---|---|
| `visual_cli source-review` | Correct / Source-off / Source-swap，自由描述；writer 读取 GEN 来源 | Observe 的首要测试 |
| `visual_cli eval`，Feedback/C3 | 固定状态内计算反馈，消费一次最终速度，然后完成原生采样后缀 | 隔离当前节点的反馈效果 |
| `visual_cli sample` | 从纯噪声完成独立轨迹，默认前 20 次更新逐节点写入反馈 | 完整生成的已有实现基础 |
| `feedback_cli` | 完整 Base → 文字反思 → 固定 memory → 相同初始噪声重生成 | 显式文字参照 |
| `feedback_cli --text-control` | 增加原生追加文字与对应位置布局对照 | 分解文字压缩和 GEN 消费 |
| `feedback_audit` | 核对显式运行的初始噪声、窗口及产物 | 显式产物审计 |
| `score_explicit_feedback.py` | 读取显式 16 worker 目录，用既有 GenEval2 问题评分 | 当前显式产物的评分适配器 |

显式路径只使用完整生成后的文字反思。第二次生成复用第一次的精确初始噪声，窗口内消费固定文字 memory。

### 2.2 统一 benchmark 前的必要修改

以下记录当前接口状态及待实现的工作。

| 项目 | 当前代码事实 | 部署要求 |
|---|---|---|
| Checkpoint 加载 | `initialize_student` 同时收到 C1 `writer_init` 和视觉 checkpoint 时，采用 C1 writer 与视觉 query | 正式推理只加载待测 checkpoint 的 writer/query；独立教师使用 `--text-checkpoint` |
| 浅层脚本 | 从视觉 checkpoint 加载 writer/query；可选 `--text-checkpoint` 作为独立冻结教师；元数据记录训练阶段与当前来源协议 | 正式模型与 NPU 验证待执行 |
| 轮次参数 | `visual_cli sample` 固定输出 Base、一次反馈、两次反馈；该分支没有使用 `--rounds` | 统一为明确的 `feedback_updates`，每个输出记录实际次数 |
| 时间窗口 | `complete_sampling` 只支持从开始到 `loop_active_updates`；显式入口只支持 early 窗口 | 增加按实际 t 选择任意窗口的共同调度，保留 early 作为一种选择 |
| Observe `eval` | 按 `flow_supervision` 资格记录 flow 指标；普通 Observe 样本保留文字评估，flow 指标为空 | 读取对照可使用 `eval` 或 `source-review` |
| Feedback 文字读出 | `evaluate_feedback` 输出终图，尚未保存各臂、各遍的自由反馈文本 | 增加反馈文本、保持判断和对应状态记录 |
| 逐遍产物 | C3 可生成 New/Old/Fixed 终图，尚无统一的逐遍语义与来源报告 | 保存每遍来源、memory 与自由反馈，关联同一节点 |
| 显式规模 | `explicit_feedback_npu.sh` 固定抽取 16 条；CLI 本身可以处理多条输入 | 全量 benchmark 使用完整清单分片，逐条覆盖所有 prompt/seed |
| 统一评分 | 既有评分器按显式目录结构与 prompt 关联，配对键没有 seed 维度 | 导出统一图片清单，按 ID+seed+配置配对，再接各 benchmark 评分器 |
| 成本统计 | `complete_sampling` 计入真实额外 GEN 重算；浅层日志主要记录条件层调用 | 用实际 GEN/UND 层调用和端到端时间统一计量 |

当前 `sample` 的独立轨迹与 `eval` 的单节点后缀是两种不同测试。Benchmark 主表采用完整轨迹；单节点后缀用于分析反馈作用的位置。

## 3. 共同推理协议

### 3.1 权重、来源与轮次

- 正式推理加载 checkpoint 中训练后的 writer 与视觉 query，配置从该 checkpoint 导出。记录模型、checkpoint、配置和代码版本。
- Observe writer 直接读取 GEN KV。受控加噪图的 GEN 使用中性 prompt；真实生成轨迹保留其原始 prompt。
- Feedback writer 读取目标 prompt KV 与当前 GEN query KV。Source-off 只去掉 GEN 来源，保留目标 P。
- GEN query KV 保留图像 query 与原生边界 token。目标 KV 与 memory KV 分别处理。
- 每次反馈从同一中性种子重新写入。每个外层节点更新一次图像状态，使用该节点最后一遍的 CFG 速度。

统一使用以下轮次名称，保留原始输出中的历史名称作为元数据：

| 名称 | `feedback_updates` | 当前状态内的条件 GEN 遍数 |
|---|---:|---:|
| Base / R1 | 0 | 1 |
| 一次反馈 / R2 | 1 | 2 |
| 两次反馈 / R3 | 2 | 3 |

完整 GEN 路径与浅层中间遍分别报告。它们使用不同计算图，分别核对来源与最终速度后再比较效果。

### 3.2 配对条件

同一测试单元固定 prompt、seed、精确初始噪声、分辨率、原生采样节点、shift、CFG 和图像条件。单节点对照进一步固定 x_t 与实际 t。

Base 使用原生位置。Layout 为新增条件保留相同位置，但屏蔽该条件。Text 与 Text-memory 各自使用长度对应的 Layout。所有结果同时报告相对 Base 和对应 Layout 的变化。

来源交换保持接收方 P、x_t、t 和图像条件，只替换 writer 读取的 GEN 来源。供体使用相同 P、t 与 seed，并具有不同图像事实。交换后同时按接收方与供体事实评分：文字是否跟随供体，是读取来源的直接证据。

GEN KV 本身经过 prompt 条件化。相同 P 的来源交换用于区分目标先验与图像证据。

### 3.3 噪声与时间窗口

原生定义为 `x_t=(1-t)x0+tε`。t 越大，噪声越高；采样从 1 走向 0。

| 测试层次 | t 或窗口 | 使用方式 |
|---|---|---|
| 当前训练范围 | 0.05、0.15、0.25 | Observe/端点 Feedback 的主要读取测试 |
| 扩区诊断 | 0.5、0.75、0.9、0.95 | 分别报告范围外表现，决定后续课程 |
| 纯噪声 | 1.0 | 目标先验与无图像证据对照 |
| 当前范围内的完整生成窗口 | 实际节点满足 `0.05≤t≤0.25` | 首先测试已训练范围内的生成作用 |
| 早期生成窗口 | early10、early20 | 测试高噪声生成作用，与低噪声结果分表 |

50 节点、shift 3 对应 49 次更新。当前低噪声范围内的更新索引为 **45、46、47、48（从 0 开始）**，t 分别约为 0.210526、0.163636、0.113208、0.058824。第 44 个节点约为 0.254237，位于当前训练上限之外。

early10/20 从 t=1 开始，结束边界分别约为 0.921260、0.813084。它们主要处于高噪声范围。

读取测试可以直接使用连续 t。采样测试使用原生节点，保存 requested_t、actual_t 和更新索引。窗口按照实际 t 判断；最近节点映射后的值可能跨越训练上限。

## 4. 文字教师与显式参照

这一组先回答：正确文字能否产生正确修改，以及压成 16 槽后保留了多少效果。

### 4.1 必要对照

固定同一个 Base、同一份修正文字和同一个消费窗口，生成：

| 对照 | 新增条件 | 回答的问题 |
|---|---|---|
| Base | 无 | 原生成结果 |
| Layout-text | 仅文字对应的位置移动 | 原生文字的位置作用 |
| Text | 原生追加修正文字 | GEN 能否消费这条文字修正 |
| Layout-memory | 屏蔽 16 槽 memory | memory 位置作用 |
| Text-memory | P+修正文字压入 16 槽 | 压缩后能否保留修正效果 |

文字反馈来源分为 `bagel_auto` 与 `manual_checked`。同一批 Base 同时使用两类反馈，以比较反思质量。核对反馈只描述 prompt 要求与 Base 的可见差异，并明确需要保留的正确内容。

目标已满足的样本计入自动保持判断。当前显式实现对 `satisfied=true` 直接返回 Base；这一组衡量反思器的保持决策。GEN 消费后的稳定性另外用非空、语义一致的保持描述测量。

### 4.2 报告与解释

- 自动反思：目标冲突率、错误保持率、过度修改建议率。
- 正确 Text：相对 Base/Layout-text 的事实修正、事实损伤与图像质量。
- 正确 Text-memory：相对 Base/Layout-memory 的同类结果，以及相对 Text 的差值。
- 自动 Text-memory：保留同一组指标，量化实际显式 pipeline 的效果。

自动与核对反馈的差异用于分析反思。Text 与 Text-memory 的差异用于分析压缩及其消费接口。核对后的 Text-memory 是固定 GEN、文字教师和窗口下的**经验参照**；后续训练可能超过这组结果。

## 5. C2-observe：先测是否读懂，再测迁移

### 5.1 受控状态读取

从 held-out 图像构造加噪 latent，GEN 使用中性 prompt，writer 只读取 GEN KV。UND 通过 memory 自由输出描述。

最低必要对照为训练前初始化与 Observe checkpoint；每个 checkpoint 运行 Correct、Source-off、Source-swap。每个状态保存接收图、供体图、自由描述、事实标签与来源哈希。

主指标按对象、属性、数量、关系分别统计：

- 已知事实召回率：正确提到的已知事实 / 全部已知事实。
- 断言精确率：正确的明确断言 / 全部可核对的明确断言。
- 不确定事实的错误断言率，以及已知事实上的弃答率。
- 来源增益：Correct 与 Source-off 的事实分差。
- 交换跟随：Source-swap 描述与供体事实的匹配，以及与接收方事实的变化。

NLL 辅助观察读出学习。自由描述同时检查遗漏、错误与弃答，避免将统一输出 unknown 记为读取成功。既有 `score_facts` 只覆盖预设词表，语义同义、额外断言和关系方向由独立评分或人工标注补充。

### 5.2 高噪声的信息范围

干净图 caption 在低噪声区间提供恢复目标。随着噪声增大，单独报告“对原图事实的恢复”与“对实际状态的描述”。后者采用绑定该 x_t 的状态标签，并记录标签来自何种观察方法。

原生条件 x0 预测图与完整后缀终图分别表示局部预测和轨迹结果；各自的 caption 使用对应标签类型。它们用于分析 GEN 当前提案，标签含义与原始干净图描述分开记录。

中性 prompt、相同 seed 与相同 latent 形状下，t=1 的 x_t 与原图无关。该组测量模型是否编造来源特有内容；按原图 caption 恢复率计算的结果单列为先验诊断。

当前 `source-review` 接受 `t<1`。纯噪声对照需要在统一读取入口补齐。

### 5.3 真实生成轨迹迁移

从原生 Base 的 prompt-only 生成保存若干原生节点。Observe 读取这些节点的 GEN KV，输出描述。GEN 保留实际 prompt，writer 仍只读 GEN 来源。

这组检查“受控加噪图 → 实际生成状态”的迁移。选择目标要求相同、实际生成结果不同的样本，比较描述是否反映生成差异。先覆盖当前低噪声范围，再扩展至高噪声。

受控状态与真实轨迹分别汇总。Observe checkpoint 的终图变化列为探索性诊断；其阶段结论来自信息读取。

## 6. C2-feedback：修正与保持同时测试

### 6.1 两类必要样本

| 类型 | 当前状态 | prompt | 正确行为 |
|---|---|---|---|
| 需要修正 | 编辑对的 A，或不满足目标的真实生成状态 | 目标 B 的要求 | 修正差异，保留已正确内容 |
| 已经满足 | 编辑对的 B，或满足目标的真实生成状态 | 同一目标要求 | 保持 |

编辑对 A/B 使用相同 family、P、噪声 seed 与 t。A 的局部编辑标签和 B 的保持标签分别绑定其状态。真实轨迹按该节点所对应的状态/提案标签测试。

### 6.2 UND 的文字反馈

对 Observe parent 与 Feedback checkpoint，各运行 Correct、Source-off、Source-swap，自由读出文字反馈。Correct 使用 P+GEN；Source-off 保留 P。使用同一目标下“一项需要改、一项已正确”的供体对，检查反馈是否随状态切换。

另外设置同状态、不同目标的少量对照，验证目标相关性。目标变化后重新计算该目标条件下的 GEN 来源，再写入反馈。

主要指标为修正方向正确率、保持判断准确率、正确内容的错误修改建议率，以及反馈与目标冲突率。C2-observe 的固定读取测试继续运行，记录信息获取能力是否保留。

### 6.3 GEN 消费后的终图

先做单节点对照：共享同一 x_t、P、t，分别计算 Base、Layout、Source-off、Source-swap、正确 Text-memory 与连续反馈；消费一次最终速度，再完成原生后缀。已有 `evaluate_feedback` 提供其中大部分对照。

随后做 prompt-only 完整生成：从同一精确噪声开始，逐节点写入一次反馈。主表至少保留 Base、Layout、连续反馈、Source-off，以及显式自动/核对 Text-memory 参照。受控编辑状态的后缀结果与 prompt-only 完整生成分别汇总。

开始部署时先完成训练范围内的低噪声窗口，再单独运行 early10/20。固定文字参照也使用相同消费窗口，保证比较发生在相同时间范围。

通过比较文字反馈与终图，可以定位两类问题：反馈方向已经错误；反馈方向正确，但 GEN 的实际修改错误或不足。

## 7. C3-loop：验证新来源是否带来额外纠错

先固定同一个状态测试，再运行完整生成。当前 C3 配置为两次反馈，即三遍条件 GEN。

| 对照 | 第一份 memory | 第二份 memory | 作用 |
|---|---|---|---|
| 一次反馈 | 读取初始 GEN 来源 | 无 | 前一阶段参照 |
| New | 读取初始 GEN 来源 | 读取消费第一份 memory 后的新 GEN 来源 | 正式两次反馈 |
| Old | 读取初始 GEN 来源 | 再读取初始 GEN 来源 | 旧来源对照 |
| Fixed | 读取初始 GEN 来源 | 直接复用第一份 memory | 固定内容对照 |
| Swap | 读取初始 GEN 来源 | 读取供体的新 GEN 来源 | 状态内容依赖 |

同状态、同 t、同种子且 writer 确定时，Old 重写与 Fixed 复用应得到相同或数值一致的结果。它们共同用于核对“重复计算”与“更新内容”，统计时按各自作用解释。

保存每遍来源哈希、memory 哈希、自由反馈文本、条件速度，以及消费该遍速度后的独立终图。各遍终图从相同 x_t 分别继续原生后缀，用于比较当前节点的修改；完整生成另存每个节点的反馈记录。

核心报告为：

- New 相对一次反馈的新增修正、回退和视觉质量变化。
- New 相对 Old/Fixed 的差值，及对应的反馈内容变化。
- 第一遍已改正确的事实在第二遍后的保持率。
- 剩余错误是否被继续修正，是否出现来回修改同一事实。
- 增加一次反馈所需时间与实际 GEN/UND 层调用。

完整生成另设跨外层节点复用固定 bank 的对照。它与同一节点内的 Fixed 分别命名、分别记录；前者测量是否需要随采样状态更新，后者测量是否需要当前节点的第二次反馈。

## 8. 后续 GEN 适配：固定反馈内容再比较消费

这一阶段尚未训练。推理测试先冻结同一 UND/writer，固定 memory 内容、状态与时间窗口，比较原 GEN 和适配后的 GEN。

必要对照为 Base、Layout、正确 Text、正确 Text-memory、连续 memory；自动反馈与核对反馈分别报告。额外测量无 memory 的原生生成质量，以及正确内容上的保持行为。

先用核对后的 Text-memory 判断 GEN 是否学会按反馈修改，再用连续 memory 判断读取与消费组合后的收益。最后运行完整端到端轨迹。

若 GEN 参数改变，其内部 KV 分布也可能改变。增加一组 Observe/Feedback 读取测试，对比适配前后 GEN 来源；这组用于测量冻结 UND 在新来源上的兼容性。

## 9. 测试数据与指标

### 9.1 三份固定清单

| 清单 | 内容 | 用途 |
|---|---|---|
| 读取集 | 图像、事实、同 prompt 的交换对、不同噪声状态 | Observe 与 Feedback 的语义测试 |
| 反馈校准集 | 修正/保持 family、Base、核对文字、窗口和种子 | 选择 checkpoint 与推理设置 |
| 正式 benchmark | benchmark 提供的完整 prompt/条件图及官方问题或目标 | 冻结设置后的最终评价 |

校准集起始使用既有计划中的 256 个 family。覆盖对象、属性、数量、关系和文字；修正/保持均包含。正式测试清单固定来源划分与 family，记录各清单的划分方式。当前训练基座保留重复标注，不做图像内容去重。公开 benchmark 另外保存版本与完整输入清单。

配置选择在校准集完成。正式测试对所有方法使用固定窗口、轮次和相同种子列表；每个 prompt 的全部种子均纳入汇总。生成 seed 与样本选择 seed 分开记录。

### 9.2 终图指标

将目标拆为可核对事实。对相同 prompt/seed 的 Base 与反馈终图，统计：

- **修正率**：Base 中错误且反馈后正确的事实数 / Base 中错误事实数。
- **损伤率**：Base 中正确且反馈后错误的事实数 / Base 中正确事实数。
- **严格成功率**：需要修改的事实全部改对，同时指定保持事实全部保留的样本比例。
- **保持成功率**：原本已满足目标、反馈后仍满足的样本比例。
- **目标总分**：各 benchmark 自身定义的原始指标与分项。
- **图像质量**：伪影、结构完整性与盲评偏好；局部编辑另外评价非目标区域保持。

分母为零时记录数量并将该项标为不适用。修正率与损伤率同时给出原始计数。完整目标的事实转移相对 Base 计算；附加 Layout 对照用于分离位置作用。

图像像素差、memory 距离与速度范数反映响应程度。修改方向通过目标事实、保持和质量指标评价。

### 9.3 稳定性、评分与成本

每个 prompt 报告逐 seed 分数、均值和方差，再汇总整体结果。置信区间按 prompt/family 成组重采样，保留同组内所有 seed 与对照。目标变化、反馈来源、时间窗口和状态类型分别分组。

使用 benchmark 的既有评分协议保存原始分数。新增事实修正/保持评价另存附加报告。独立视觉评分器与 BAGEL 自动反思分开记录，并人工复核一部分评分分歧。文字任务使用相应文字识别或人工答案核对。

成本包含原生生成、反思、文字预填充、writer、反馈生成与解码。显式方法计入第一次 Base 和第二次生成；连续方法计入所有内层遍。共享 Base 的研究对照分别记录共享计算量与单样本独立部署成本。

为检验相同计算量下的收益，增加原生 BAGEL 的候选生成或采样预算对照。候选选择使用事先固定、部署时可用的规则，并计入选择成本。使用真值评分选出最好图片的结果单列为 oracle 分析。

## 10. Benchmark 接入约定

以下是待接入统一 runner 的约定；当前已有输出通过适配器转换。

### 10.1 输入

推理最小输入包含 `sample_id`、`prompt`、`noise_seed`。条件生成增加 `condition_images`。读取/状态测试增加 `family_id`、图像或状态文件、标签类型、实际 t、事实与 `swap_id`。

目标图、核对反馈与原始标注由评价清单持有。正式自动方法的生成输入采用该方法部署时可获得的字段。使用核对反馈或目标图的对照带明确的参照标记。

统一 runner 按完整输入清单在 16 卡间分片，每个 worker 独立加载模型并处理分配的样本。当前推理以数据并行为主，16 卡处理不同样本；相同 seed 的所有方法保留一致输入。

### 10.2 图片输出

每张终图对应一条记录。最小配对键为：

```text
(benchmark_version, sample_id, noise_seed, method, checkpoint_hash, window, feedback_updates)
```

记录图片路径/哈希、初始噪声哈希、prompt、条件图哈希、采样设置、实际消费节点、来源策略、反馈来源、逐步记录路径和耗时。评分结果使用同一键关联，保留 benchmark 原始 ID。

推理、终图评分、统计汇总分别运行。推理输出完整图片清单；评分器消费图片清单；汇总检查每个样本/种子/方法的覆盖并输出配对结果。

### 10.3 最小交付

- `run.json`：模型、代码、权重、输入版本、采样与运行设置。
- `images.jsonl`：所有终图及配对键。
- `states.jsonl`：状态、来源、memory、轮次与时间窗口记录。
- `readouts.jsonl`：Observe 描述或 Feedback 文字，以及对应标签作用范围。
- `scores.jsonl`：每图、每问题或每事实的原始评分。
- `summary.json`：分阶段、分窗口、分 seed 的汇总、配对差值与成本。

其中命名是统一导出的目标约定。既有 `source_review.json`、`final_images.jsonl`、显式比较 JSON 与逐节点 JSON 保留，由适配器转换。

## 11. 现在可以使用的命令

### 11.1 Observe 的分噪声来源读取

在最新代码的远端仓库根目录执行。先将 `CKPT` 和 `READING_DATA` 设置为本次 checkpoint 与准备好的读取集。诊断集使用 `--data`，训练全量索引使用 `--corpus`。

```bash
export MODEL_PYTHON=/root/venvs/bagel-NPU/bin/python
export MODEL_PATH=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export CKPT=/absolute/path/to/observe/best.pt
export READING_DATA=/absolute/path/to/observe_reading_test.jsonl
export RUN=/root/outputs/observe_reading_new
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN"
"$MODEL_PYTHON" - "$CKPT" "$RUN/checkpoint_config.json" <<'PY'
import json, sys, torch
saved = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
with open(sys.argv[2], 'w') as stream:
    json.dump(saved['config'], stream, indent=2)
PY

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
OMP_NUM_THREADS=4 \
"$MODEL_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=16 \
  -m qwen_latent_cot.diagnostics.visual_cli source-review \
  --model "$MODEL_PATH" --checkpoint "$CKPT" \
  --config "$RUN/checkpoint_config.json" \
  --data "$READING_DATA" --split test \
  --review-t 0.05 0.15 0.25 0.5 0.75 0.9 0.95 \
  --noise-seeds 0 1 2 3 \
  --device npu --distributed --output "$RUN/results"
```

该命令加载 Observe checkpoint 自己的 writer/query。结果位于 `results/source_review.json`，各 rank 保存自由描述和图像。交换描述的供体事实评分需要在这些记录上追加。

读取集的最小双向交换例子如下。替换真实图像路径，事实词表按人工核对内容扩充：

```jsonl
{"sample_id":"cup_red","scene_id":"cup_red","family_id":"cup_red","split":"test","task":"observe","label_kind":"target_clean","label_type":"clean","verified":true,"prompt":"An image.","image":"/absolute/path/to/red_cup.png","caption":"A red cup.","facts":[{"category":"attribute","aliases":["red cup"],"alternatives":["blue cup"]}],"swap_id":"cup_blue"}
{"sample_id":"cup_blue","scene_id":"cup_blue","family_id":"cup_blue","split":"test","task":"observe","label_kind":"target_clean","label_type":"clean","verified":true,"prompt":"An image.","image":"/absolute/path/to/blue_cup.png","caption":"A blue cup.","facts":[{"category":"attribute","aliases":["blue cup"],"alternatives":["red cup"]}],"swap_id":"cup_red"}
```

### 11.2 显式文字参照的小批量核对

```bash
export MODEL_PYTHON=/root/venvs/bagel-NPU/bin/python
export MODEL_PATH=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export TEXT_TEACHER=/absolute/path/to/validated_c1/best.pt
export PROMPTS=/absolute/path/to/benchmark_prompts.jsonl

bash scripts/explicit_feedback_npu.sh \
  /root/outputs/explicit_reference_new --text-control
```

脚本的 prompt 行使用 `prompt_id` 与 `prompt`，固定选择 16 条，自动反思，运行 early10/20。全量运行接统一清单分片；核对反馈通过 `feedback_cli --prepare-only` 生成模板，再填写反馈并运行，详见 [显式运行入口](experiment.md#61-显式固定-memory-重生成)。

## 12. 部署顺序

1. 建立完整 prompt/seed 清单、原生 Base、统一图片输出和评分接入。
2. 运行正确 Text/Text-memory 与自动反思参照，校准文字反馈接口。
3. 对 Observe checkpoint 运行低噪声 Correct/Off/Swap，再做范围外与真实轨迹读取。
4. 补齐任意时间窗口、正确权重加载与 Feedback 自由读出；运行修正/保持的单节点对照。
5. 运行 Feedback 完整生成，先测训练范围内窗口，再测 early10/20。
6. 运行 C3 的一次/两次反馈和 New/Old/Fixed 对照，追加实际计算预算对照。
7. 固定校准所得 checkpoint、窗口与轮次，覆盖正式 benchmark 的全部样本与种子。

每轮交付同时包含事实修正、正确内容保持、图像质量、逐 seed 稳定性与计算量。各阶段的结论对应其直接测试的能力。
