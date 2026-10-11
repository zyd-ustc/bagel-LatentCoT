# BAGEL 独立 Memory：caption 接地与生成消费的分阶段训练

日期：2026-10-11。状态：设计分析；未运行训练、未复查用户新实验原始输出。本文修订此前将legacy_layerwise固化为唯一训练拓扑的建议。

## 1. 结论

独立的CLS式Memory序列值得重新做，但caption只解决“Memory是否携带可读语义”，不自动解决“GEN是否使用这些语义”。建议把训练拆为两个接口：**GEN状态→Memory的观察接口**，以及 **Memory→GEN的生成条件接口**。原prompt始终保留，原始BAGEL参数首轮全部冻结。新增参数和逐项实验负责建立这两个接口。

CLS在此指若干可学习摘要槽位，不是BAGEL已有的特殊token。不要假定一个CLS能容纳细粒度计数和关系，也不要未经训练就把槽位当成原生文本。可从小规模多槽位开始，容量由对象/属性/关系覆盖率选择，不预先声称某个数量最佳。

用户观察：独立Memory在training-free时生成侧不敏感；prompt初始化/替换后有作用；独立逐层与连续UND更新的人工结果差异不大；joint-micro在首轮后出现噪点。这些观察说明需要重新考虑接口，而非固定某一种循环。它们尚不能区分信息未写入、写入但GEN忽略、位置/尺度不匹配或评价不敏感，也不能单独定位joint-micro失败原因。

## 2. 源码能证明什么

审计版本：`loop-layer-NPU@42e71148826011b74473a170c4ab62072d6c70ab`，独立快照位于 `artifacts/loop_layer_source_20261011/npu/`。

|源码事实|结论与范围|
|---|---|
|`forward_cache_update_text`：token embedding→因果UND→逐层KV|文本prefill是预训练形成的语义接口；不是单个最终hidden向量|
|`forward_cache_update_vit`：ViT→connector+position→UND→KV|文本不是唯一进入UND的表示|
|`forward_cache_update_vae`：VAE→vae2llm+time+position→GEN→KV|图像GEN专家也能生成可缓存上下文|
|`PackedAttentionMoT.forward_inference`合并历史KV，不要求KV来自与query相同的专家|UND/GEN有共享attention接口；形状可用不代表语义分布匹配|
|训练mask的`noise`段只允许同段query读取|当前带噪图像→独立UND Memory是新增可见关系，不能由干净视觉上下文读取能力推导|

重要限定：mask说明当数据段标为`noise`时的行为。本快照没有完整预训练数据流水线，不能据此声称所有BAGEL预训练任务从未有任何UND/GEN交互。

来源：[bagel.py](../artifacts/loop_layer_source_20261011/npu/qwen_latent_cot/bagel/modeling/bagel/bagel.py)、[qwen2_navit.py](../artifacts/loop_layer_source_20261011/npu/qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py)、[bagel_utils.py](../artifacts/loop_layer_source_20261011/npu/qwen_latent_cot/bagel/modeling/bagel_utils.py)。

### “文本latent唯一通用”应怎样表述

更准确的是：**原生文本prefill产生的各层KV，是已经被文本理解与图像生成两条路径使用的共享语义条件。** 这使它适合做teacher或接口校准参照，但它不是唯一可消费表示。

维度相同、最终hidden相近都不足够。消费端使用的是不同深度、位置、上下文下的K/V；同一最终hidden不能直接复制给28层。Memory若经过UND连续处理，应在每层attention输入处生成该层KV，而不是把最终Memory重新投影到所有层。对槽位与caption token一一做MSE也没有天然依据，二者长度、角色和上下文不同。

### UND读GEN的分布差异

区分两种实现：

1. UND query直接读取GEN专家已经投影好的K/V。当前legacy是这个接口；主要新增因素是噪声时间、生成状态、注意力可见性和循环语境。
2. 把GEN原始hidden当作UND token，经UND norm/K/V重新投影。这里还新增专家表示转换问题，不能因为hidden_size相同就直接混用。

首版优先明确选一种，保存同层GEN输入hidden及其原生KV作为对照。若训练桥接模块，使用小型、分层/时间感知的读取支路；不改原GEN投影，不混用已经带RoPE的K与未带RoPE的hidden。需要记录层号、t、位置、专家和归一化位置。

## 3. joint-micro为何不能凭1/K认为安全

[Training-Free Hidden-State Refinement](https://arxiv.org/pdf/2608.29160)在固定条件下重复原层残差，步长为1/K；K=1恢复原层，K>1没有等价保证。它验证的是RAE类模型，不包含本地同步GEN/UND Memory耦合。

本地joint-micro还替换条件、同时更新两路状态并改变suffix交互。因此不是只把论文的步长换成BAGEL版本；K1整条路径已经不等于Base。对于非线性block，`K次1/K残差`与`1次完整残差`也不等价。

用户报告多轮噪点是停止扩大该方案的直接依据，但具体原因未确定。若以后排查，分别比较固定原prompt的纯GEN微循环、固定GEN反馈的Memory更新、同步耦合；检查同x_t/t的velocity范数/方向、CFG前后差异、位置与mask。首版caption框架不同时采用同步GEN微循环，以减少变量。

## 4. Caption任务的设计：必须形成信息瓶颈

建议先建立如下单向计算：

```text
冻结原生GEN(x_t,t,p) → 停止梯度的GEN特征
                              ↓
                      新增观察桥接模块
                              ↓
               独立Memory槽位 → UND逐层处理
                              ↓
                         Memory各层KV
                              ↓
                 冻结UND解码器 → caption/短问答
```

观察模块可以读取prompt辅助定位，但第一轮接地校准应使用固定中性指令，或在相同prompt下提供不同图像。目标是逼迫Memory反映图像证据，而不是复述指令。

caption decoder只能访问Memory KV、固定任务指令以及已经生成的答案前缀。不能直接读取图像KV、GEN KV或包含目标描述的prompt。Memory writer不能读取caption答案token。训练可使用标准teacher forcing，但必须按因果顺序移位目标，并做自由解码验证；否则常见语言模式可能让loss降低而Memory无用。

`L_cap = -sum_i log p_frozen(y_i | y_<i, M)`。

冻结解码器参数仍需要保留到Memory输入的梯度。`generate_text`是推理接口，不应直接作为训练入口；使用共享模块的可微教师强制前向。不要通过解冻caption decoder让它适应任意编码，从而掩盖Memory缺少共享语义的问题。

### 两种caption标签不可混用

- **观察标签：** 描述当前可见预测图，如 `decode(x_t-t*v_base)` 中确实出现的对象和关系。看不清时允许unknown。可用冻结BAGEL原生ViT caption作为候选，再抽查/校验。这是有损图像代理，不是GEN hidden的完整真值。
- **目标标签：** 描述配对干净图x0或prompt希望生成的场景。这是在训练恢复目标语义，属于去噪/规划监督；不能拿它的准确率证明Memory看出了当前错误。

接近纯噪声端时，信息不足；不能要求Memory确定地报告目标细节。GEN特征本来就条件于prompt，所以仅移除decoder的prompt仍不能排除间接复述。需要固定prompt、改变噪声/图像，并让可见结果与prompt发生受控不一致来检验。

建议先训可观察的对象、属性、计数和关系短描述，再考虑“目标与观察差异”。描述当前错误和建议如何修复是不同任务；避免把“现在两个对象”直接作为生成命令强化错误。

## 5. 用BAGEL文本prefill做桥梁，而非用它替代验证

分两级建立teacher：

**文本接口校准。** 正确caption走原生文本prefill，作为显式文本参照。另训练压缩模块把这份文本信息编码成固定长度Memory，再由冻结caption decoder重建caption。这一阶段允许caption进入压缩器，因为任务就是文本自编码；但不能将此阶段成绩当成视觉接地。预训练文本embedding可作为槽位初始化参照，避免全相同槽位，但任意soft token仍属于新增输入。

**视觉接口迁移。** 独立学生只读取实际GEN特征、t和允许的prompt；不能看到caption。它产生同容量Memory，接受caption CE，以及冻结decoder输出分布或经过验证的文本压缩teacher的弱蒸馏。教师接触答案是训练监督，学生接触答案是泄漏。评测严格用学生Memory。

若使用token级logit KL，teacher/student必须在相同答案前缀上比较。不要把不同长度caption的KV逐项硬对齐，也不要仅对最终hidden做对齐就宣称GEN接口完成。

## 6. 第二个瓶颈：让GEN确实使用独立Memory

Caption成功不保证GEN利用Memory。冻结UND query与冻结GEN query不同，attention投影和任务也不同。必须设独立的生成消费实验。

先保持原prompt KV完整，比较：

1. 原prompt，Memory完全关闭。
2. 原prompt＋已验证有用的显式文字补充条件，通过原生prefill进入模型。
3. 原prompt＋上述文字压缩出的独立Memory（oracle Memory，只校准接口）。
4. 原prompt＋从GEN状态预测出的学生Memory。

第2组只是诊断参照，并不保证一定比Base好。如果显式文字本身无帮助，不应把它当生成蒸馏teacher；它还改变了长度/位置，不能把2与3的所有差异归因于压缩。

如果2有效而3无效，先解决Memory生成读取接口；如果3有效而4无效，先解决观察/写入。不要一开始同时解冻GEN和UND，使失败不可定位。

### 保留prompt仍不足以保证等价

直接拼接Memory KV会改变softmax分母；即使V全零，原prompt/self贡献仍会被稀释。增加前缀也可能改变GEN RoPE位置。因此“零Memory”“零value”“原权重冻结”都不构成Base等价证明。

低风险起点是独立、可关闭的Memory读取支路：

`attention_output = native_attention_output + a_l(t) * memory_read(Q_GEN, M_l)`。

原native attention完全保留原prompt/self、mask与位置。新增支路的slot位置/编码需要明确设计与训练；不能称其为原生接口。输出回到GEN的方式应受小幅度约束，关闭时逐层恢复Base。优先训练Memory端投影与新增读取模块，保持原GEN Q/K/V/O、norm、MLP、time/position和输出头不动。

这是新模块方案，不是“完全不改变DiT行为”。若坚持不新增生成读取模块，则可测试原生attention拼接Memory，但应承认初始就有分布偏移，且训练可能持续不敏感；不能保证仅caption训练足够。

零初始化只选一个有效出口，例如输出门a=0而读取支路正常初始化，或输出投影为零但gate不为零。不能gate与输出投影都置零导致双方无梯度。精确零门控会使第一步writer没有生成loss梯度，可先单独训练接口门，再联合；caption监督仍可训练writer。

### 生成目标与消费证据

主任务保持BAGEL `x_t=(1-t)x0+t*eps, v*=eps-x0` 的conditional flow matching；加保真集Base速度约束和小残差约束。再按需保留caption辅助，避免生成训练让Memory语义漂移。

最重要的因果对照：正确Memory、同prompt不同图的错配Memory、打乱Memory和彻底关闭支路。布局、容量、位置、计算预算一致。要求正确Memory在相应语义上更好，错配Memory按可解释方向影响结果；仅速度变化或attention质量增大不算有用消费。

避免“模型只学新增参数，忽略M”：保持新增模块启用但换成固定无信息M，与输入依赖M比较。如果二者同样提升，收益可能来自普通adapter而非Memory。caption侧同样要比较空/错配Memory的自由解码和事实准确率，而非只看teacher-forcing loss。

## 7. 更新拓扑：先选容易校准的一种，再公平比较

|方案|优势|新增风险与建议|
|---|---|---|
|每层独立Memory|对应局部GEN特征，便于层级消融|各层语义不连续，同层重复是新分布；保留为比较组|
|一份Memory经连续UND层|便于复用prefill式处理和逐层输入KV|独立槽位和读GEN仍非原生；建议作为caption首轮实现|
|每轮从同一浅层Memory seed重启，读取新GEN反馈|避免末层hidden回到首层；轮间通过GEN传播信息|增加开销；R1通过后再试|
|末层hidden直接送回首层|实现直接|跨深度分布错位；首版不采用|
|GEN/UND同步micro-loop|两路共同迭代|用户已观察多轮噪点；暂不叠加到新训练架构|

首轮只做一次冻结GEN特征提取、一次Memory writer、一次条件重算。在多轮中，GEN每次完整正常计算，Memory读取其新状态；每个去噪步最终只推进一次。先冻结GEN特征训练观察，再用部署轨迹更新数据，最后才考虑循环内联合梯度。逐层与连续更新用相同槽位数、来源、caption任务、生成接口及近似算力比较；当前人工结果不足以判定哪一种长期更好。

## 8. 参数开放顺序与阶段准入

|阶段|训练参数|保持冻结|进入下一阶段的证据|
|---|---|---|---|
|A 文本/视觉caption接地|独立slot、GEN→Memory桥、writer专用小adapter|BAGEL全部原参数与caption decoder|自由caption/短问答依赖正确Memory；反事实图像变化能被读出|
|B GEN消费校准|Memory端读出投影、独立生成读取支路/门；初期可冻结A|原GEN与原prompt prefill|oracle M有用，错配/固定M不能获得同等收益，关闭支路恢复Base|
|C 单轮联合|上述新增模块|原BAGEL权重|学生M带来语义收益，保真集无明显退化|
|D 多轮与部署分布|仍优先仅新增模块|原BAGEL权重|多轮相对单轮有稳定净收益，非噪点/画风损伤|

UND adapter必须只在Memory writer调用启用；不能全局挂在共享UND层上，否则会同时改prompt prefill、caption decoder和GEN中的边界token。若A能力不足，先增加桥/槽位容量或writer专用LoRA；先不要解冻原UND。若B失败，先检查可见性、位置、attention竞争和表示接口，再考虑新增reader容量；先不要解冻原GEN。

以后若确需解冻，须有冻结方案的具体瓶颈证据、独立消融和能力保持数据。低学习率或低秩本身不保证先验不受损。特别保留time embedding、VAE输入/输出头、position和norm，不将它们当作首批调节旋钮。

本轮仅形成设计。最新结果的具体R/K、图片、权重和日志仍未核对；不会把joint-micro失败归因写成事实，也不会把caption可行性写成已验证质量收益。
