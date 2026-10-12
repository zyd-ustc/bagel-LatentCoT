# 实验

状态：C1已完成并保留C；下一步GEN颜色读取；蒸馏待实现。更新：2026-10-12。

[返回 README](../README.md)

## 1. 固定比较条件

Ascend默认16卡并行，使用 `loop-layer-NPU`。评测按样本分片；联合训练同步新增writer梯度，`effective_batch`表示全局batch。C0已完成16条短文本，结论见[验证](validation.md)。

L/C共同使用16槽位、全部UND层、512分辨率、相同新增参数量、来源、读出位置和预算。每层一次writer，Memory不跨去噪步，GEN只在一个节点读取一次。原BAGEL与原prompt保持冻结/保留。

记录模型/源码/tokenizer/VAE/config hash，设备、dtype、attention后端与随机流。按原图/场景族固定train/val/test；同图的噪声、caption改写与反事实变体不得跨split。

工程检查先用16样本。拟议数据预算：约2,000训练图、256验证图、256测试图，测试至少64对单属性反事实场景。规模是诊断起点，不代表训练充分。缩减时L/C同步缩减。

## 2. 状态与标签

|来源|构造|用途|
|---|---|---|
|受控加噪|`x_t=(1-t)x0+tε`，同图/seed各t复用ε|训练与噪声曲线|
|原生轨迹|保存原生采样的actual t/step/x_t|部署分布验证|
|文本oracle|caption原生prefill|文本压缩与消费接口校准|

测试t为`{0.05,0.25,0.50,0.75,0.90,1.00}`；训练只用前五桶，直接使用actual t，不二次sigmoid/shift。t=1为纯噪声负对照，不纳入语义恢复平均。

`y_clean`描述干净图像；`y_obs`描述`decode(x_t-t*v_base)`中人工确认的可见事实，允许unknown。两者分表，不将目标正确率当观察准确率。轨迹自动caption必须核验后使用。

固定中性prompt用于主视觉写入任务；原生完整prompt用于额外轨迹任务。即使decoder看不到prompt，GEN仍可能携带其内容，因此必须做相同prompt、不同图像/seed的反事实检查。

## 3. Caption与蒸馏阶段

1. **C0**：原生caption上下文能自由复述；用正确补充文本验证GEN探针协议，不预设文字一定改善图像。
2. **C1**：同预算L/C首跑已完成；保留C，L暂停。结果与选择范围见[验证](validation.md#4-c1结果与拓扑选择)。
3. **文本消费验证（下一步）**：固定C1-C，用第5节的颜色校准验证GEN读取；无需C2 checkpoint。
4. **后续训练**：文本消费成立后再投入C2；教师文本有效而Memory无语义影响时，先在C上实现响应蒸馏。C2-scratch等消融延后。

当前CE预算为每组5,000 optimizer steps、有效batch32、每250步验证；其余参数由[配置](../configs/memory_topology.json)绑定。蒸馏沿用共同预算，另记录query数、采样层和训练成本。checkpoint仅按验证集共同规则选取；待实现蒸馏需预先固定选择规则，不能沿用规则名却暗改含义。

C1首轮缩减预算：L/C依次各用16卡，100 steps、全局batch16、每50步验证、解码上限48 token。使用 `c1_text80.jsonl` 的64/8/8条train/val/test；同一场景颜色反事实成对绑定split。文本由模板编写，仅测试压缩与消费接口，不代表图像接地或自然caption泛化。共每组1,600条训练样本处理，不直接启动默认5,000步。

拟议最小消融（仅C，GEN读取后按需启动）：

|组|损失|回答的问题|
|---|---|---|
|CE-only|当前caption交叉熵|现有基线|
|Read-only|GEN响应蒸馏|生成消费约束是否有效|
|Read+Sem|响应蒸馏＋UND语义读出|是否保留细粒度语义|
|Read+Sem+CE|再加caption交叉熵|可解码性辅助是否有用；后续项|

对齐方式与位置契约见[设计](design.md)。不增加额外可训练GEN reader来解释这一轮结果。

## 4. 信息路径对照

|对照|改变|判断|
|---|---|---|
|Correct|正确来源/Memory|主结果|
|Source-off|writer不读来源|是否依赖视觉/文本来源|
|Source-swap|同t/p换反事实来源|写入是否随事实改变|
|Memory-swap|decoder/GEN读另一图的完整bank|消费是否依赖内容|
|Memory-masked|彻底排除Memory softmax项|语言/原条件基线|
|Pure-noise|t=1，同ε跨不同x0|输入和自由输出必须相同；检查泄漏|

单样本确定性writer下，Source-swap与Memory-swap等价，不能算两份独立证据。交换保留槽位数、位置与全部层级，记录donor ID。

Caption decoder只能读取Memory、固定指令和答案前缀。教师强制需要因果移位；自由解码与反事实事实分是必需检查。对齐loss下降不能代替这些证据。

## 5. GEN读取阶梯

|组|条件|作用|
|---|---|---|
|G-base|原prompt，原生布局|原生基线|
|G-layout|保留新增位置，彻底mask Memory|位置/布局控制|
|G-fixed|全零但可见的KV槽位|softmax可见槽位控制|
|G-neutral|固定中性文本seed KV|固定内容/分布控制|
|G-text|原prompt后原生追加caption|显式文本oracle|
|G-TL/TC|C1文本压缩Memory|压缩到GEN接口|
|G-VL/VC|C2视觉Memory|完整学生路径|
|G-swap|替换成反事实Memory|内容因果影响|

所有Memory组容量/位置一致。G-text文本长度可能不同，只作功能参照。零V仍改变softmax分母；G-layout才是真mask。当前直接拼接KV会改变GEN位置，G-layout不保证等于G-base。

先固定同一`x_t,t,p`比较conditional与CFG后velocity，再做后续采样。记录重复原生调用误差，比较响应范数/方向、预测图与对应事实；attention权重大或输出变化不能证明有用。

两类任务分开报告：欠指定prompt＋caption补充，用于校准语义接口；完整prompt的原生轨迹观察，用于检查实际部署。当前事实不一定是修复指令，Memory被读取也可能强化错误。

继续采样仅首节点注入，然后恢复原生条件。取最近原生时间节点并按actual t重建受控状态；保存requested t/actual t/全部后续节点。初步8场景检查，通过后扩到64场景、2seed、约0.9/0.5/0.25三个节点。受控图继续采样称状态条件补全，不能称纯噪声文生图提升。

### 当前首跑：C的颜色消费

固定[已选C checkpoint](validation.md#4-c1结果与拓扑选择)，所有参数冻结。使用8对红/蓝图像（16张），对象与承载面采用C1训练已覆盖的词汇，caption只改变颜色；原prompt给出对象/承载面而不含颜色。图像与caption必须逐张核验；可先原生BAGEL生成候选图再核验，不把生成提示词当图像真值。

此任务刻意校准已学会的颜色接口，不作为未见文本泛化或视觉Memory证据。`c1_text80.jsonl`无图像，不能直接用于GEN。新数据统一标为GEN接口校准用途并绑定场景对，不能混入后续训练再称独立测试。

首轮actual `t=0.5`，noise seeds为0/1。8个场景对×2seed形成16个并行任务；每卡处理完整一对、共32个样本状态。配对成员保留在同一worker，以满足swap来源读取。每个样本的各arm共享同一x_t/t/p；同图跨t的noise规则仍按第2节执行。

执行现有7组：G-base、G-layout、G-fixed、G-neutral、G-text、G-TC、G-swap-TC。不提供visual checkpoint，不跑G-VC，不续采样，不展开噪声网格。固定/布局项用于解释原有读出，不另扩展消融。原生GEN命令是单进程；`scripts/gen_npu.py`已提供16卡按场景对/seed分片与汇总。训练的 `--distributed`不能用于GEN。

记录conditional与CFG后velocity、预测图 `decode(x_t-t*v)` 的目标/错配颜色、对象和承载面、伪影及耗时。输出变化与方向相似均不能单独证明有用；G-text的原生文本布局不同，不能将端到端velocity差直接当作设计中的共享query响应蒸馏loss。

|首轮结果|后续动作|
|---|---|
|G-text未产生可辨颜色作用|该节点不能校准接口；只选2对在t=0.75复查，仍无作用则先修订任务|
|G-text有效，G-TC与错配/中性内容无可辨语义差异|先实现C的GEN响应蒸馏，继续冻结DiT；不扩大C2|
|G-TC与错配TC产生方向合理的颜色变化，对象/承载面与预测图无明显损伤|接口部分成立；再准备独立图像-caption训练/验证数据，启动小规模C2-C|
|Memory读取有颜色作用但产生明显伪影/噪点|先处理消费位置、响应与幅度约束；不增加循环或续采样|

首轮结论只覆盖已见文本词汇、两个seed、一个状态节点。对象/关系泛化不足保留为C1待改进项，后续以caption覆盖和语义读出处理，不用增加同一模板的CE步数替代。

## 6. 报告与继续条件

主表：文本桥；视觉caption按t；GEN响应与反事实；最终事实与画质。按场景族聚类计算区间，同图t/seed不当独立样本。t=1与clean/observed分表。

保留自由文本、对象/属性/计数/关系分项、unknown错误断言率、donor/未修改事实、conditional/CFG变化、盲评胜平负、耗时与显存。响应蒸馏额外报告未见query/噪声/改写的误差与梯度。

|结果|下一步|
|---|---|
|C0失败|修复原生解码/探针协议|
|文本桥失败|检查压缩、读出、位置、容量、梯度|
|文本Memory有效，视觉Memory无效|检查GEN→Memory与噪声适配|
|Caption可读，GEN仅无语义扰动|消费接口仍不成立|
|正确/错配/fixed效果相同|不能归因Memory内容|
|语义随Memory改变但画质退化|接口可读，保真不足；不增加循环|
|语义改善且质量稳定|只支持已测任务/t/单步范围|
|L/C差异区间重叠|证据不足，按成本选择后续对象|

参数变化、串样、答案泄漏、重复RoPE、t=1关联不可见x0时先修复实现。出现大面积噪点停止扩大该组。数值容差按目标设备重复误差设定；质量非劣门槛在盲评前确定。
