# 冻结 BAGEL：两种独立 Memory 的诊断实验安排

日期：2026-10-11。状态：待实现的实验协议。本轮只写文档，不实现代码、不运行训练或评测。

## 0. 要回答的问题与范围

比较两种独立于原 prompt 的 Memory：

- **L（Layerwise）**：每层有自己的 Memory hidden，各层更新不传递 hidden。
- **C（Continuous）**：一份 Memory hidden 顺序经过 UND 层，上一层输出进入下一层。

实验分两部分：

1. **Caption 诊断**：借助 BAGEL 原生文本 prefill 校准 Memory 接口，检查不同噪声下，GEN 状态中的视觉信息能否经 Memory 被冻结 UND 解码。
2. **GEN 读取诊断**：冻结 Caption 阶段得到的 Memory 模块，保留原 prompt，检查 GEN 是否使用 Memory 内容，以及内容影响是否符合语义。

原始 BAGEL 参数始终冻结。Caption 阶段允许训练明确列出的新增 Memory 模块；GEN 诊断阶段不更新任何参数。首轮不做多轮闭环、不做 joint-micro、不训练 GEN、不替换原 prompt。

这不是“选最终训练架构”的实验。先定位失败发生在文本压缩、视觉写入、噪声适配还是 GEN 读取，再决定后续训练。

背景与源码依据见[独立 Memory 设计分析](independent_memory_caption_design_20261011.md)。

## 1. 实验开始前固定的清单

|项目|首轮规定|
|---|---|
|模型|记录 BAGEL 权重 hash、源码 commit、配置、tokenizer 和 VAE版本|
|实现参照|已审计 `loop-layer-NPU@42e71148826011b74473a170c4ab62072d6c70ab`；实际实现若不同，记录差异|
|原参数|UND/GEN全部参数、norm、Q/K/V/O、MLP、文本embedding、LM head、VAE、ViT、connector、time/position、vae2llm/llm2vae全部冻结|
|Memory|16个内容槽位作为首轮工程默认；不将其注册为已有特殊token，不删除或压缩原prompt|
|诊断深度|先覆盖全部28层，让caption decoder与GEN都得到完整层级bank；若权重层数不是28，按配置处理并明确记录|
|轮次|每个状态只生成一次Memory，每层执行一次writer；不把末层hidden回送首层|
|分辨率|首轮固定512；这是诊断预算，不是对高分辨率的验证|
|数值设置|相同设备、dtype、attention后端、采样器、CFG和随机数隔离；保存实际值|
|模型选择|L/C使用相同验证标准、训练数据顺序、优化步数、可训练参数预算和超参搜索预算|
|测试集|按原始图像/场景族划分，所有噪声、caption改写和反事实变体留在同一split|

16槽位、下面的样本数和训练预算均为拟议诊断设置，不是已验证最优值。不得在看过测试结果后只给某个拓扑增加容量或训练时间。

## 2. 两种拓扑的精确定义

### 2.1 公共输入与符号

- `l`：UND层编号。
- `Z_l`：该层冻结来源KV；文本桥实验来自caption prefill，视觉实验来自同层GEN原生KV。
- `S_l`：固定中性槽位序列经过原生因果UND prefill，在第l层的输入hidden。它不包含目标caption或图像信息。
- `F_l(H; Z_l)`：冻结UND第l个完整block；槽位query读取来源KV及自身槽位KV。
- `A_l`：仅在Memory writer中启用的新增小型残差模块。
- `Π_l`：冻结UND norm/K/V/K-norm产生的第l层KV投影；RoPE位置单独按固定协议应用。

初始化槽位由一份固定中性文本的16个内容token embedding确定，保存token IDs及选择方法；不使用目标caption。`S_l`由这份固定序列的原生prefill得到并缓存。两种拓扑共享同一份初始化。首轮不训练槽位embedding，以免L的种子prefill反向计算成为另一个变量。

中性槽位只是可复现初始化，不代表16个原生CLS，也不代表天然具有摘要语义。

### 2.2 L：每层独立

```text
U_l = F_l(S_l; Z_l)
H_l = U_l + A_l(U_l)
B_l = Π_l(H_l)
```

各层从自己的`S_l`出发。`H_l`不送入第l+1层；每层输出一个Memory KV bank。

### 2.3 C：连续更新

```text
H_0 = S_0
U_l = F_l(H_l; Z_l)
H_(l+1) = U_l + A_l(U_l)
B_l = Π_l(H_(l+1))
```

每层读取相同来源规则的`Z_l`，hidden连续向深层传递。两组均每层执行一次冻结UND block、一次新增模块和一次读出投影。

**首轮统一使用“更新后hidden的同层投影”读出。** 这是新增Memory接口，不是原生prefill KV读出；两组共同采用，目的是让每层独立Memory在该层确实包含视觉信息，并隔离更新拓扑。不能将此首轮结果宣称为“原生KV分布比较”。

原因：若L直接导出该次writer的attention输入KV，读出的仍是尚未看见`Z_l`的`S_l`；这会把无视觉信息的L与已经携带前层信息的C混比。

后续仅在必要时增加 **C-input**：C改为输出`Π_l(H_l)`，即原生输入位置读出。此时第0层bank尚无视觉信息，必须明确。C-input与C主组比较的是读出位置，不能替换主表中的C或将差异归因于拓扑。首轮不为L另加第二次writer来补救，否则计算预算与拓扑都变了。

### 2.4 新增模块与冻结边界

首轮`A_l`统一使用相同规格的低秩残差，例如瓶颈宽度32，输出矩阵零初始化、输入矩阵正常初始化；实际规格写入配置。仅更新`A_l`，原始UND权重不更新。两组训练参数数目一致。

首轮直接读取来源的原生KV，不同时训练GEN→UND投影桥；先看冻结读取路径加小残差是否足够。若失败，可另开对称增容实验，为L/C同时增加writer专用query/读取adapter，不能把增容后的结果混入首轮表。

### 2.5 Attention与位置约定

1. Writer只读取指定来源KV和当前槽位self KV。固定任务指令若加入，L/C完全相同。视觉writer不额外读取完整目标caption或目标答案。
2. 来源KV保持原生专家、层号、K-norm和RoPE，不再用UND投影处理GEN原始hidden。槽位使用明确记录的新位置；GEN来源的原始位置不改。
3. Writer槽位可双向读自身，L/C相同。这是摘要接口，不假称原生因果文本任务。
4. 消费端保留Memory投影后的未旋转K及V，再按消费端固定槽位位置应用一次RoPE。禁止对已经旋转的K再次旋转；禁止跨层拷贝KV。
5. Caption端将Memory视为16槽位前缀，随后追加固定解码指令及答案；所有Memory组位置一致。GEN端布局另见第6节。位置重建属于新增接口，必须记录，不保证原生语义等价。
6. Packed batch必须保持样本隔离。所有反事实实验只交换指定内容，不更换长度、位置或样本索引。

## 3. 数据、噪声与标签

### 3.1 两种状态来源分开报告

**A. 受控加噪集：用于训练和可解释的噪声曲线。**

从有核验caption的图像编码得到`x0`，按BAGEL定义：

`x_t = (1-t)*x0 + t*eps`，`eps ~ N(0,I)`。

固定原生VAE尺度与patchify。同一图像/seed在各t复用同一份eps。提取冻结GEN的逐层原生输入KV；输入条件先使用统一中性prompt，不提供图像caption。

主评测t取 `{0.05, 0.25, 0.50, 0.75, 0.90, 1.00}`。这里t是加噪公式中的实际值，不再二次sigmoid或shift；t=1是纯噪声负对照。t=0可以作为额外边界检查，不纳入默认训练与主平均。

**B. 原生采样轨迹集：用于检查部署分布差异。**

用固定prompt/seed运行原生BAGEL，保存实际`x_t,t`与GEN特征。按t桶归类，并记录实际step与t，不把受控加噪状态当成真实采样轨迹。主评测可取最接近上述非纯噪声t的节点；若关注Early20，额外报告该窗口全部所选节点的实际t。

不把轨迹的“初始噪声减最终图像”当作任意中间状态的flow真值。本协议GEN阶段也不训练flow损失。

### 3.2 Caption标签的含义

|标签|来源与用途|允许的解释|
|---|---|---|
|`y_clean`|核验后的原始干净图caption；受控加噪集主训练目标|从带噪状态恢复原图语义；不是当前图像观察准确率|
|`y_obs`|原生速度预测得到`x0_hat=x_t-t*v_base`，VAE显示图经人工核验后的观察描述|对该显示图的观察一致性；不代表GEN内部全部信息|
|`facts`|对象、属性、计数、关系及unknown标签|事实级评价，区分看不清与答错|

`y_clean`首轮在t<1的受控加噪集上训练，t=1不要求恢复某个独立x0。t=1评测看是否退回先验、过度确定，以及是否仍异常关联x0标签。对于固定中性prompt，同一eps在t=1与不同x0配对应得到相同输入和输出；这是强泄漏检查。

`y_obs`先用于小规模轨迹评测，不与`y_clean`混成一个accuracy。若之后单独训练观察caption，必须命名为另一实验，并单独选择checkpoint。教师自动caption只作为候选，不能未经校验就充当视觉真值。

### 3.3 拟议最小规模

- 工程检查：16个图像样本，不用于质量结论。
- 训练池：先用约2,000个不同图像/场景的核验caption；这是接口诊断规模，失败不能直接归因于架构上限。
- 验证池：256个图像，用于共同checkpoint选择。
- 测试池：256个未见图像，6个t、2个噪声seed，共3,072个状态；各组复用同一批状态。
- 测试池中至少64对同类反事实场景，分别改变颜色、数量或空间关系；配对关系保存。
- 部署轨迹检查：32个prompt、2个seed、每轨迹最多5个节点；对可见预测图标注`y_obs`与unknown。另记录prompt与观察不一致的子集。

所有尺寸可因预算统一缩减。数据split和反事实配对必须在训练前固定。不能用训练caption重建表现替代未见样本结果。

## 4. 实验一：Caption任务与文本prefill桥梁

### 4.1 C0：原生文本参照

正确caption通过原生`forward_cache_update_text`得到逐层KV。冻结UND decoder读取这份KV并执行固定复述指令，输出caption。此组输入已含答案，明确标记 **text oracle**；用于校准解码协议，不是视觉任务成绩。

若该组不能稳定复述，先修复BOS/EOS、角色/任务格式、位置、KV布局和解码入口，再训练Memory。记录自由解码结果，不能只看teacher forcing。

### 4.2 C1：文本压缩桥，L/C两组

源`Z_l`使用正确caption的原生prefill KV。L/C把不同长度文本压缩成16槽位Memory。冻结caption decoder仅读取Memory、固定指令和已生成答案前缀。

主损失为caption交叉熵：

`L_cap = -mean_i log p_frozen(y_i | y_<i, B)`。

允许teacher forcing，但答案必须因果移位，writer不能读取答案token；答案只存在于文本源这一明确的自编码任务。保留不更新新增模块的C1-init对照，以区分初始化与训练作用。

拟议共同训练预算：L/C各最多5,000 optimizer steps，有效batch32；相同优化器和初始学习率，例如AdamW、1e-4、weight decay 0、全局梯度裁剪1.0。每250步评估一次验证集。若不稳定，作为新配置同时重跑两组；不只救一个拓扑。参数是诊断起点，不是BAGEL推荐微调配方。

按验证集事实正确率优先、自由caption质量次之选择checkpoint；报告最后一步与最佳验证checkpoint，禁止按测试集选。

### 4.3 C2：GEN特征迁移，L/C两组

分别从各自C1 checkpoint开始。源`Z_l`改为受控加噪状态的GEN原生KV，学生没有caption输入，主目标为`y_clean`。随机均衡抽取五个t<1桶，训练预算与C1同上。

首轮保持相同`L_cap`；文本prefill桥通过C1初始化体现。若要增加text-teacher logit KL，另开C2-KD消融，teacher/student使用相同答案前缀；不要让KL有无与拓扑绑定。KL降低不是视觉信息写入的证据。

另外为L/C各保留直接从C1-init训练GEN输入的C2-scratch组，预算相同。它用于判断文本桥是否实际帮助迁移；若预算不足，可先不跑，但文档结果必须标为“桥梁增益未验证”。

不训练原始UND decoder。冻结参数不等于整个前向`no_grad`：decoder到Memory、Memory经过冻结UND block的输入梯度必须保留。冻结GEN来源和text teacher可停止梯度。首个训练批次必须确认新增模块梯度有限、原权重未变化。

### 4.4 每个checkpoint必须做的对照

|组|改动|用于判断|
|---|---|---|
|Correct|正确来源生成Memory|主结果|
|Source-off|writer不读取视觉来源，其他部分相同|是否仅凭槽位/语言先验作答|
|Source-swap|相同t、相同中性prompt下交换反事实图像来源|是否随视觉事实改变|
|Memory-swap|caption decoder读取同布局但另一图像的Memory|是否真的经过Memory瓶颈|
|Memory-masked|完全屏蔽Memory，仅保留任务指令|语言先验基线；不是零V|
|Pure-noise|t=1，同eps跨不同x0复用|泄漏与过度确定检查|

不能让caption decoder读取GEN/ViT/VAE KV，也不能保留原图caption的prompt KV。视觉任务使用真实prompt的追加评测时，要单独报告；同prompt多seed轨迹中，GEN本身仍携带prompt语义，不能只凭decoder看不到prompt就排除复述。

### 4.5 Caption报告

按L/C、文本桥/视觉迁移、噪声桶分表，至少记录：

- teacher-forcing NLL和自由解码原文。
- 对象、颜色/属性、数量、关系各自的precision/recall或准确率；计分规则在评测前固定。
- `y_obs`的已知事实准确率、unknown比例及对unknown的错误断言率。不能靠全部答unknown获得高分。
- 正确Memory相对masked/错配Memory的事实差值。
- 反事实修改是否跟随真实修改，未修改事实是否保持。
- 每组训练参数量、优化步数、吞吐、峰值显存和writer/decoder耗时。

置信区间按原始图像或反事实场景族聚类重采样；不要把同图的多个t/seed当独立样本。t=1不与五个非纯噪声桶合并为“恢复能力平均”。

## 5. Caption阶段的继续条件

|观察|判断与下一步|
|---|---|
|C0失败|先修复原生文本解码协议，停止解释Memory训练|
|C0正常，C1的L/C均失败|优先检查压缩读出、位置、容量、梯度；暂不能评价GEN→Memory|
|C1正常，C2只在低噪声正常|存在噪声适配/可观测性限制，按t报告；不能宣称早期循环可用|
|C2 loss下降，但自由caption及swap无差别|可能依赖语言或prompt先验，接地未证实|
|L/C差异小且区间重叠|报告证据不足，结合成本选择后续对象；不强行选赢家|
|Correct稳定优于masked且反事实跟随正确|可以进行GEN消费解释；GEN诊断本身也可先跑以检查接口|

即使Caption失败，也可以跑文本oracle的GEN读取组定位另一接口；但不能用失败的学生Memory证明GEN不会读。

## 6. 实验二：GEN读取Memory对比

### 6.1 冻结所有参数，先做直接KV接口

冻结选定的C1/C2模块，不新增训练中的GEN reader，不使用可学习gate。首轮将Memory作为独立条件bank接入原生GEN attention，原prompt KV保持完整。

明确承认：拼接KV会改变attention归一化和位置布局。这一阶段测量的是直接读取能力及其副作用，不预设它安全。

固定布局：原prompt在原位置；Memory占后续16个独立位置；GEN query按这个长度通过统一准备逻辑取得位置。Memory新K按该布局旋转一次。L/C、correct/swap/control使用完全相同的位置和容量。

同时保留两种基线：

- **G-base**：完全原生布局、没有Memory。
- **G-layout**：使用Memory组相同的GEN位置与packed布局，但16个Memory位置被彻底mask，不参与softmax。用于测量位置/布局变化。

G-layout不保证等于G-base。若后端不能表达mask，可用功能等价的诊断attention实现，并先与原生路径验证；不能把零value当作mask。另设固定无信息Memory组，测量新增可见KV本身的影响。

初步只在同一次denoiser调用中读取bank，调用结束释放，不跨时间步缓存。完整后续采样恢复原生条件。这样先测单次影响，不混入Memory闭环演化。

### 6.2 G0：显式文字与Memory的接口阶梯

|ID|条件|含义|
|---|---|---|
|G-base|原prompt|原生参照|
|G-layout|原prompt＋masked槽位，位置与Memory组相同|布局参照|
|G-fixed|原prompt＋相同长度的固定无信息Memory|可见KV/新增容量参照|
|G-text|原prompt＋正确补充caption，走原生文本prefill|显式语义条件oracle|
|G-TL / G-TC|原prompt＋C1文本压缩Memory|L/C文本oracle Memory|
|G-VL / G-VC|原prompt＋C2视觉Memory|L/C学生Memory|
|G-swap-*|上述各Memory组换成反事实样本的bank|语义读取对照|

G-text保留原prompt并追加文本，不替换prompt；它的token长度可能不同，因此只是功能参照，不是完全布局匹配对照。Memory各组之间才严格匹配16槽位。所有交换使用完整层级bank，不能把不同层来自不同图的bank作为主要swap。

### 6.3 两类GEN样本分开做

**A. 受控补充语义任务：先确认接口可以传递信息。**

使用固定且欠指定的prompt，例如只描述对象类别，不指定颜色或左右关系。正确caption补充缺少的属性；反事实caption仅改变一个属性。原prompt在全部组中保持不变且始终保留。

对同一受控`x_t,t`，比较正确/反事实text Memory与视觉Memory。低噪声state本身已经包含目标信息，可能让Memory冗余；高噪声视觉Memory又可能不可靠。因此必须按t给曲线，不只给总平均。

此任务若从受控加噪图继续采样，名称是“状态条件补全”，不能写成从纯噪声文生图提升。G-text和文本Memory拥有额外caption信息，属于oracle上界诊断，不是与学生等信息竞争。

**B. 原生轨迹观察任务：检查实际生成状态的读取。**

使用原生完整prompt，在保存的同一轨迹节点提取学生Memory，比较正确/同prompt其他seed的Memory与关闭Memory。只在该节点重算一次，再完成剩余采样。

这里Memory描述的是当前状态，未必是修复指令。若它强化已有错误，说明可能被读取，但不能说带来质量收益。分别报告“是否读入事实”和“是否改善最终图像”。首轮不把最终目标caption作为视觉学生输入。

### 6.4 固定状态的局部测试

对完全相同的`x_t,t,p`计算各组conditional velocity；同时按原生CFG流程记录最终velocity。

记录相对G-layout的归一化速度差、余弦相似度、范数比，以及`x0_hat=x_t-t*v`的对照图。G-base对G-layout单独报告布局影响。数值距离只作敏感性诊断，不作语义质量指标。

若抓取attention统计，记录GEN对prompt/Memory/self的质量分配、Memory贡献量以及层号。不要仅凭attention权重大宣称有效读取。

判读顺序：

1. Correct与swap输出是否超过重复运行的数值误差范围。
2. 改变是否沿反事实颜色/数量/关系的方向，而非随机破坏。
3. Correct是否比G-fixed与G-layout更符合目标。
4. conditional变化是否被CFG放大为异常速度或画质退化。

### 6.5 后续采样与人工盲评

局部比较通过后，选择64个平衡场景、每场景2个seed，初步取t≈0.90/0.50/0.25三个节点。实际节点来自受控状态还是原生轨迹分别标注。

每个arm从同一状态继续：仅选定节点读取Memory一次，此后恢复原生；沿用相同sampler、CFG、分辨率和后续噪声规则。先做8场景的小检查再展开，不将t×arm全部铺满作为默认首跑。

盲评至少分开标注：目标事实、未改动事实、画质/细节、噪点/结构崩坏。结果匿名、显示顺序随机。报告成对胜/平/负和按场景聚类的置信区间，展示失败例而非只展示改善例。

若局部变化明显但单次注入最终影响消失，只能说“当前单步干预未形成持久作用”。后续可单独增加连续多步读取，但不能把跨步实验混入本协议主表。

## 7. GEN诊断的结果解释

|结果|支持的判断|不支持的判断|
|---|---|---|
|G-text也不改变目标事实|任务/状态可能对该补充信息不敏感，或文本布局有问题|独立Memory一定无效|
|G-text有效，G-TL/TC无效|压缩Memory→GEN接口存在瓶颈|一定需要解冻GEN|
|文本Memory有效，视觉Memory无效|视觉写入/迁移/噪声适配优先排查|GEN不会读任何Memory|
|Caption可读，GEN只出现无语义扰动|UND可读性没有转成GEN语义消费|grounding已解决|
|正确/错配Memory均比base同样变化，且fixed也相同|容量、位置或一般扰动可能解释效果|Memory内容带来收益|
|反事实Memory引发对应语义变化，但质量下降|GEN消费了内容，但接口保真不足|可以直接增加循环深度|
|正确Memory改善事实且质量稳定|该任务/t/单步设置存在有效消费证据|所有噪声、分辨率或多轮都成立|

若直接KV接口失败，另写后续实验比较独立门控读取支路；本轮不在失败后悄悄加入reader训练，因为那会改变问题与参数预算。

## 8. 必须保存的实验记录

每个run保存模型/源码/数据split/config hash、参数冻结清单、可训练参数名/数量、训练种子、优化器设置和checkpoint选择规则。

每个样本至少保存：

- sample/scene/pair/split ID；prompt、caption、token IDs；标签类型`clean/observed`及unknown事实。
- 状态来源`controlled/trajectory`；噪声seed、actual t、step index、x_t hash；VAE编码及RNG方式。
- 拓扑L/C、writer深度、槽位数、读出位置`post_block`、源专家/层号、mask与position IDs。
- 来源是否正确/关闭/交换，交换对象ID；Memory bank或可确定重建的checkpoint与输入清单。
- Caption自由文本、逐事实评分与teacher-forcing NLL；GEN conditional/CFG velocity统计和最终图。
- 权重未改变检查、有限值检查、峰值显存和各阶段耗时。

主报告只需四张表：文本桥结果；视觉caption按t结果；GEN局部敏感性与反事实结果；最终图像事实/画质结果。附录保留完整配置、样本和失败案例。

## 9. 执行顺序与停止规则

1. 固定数据split、配置和所有冻结参数，做16样本工程检查。
2. 跑C0原生文本参照，验证解码协议。
3. 跑C1-L/C文本压缩及初始化对照，验证固定槽位接口。
4. 跑C2-L/C不同噪声的视觉caption，执行masked/swap/pure-noise对照；有预算再做scratch消融。
5. 全部模块冻结，跑GEN的text→文本Memory→视觉Memory阶梯与布局对照。
6. 对可解释的局部结果做单节点后续采样与盲评。
7. 分别报告caption接地、GEN消费和画质，不用一个总分替代三个结论。

原参数发生变化、样本串扰、答案泄漏、mask/位置未绑定、t=1依赖不应可见的x0时，停止解释质量结果并先修复实现。出现大面积噪点时停止扩大该arm，保留失败轨迹；不通过更大训练或更多循环掩盖问题。

Caption差或GEN不敏感本身不是工程失败。只有排除上述实现问题后，才把它们当作接口证据。所有数值容差由目标设备的重复运行误差确定，质量非劣门槛在盲评前约定；本协议不预设任意百分比为安全阈值。
