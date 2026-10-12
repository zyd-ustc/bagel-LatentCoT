# 设计

状态：保留C推进GEN读取；响应与表示蒸馏待实现。更新：2026-10-12。

[返回 README](../README.md)

## 1. 目标与原生约束

用固定容量的独立 Memory 表达 caption 语义，使其在冻结 BAGEL 的理解与生成接口中具有接近原生文本条件的作用。Memory 无需复制 caption 的 token 数量、顺序或逐 token hidden。

BAGEL 的理解专家称为 UND，生成专家称为 GEN；两者组成 MoT（Mixture of Transformers）。KV 是注意力的 key/value 张量。

原生文本 prefill 提供各层已训练的语义 KV，因此适合做教师。ViT 与 VAE 也能形成上下文，文本并非唯一可消费表示。共享形状与注意力接口不能证明语义分布兼容。

|约束|方案要求|
|---|---|
|参数|冻结全部 BAGEL、UND/GEN 投影与 MLP、norm、embedding、VAE/ViT、time/position 和输入/输出头|
|条件|保留原 prompt，Memory 使用独立槽位；无文本 CFG 分支保持原生|
|目标|对齐 caption；学生不接触目标 caption，教师可用 caption 构造监督|
|时间|沿用 `x_t=(1-t)x0+tε`，`v=ε-x0`；t=1为噪声端|
|KV|按层生成，保留专家与归一化位置；K 只旋转一次，不跨层复制最终 hidden|
|循环|首轮一次 writer、一次条件重算；固定 x_t/t，采样器只推进一次|
|兼容性|仅训练 writer 调用专用的新增模块，避免改变 prompt prefill 与共享 UND 路径|

训练 mask 对标为 `noise` 的图像段限制跨段读取。UND 读取当前带噪 GEN KV需要单独适配；原生干净图像理解能力不足以证明该路径已受训练。直接读取 GEN 原生 KV与将 GEN hidden重新经 UND 投影也是两种不同接口。

## 2. 两种 Memory 拓扑

`S_l`是固定中性文本的第l层输入种子，`Z_l`是同层来源KV，`F_l`是冻结UND完整block，`A_l`是新增残差，`Π_l`是冻结UND KV投影。

|拓扑|更新|层间关系|
|---|---|---|
|L：每层独立|`U_l=F_l(S_l;Z_l)`，`M_l=U_l+A_l(U_l)`|每层从自己的种子开始|
|C：连续更新|`U_l=F_l(H_l;Z_l)`，`H_(l+1)=U_l+A_l(U_l)`|上一层输出进入下一层|

两者读出均为更新后状态的同层投影 `Π_l(M_l)` / `Π_l(H_(l+1))`。这统一了比较接口，但不是原生 prefill 的输入位置读出。若L导出更新前KV，该层Memory尚未包含来源信息；不能据此与C比较写入能力。

初始设置为16槽位、全部模型层、同规格低秩残差。L/C使用相同参数量、训练预算、来源与消费位置。当前残差为无bias的线性瓶颈，输出矩阵零初始化。容量与读出位置的改变单独消融。

2026-10-12的同预算C1比较后，保留C作为后续学生拓扑；L保留为历史参照。C能保留红蓝差异，L出现颜色输出塌缩；C仍有对象/承载面替换，文本桥只部分成立。该选择仅用于推进当前方案，不能推导为C具有普遍架构优势。checkpoint与证据见[验证](validation.md#4-c1结果与拓扑选择)。

## 3. 路线一：冻结 UND 的语义读出对齐

固定少量语义查询，覆盖场景、对象与属性、数量、空间关系。教师读取caption KV，学生读取Memory KV；同一个冻结UND读出器生成查询的回答前hidden。

$$
z_j^T=R_{\mathrm{UND}}(q_j,C),\qquad z_j^S=R_{\mathrm{UND}}(q_j,M).
$$

`C`是caption条件，`M`是Memory，`q_j`是固定查询。对应关系建立在查询上，允许两侧条件长度不同。

$$
L_{\mathrm{sem}}=\frac1J\sum_j\left(1-\cos(z_j^S,\operatorname{sg}(z_j^T))\right).
$$

`sg`表示停止梯度。查询模板、读出层、norm与位置规则先固定；读出器不训练，避免投影头独自吸收差异。可将不同问题对应的回答token分布KL作为独立消融，不要求caption与Memory逐token对应。

此项检验共享语义，不单独证明GEN使用Memory。计数和关系分项验收，不能只看全局平均向量或loss。

两侧查询采用同一指令与读出层，并预留共同条件区间，使查询位置不随caption长度变化。该读出属于固定语义探针；真实自由解码另行验证。

## 4. 路线二：冻结 GEN 的注意力响应蒸馏（优先）

冻结BAGEL在同一`x_t,t,p`运行，取得第l层GEN query `Q_l^G`与共同prompt/self KV `B_l`。教师额外读取caption KV `C_l`，学生额外读取Memory KV `M_l`。

$$
O_l^T=\operatorname{Attn}(Q_l^G,B_l\cup C_l),\qquad
O_l^S=\operatorname{Attn}(Q_l^G,B_l\cup M_l).
$$

两侧输出长度由query决定，与caption/Memory槽位数无关。共同prompt/self必须进入同一softmax，才能约束Memory与原条件的竞争。

$$
L_{\mathrm{read}}=\sum_{l\in\mathcal L}w_l\,
\frac{\|O_l^S-\operatorname{sg}(O_l^T)\|_F^2}
{\operatorname{mean}((O_l^T)^2)+\epsilon}.
$$

分母按相同层/样本计算并停止梯度，防止大幅度层支配损失。记录未归一化误差、输出范数与方向；归一化损失小不能代替质量证据。

### 位置与消费接口

1. 教师caption通过原生文本prefill构造；C1可独立prefill，C2/生成诊断优先在相同原prompt上下文后追加caption，记录构造模式。
2. Query、共同KV与原prompt均来自同一冻结基线前向，两侧完全共享；不分别重跑教师/学生生成hidden充当共享query。
3. 探针为caption与Memory分别规定条件位置，两侧GEN query使用共同的固定位置；caption长度不同不能暗中改变query位置。可预先保留共同位置区间并限制最大caption长度。
4. Caption KV按原生prefill位置保留；Memory的未旋转K按探针规则旋转一次。固定query位置的探针属于局部校准，不宣称是原生端到端生成。
5. 先比较原生逐head attention输出，或经冻结GEN输出投影后的量。首轮固定一种读出，另一个单独消融。
6. 训练只更新Memory模块。冻结Q、共同KV、教师响应和输出投影，但学生attention到Memory K/V的梯度必须保留。

初期在代表性浅/中/深层诊断；采样层集合与权重两拓扑一致。验证用未见query、噪声、图像和caption改写，防止Memory只拟合训练探针。

C1可使用预先缓存、与目标caption独立的冻结GEN query库，覆盖多种图像状态与t；C2优先使用当前样本基线前向的query。两阶段记录query来源，不把C1固定探针成绩当成部署状态成绩。

局部响应匹配不保证多层闭环等价。随后检验真实全层GEN、CFG后速度与图像；若teacher的补充caption本身有害，不把它当质量提升目标。

## 5. 推荐训练阶段

|阶段|学生输入|教师/监督|训练参数|
|---|---|---|---|
|C0 校准|原生caption上下文|冻结UND复述；原生caption条件的GEN行为|无|
|C1 文本压缩|caption原生KV|`L_read`为主，`L_sem`辅助；caption CE作为辅助或检查|新增writer残差|
|C2 视觉迁移|GEN原生KV、明确记录的t/p|同一样本caption教师；复用C1消费接口|新增writer残差|
|读取验证|冻结C1/C2 Memory|正确/错配/固定/屏蔽Memory与显式文本对照|无|

推荐总损失：`L = L_read + λ_sem L_sem + λ_cap L_cap`。`L_cap`是当前已实现的caption交叉熵。后续仅在C上比较CE-only、read-only、read+sem，之后才增加CE辅助；系数由验证集与各项梯度量级选取，暂无已验证数值。

当前执行顺序为：固定C1-C checkpoint → GEN读取校准（已完成）→ 实现C的GEN响应蒸馏（尚未实现）。用户人工审查发现G-text变化明显，其余组几乎相同，学生Memory的内容作用尚未成立，暂不进入C2。先不增加CE训练步数、恢复L消融或进入多轮采样；这不改变上表的长期阶段定义。

C1文本源含答案，属于压缩校准；不能称视觉接地。C2学生不能读取caption或教师KV，只使用教师作为监督。训练caption decoder时仅可看到Memory、固定指令和已生成答案前缀。

目标caption描述干净图像，训练的是目标语义恢复；观察caption描述当前预测图，训练的是观察一致性。两者分开记录。高噪声下不能确定唯一caption，t=1不强制恢复某个独立x0。

## 6. 路线三：新增模块中的语义去噪（后续）

先用前两条路线获得固定语义目标`m0`，再训练新增模块：

$$
m_t=(1-t)m_0+t\epsilon_m,\qquad
\hat m_0=S_\theta(m_t,G(x_t,t,p),t).
$$

这里只是一种条件语义恢复设计；若要称为语义flow matching，还需定义对应速度目标与完整采样过程。先校准固定目标空间，避免teacher随学生训练漂移。

带噪`m_t`留在新增模块内，GEN读取经接口校准的干净预测Memory。推理不能访问caption目标；初始化、时间更新、跨步保存和不确定性均需单独设计。该路线不进入首轮实验。

## 7. 原生适配与参考边界

|参考|采用的思想|BAGEL上的取舍|
|---|---|---|
|[REG](https://arxiv.org/html/2507.01467v2)|紧凑语义状态持续参与生成；预测与表示监督结合|不直接把带噪语义token塞入原GEN；其REPA仍有匹配结构，不证明任意集合天然可对齐|
|[Looped-DiT](https://arxiv.org/html/2609.40305v1)|重复更新需要监督与幅度控制|不迁移MiniT2I训练配方、attention gate或输出头初始化|
|[Monet](https://github.com/NOVAglow646/Monet)|先定义可信语义目标，再迁移到学生状态|不迁移自回归token协议或反馈计算图|
|[Hidden-State Refinement](https://arxiv.org/pdf/2608.29160)|控制循环层、时间和残差强度|1/K不保证BAGEL同步GEN/UND循环稳定|

已有人工观察：独立Memory在training-free时作用较弱；L/C差异不大；joint-micro多轮出现噪点。原始输出未在本轮复核，不写成原因已确定或质量已验证的结论。

完整实验与继续条件见[实验](experiments.md)。当前代码只实现CE路径，映射与限制见[实现](implementation.md)。
