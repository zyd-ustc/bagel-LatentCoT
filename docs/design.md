# 设计

更新：2026-10-12。

[返回README](../README.md) · [实验记录](experiments.md)

## 1. BAGEL架构

BAGEL的理解专家称为UND，生成专家称为GEN；组成MoT（Mixture of Transformers）。KV指注意力中的key/value张量。本方法以原生文本prefill（预填充）产生的逐层KV作为语义来源，用独立的16槽Memory压缩并表达这些条件。

全部BAGEL权重保持冻结，包括UND、GEN、embedding、归一化、位置/时间模块与输入/输出头；VAE和ViT也保持冻结。训练只更新Memory writer中的新增低秩残差。冻结UND仍保留对Memory输入的梯度，使损失能够回传到各层残差。

原prompt保持原样。Memory追加到独立条件位置，不替换prompt hidden或KV；无文本条件的CFG（Classifier-Free Guidance）分支保持原生。Memory K按同层UND接口导出，消费时只应用一次旋转位置编码RoPE。

当前已完成的训练以caption文本为writer输入，属于文本压缩与生成消费实验。学生从caption原生KV获得信息；训练query图只提供GEN状态，不提供图像caption标签。当前结果尚不包含从带噪GEN状态写入语义的视觉训练。

## 2. 连续Memory更新

### 中性prompt初始化

配置使用固定中性文本 `neutral_text`，选取前16个非特殊token。原生UND预填充构造各层输入种子；连续Memory从第0层输入种子开始，即这16个token的冻结embedding。此初始化不含样本caption中的对象、颜色、数量或关系。

生成侧的原prompt是独立配置 `neutral_prompt = "An image."`。它与用于初始化Memory的较长中性文本不是同一段文本。

### 连续读取caption与低秩残差

一份Memory依次经过全部28个冻结UND层。第l层读取同层caption原生KV，并将更新后的Memory传给下一层。

定义：`H_l`为第l层的Memory输入，`Z_l`为caption的同层KV，`F_l`为冻结UND完整block，`A_l`为新增低秩残差。更新过程为：

$$
U_l=F_l(H_l;Z_l),\qquad H_{l+1}=U_l+A_l(U_l).
$$

每层残差采用两个无bias线性投影，rank为32：

$$
A_l(U)=W_l^{\mathrm{up}}W_l^{\mathrm{down}}U.
$$

`W_up`零初始化，初始残差为零；训练调整新增投影，使Memory适应冻结模型的读出接口。残差只作用于Memory更新，不作用于原prompt或GEN图像hidden。

每层更新后状态经冻结的同层UND归一化、K/V投影和K归一化，形成该层Memory KV。各层分别导出KV，不将末层hidden复制到所有层。

### 位置与读取

Writer的Memory位置从2048开始，caption来源保留原生prefill位置。Caption解码时Memory占位置0–15，固定指令和答案从位置16开始。真实GEN读取时，Memory接在原prompt之后，GEN query位置后移16。

截至本页记录的已完成实验，每个诊断状态只运行一次writer，并在GEN中读取一次Memory。评测图像为单步局部预测，未运行多轮Memory更新或后续完整采样。后续闭环设计见第5–9节。

## 3. Caption语义读出训练

Caption先经冻结BAGEL原生文本prefill，得到逐层来源KV。连续Memory读取这些KV后，冻结UND解码器只看到Memory、固定描述指令和答案前缀。目标文本为输入caption。

训练使用教师强制的交叉熵，记作Caption负对数似然NLL：

$$
L_{\mathrm{cap}}=-\frac1N\sum_{j=1}^{N}
\log P_{\mathrm{UND}}(y_j\mid M,I,y_{\lt j}).
$$

`M`是Memory，`I`是固定指令，`y_j`是目标caption的第j个token。Memory槽位与caption token无需一一对应。当前实现使用caption token监督，没有hidden逐token匹配或教师logit KL损失。

真实实验结论：C1的64条模板训练100步后，8条验证文本的颜色均跟随输入，字面事实分45.83%；对象与承载面仍有替换。该checkpoint的GEN对照中，审查结论：除G-text外其余组几乎相同。结果记录见[Caption训练](experiments.md#3-c1-caption训练)。

## 4. GEN注意力响应蒸馏

在独立图像的同一带噪状态上运行冻结GEN，缓存图像query `Q_l`及共同prompt/self KV `B_l`。教师条件 `T_l`来自原prompt后追加caption的原生prefill；学生条件 `M_l`来自连续Memory。

教师与学生共用同一query和共同KV，条件进入同一softmax：

$$
O_l^T=\mathrm{Attn}(Q_l,B_l\cup T_l),\qquad
O_l^S=\mathrm{Attn}(Q_l,B_l\cup M_l).
$$

输出长度由query数决定，不取决于caption或Memory长度。教师响应停止梯度，学生响应保留到Memory K/V及新增残差的梯度。

读出使用第0/13/27层的attention逐head输出，位于GEN输出投影之前；每层选64个图像query。三层等权的归一化均方误差为：

$$
L_{\mathrm{read}}=\frac13\sum_{l\in\lbrace 0,13,27\rbrace}
\frac{\mathrm{mean}((O_l^S-\mathrm{sg}(O_l^T))^2)}
{\max(\mathrm{mean}((O_l^T)^2),10^{-8})}.
$$

`sg`表示停止梯度。实际训练损失为：

$$
L=L_{\mathrm{read}}+0.1L_{\mathrm{cap}}.
$$

探针统一预留64个条件位置，教师caption含包装token最多64个；GEN query及共同self K统一后移64。Memory仍占16个位置，K仅在消费时旋转一次。真实GEN评测使用16槽位置规则，区别于训练探针的64位置间隔。

真实实验结论：首轮响应训练后，人工审查发现部分G-TC样例与base/swap出现明显差异；扩展训练后，在原4图、t=0.9对照中，G-TC与G-text性能效果基本持平。具体数据与checkpoint见[实验记录](experiments.md#9-累计1100步的gen对照)。


## 5. 生成状态反馈闭环（已实现，语义与画质待验证）

第1–4节描述已有文本方法；第5–9节的首版视觉query适配、完整GEN联合目标及R2反馈已在NPU分支实现。2026-10-12，Ascend E0数值与梯度检查通过；正式视觉训练、语义收益和最终画质尚未验证。实施依据为工作区配套文件 `artifacts/docs_reference/gen_memory_loop_training_plan_20261012.md`。启动入口与配置见[README](../README.md#视觉反馈训练npu)，实际检查见[实验记录](experiments.md#10-视觉闭环实现与ascend-e0)。

下一阶段有三个独立目标：

1. UND读取不同噪声水平的GEN状态，将生成侧信息写入Memory。
2. GEN消费其中有用的信息，并通过完整网络形成有效速度修正。
3. 新生成状态在后续轮次提供有效反馈，使闭环比单轮具有净收益。

C1和响应蒸馏建立了可用的文本读出与生成读取接口，并未直接证明Memory与原生文字hidden的统计分布相同。后续应保护两侧的读出能力，而不是强制视觉Memory数值等同于文本Memory。

用`p`表示原prompt，`t`表示噪声时间，`x_t`表示当前带噪latent。箭头表示信息依赖；`S`为GEN逐层状态，`M`为逐层Memory KV，`v`为完整GEN输出的条件速度。首版候选闭环为：

```text
固定 x_t、t、p → 原生GEN → S^(0)
S^(0)、t、允许的目标条件 → UND writer → M^(1)
固定 x_t、t、p、M^(1) → 完整GEN → S^(1)、v^(1)
S^(1)、t、允许的目标条件 → UND writer → M^(2)
固定 x_t、t、p、M^(2) → 完整GEN → S^(2)、v^(2)
```

每轮writer从相同中性浅层种子开始，连续经过28层UND。轮间信息通过GEN状态反馈传递；首版不将末层Memory hidden直接送回首层，也不新增上一轮Memory直连。每轮GEN从同一`x_t,t,p`正常重算，最终只推进一次采样器。此候选使用完整GEN重算，不等同于旧`legacy_layerwise`或只循环浅层的`full_depth_restart`；必须分别记录调用流程与计算成本。

## 6. GEN状态写入与C1接口保持

### 6.1 读取接口

首版优先让UND query读取同层GEN原生K/V，保留其专家投影、归一化和位置语义。将GEN hidden重新按UND token投影属于另一种桥接方案，不与首版混合比较。来源记录必须包含层号、实际t、轮次、token角色和RoPE状态；图像token与边界token分别标记。

以累计响应1100步的writer作为初始化和冻结参照。学生优先增加仅在视觉写入时启用的、感知t的读取适配模块，保留16槽及逐层UND KV导出接口。初期冻结原writer残差；若容量不足，再开放学生writer残差，同时混入文本回放和原有响应约束。原生BAGEL始终冻结。

冻结GEN特征可以停止梯度；特征之后的可训练模块必须保留梯度。现有writer会detach来源K/V，接入来源侧可训练模块时必须调整这一边界，不能让新增参数与损失断开。

首版 `VisualReadAdapter` 对每层UND归一化后的Memory hidden增加感知t的query残差，位置在原生q_norm与RoPE之前。rank=32，输入及时间投影正常初始化，输出投影置零。来源GEN K/V保留原专家投影与RoPE；默认捕获本轮完整query，分别标记图像和边界token，不含历史prompt KV。

视觉开关关闭时恢复原文本writer；零视觉模块恢复C1直接读取GEN K/V的基线。两者都不代表无Memory的原生Base。默认只训练新增视觉模块，原writer及BAGEL/DiT/VAE冻结。开启学生writer须显式设置 `train_writer=true`，独立1100步文本教师始终冻结。

配对状态保存实际采样的x0与epsilon，按 `epsilon-x0`监督完整conditional GEN。观察代理绑定实际t、noise seed及x_t哈希；未审查代理不进入确认指标。默认中性prompt不接收目标/观察差异标签；使用差异标签须显式允许部署目标prompt。

### 6.2 是否增加Memory序列

首版沿用16槽，不默认concat新序列。新增槽位能隔离存储内容，但会改变attention竞争和可能的RoPE布局，不能据此保证原C1消费行为不变。

只有观察到语义容量不足，或视觉训练与文本保持发生明确冲突时，再比较共享槽位与目标/观察分组槽位。比较控制总槽数、参数规模和位置；额外增加容量的实验单独报告。两组槽位都必须接受消费和错配对照，防止GEN只读原组。

### 6.3 监督的含义

|标签|监督用途|证据边界|
|---|---|---|
|配对干净图的caption/事实|从带噪状态恢复目标语义|不证明识别了当前生成错误|
|当前状态的可核验观察描述|识别当前生成内容与不确定性|不自动提供修复方案|
|目标与观察的差异|为生成修正提供任务信息|需要独立验证纠正效果|

局部预测图`decode(x_t-t*v)`只能作为有损观察代理，不能当作GEN hidden的完整真值。高噪声下允许unknown；纯噪声端不施加缺乏证据的确定性图像细节标签。

学生不能读取答案caption。语义解码器只能读取Memory、固定指令与因果答案前缀。GEN本身可能已含prompt信息，因此还需固定prompt、改变图像或生成状态的对照，验证Memory不是间接复述目标。

观察与目标保持不同角色。例如“当前两个对象”不能直接作为“生成两个对象”的条件。先用短问答分别验证目标、观察与差异，再验证GEN是否做出正确修正；首版不强制以新增槽位表达这些角色。

## 7. 从局部读取到有效生成修正

保留第4节响应蒸馏作为接口校准。当前固定query和共同KV的探针只约束局部attention，不能证明学生自己的query、后续GEN层及输出头保留了有效修正。

响应教师必须与任务对应。正确目标文本可用于条件接口校准；观察文本可用于语义读出校准，但未经生成验证不能作为纠错教师。视觉训练中的图像状态、标签与教师语义必须匹配，不能直接沿用文本阶段任意caption与独立query图配对的监督解释。

检查条件增量的范数、方向和误差，避免完整响应能量掩盖较小的条件效应。师生减去同一baseline不会改变MSE分子；若增加增量归一化，须明确能量下限和弱信号权重。

随后加入完整GEN输出监督。对可控配对加噪状态，使用原生方向：

$$
x_t=(1-t)x_0+t\epsilon,\qquad v^*=\epsilon-x_0,\qquad
L_{\mathrm{gen}}=\|v_{\mathrm{cond}}-v^*\|^2.
$$

梯度穿过冻结GEN后续层、最终归一化与输出头回到Memory模块。生成损失作用于conditional速度；CFG后的输出另行评测。自采样轨迹没有可直接照搬的配对`v*`，必须采用有依据的教师或其他单独定义的监督，不能将最终生成图回填为虚构真值。

联合目标由视觉语义损失、响应损失、生成损失和文本保持损失组成，权重经小规模验证确定。正确、关闭、固定及同prompt错配Memory在同一状态与布局下比较；速度变化或attention变化本身不是质量收益。

新视觉响应教师采用独立冻结C1 writer压缩后的16槽文本Memory，师生及真实GEN消费统一预留16位置。旧文本实验的原生caption教师及64位置探针仍按第4节记录，两套设置不混报。响应记录条件增量，并按固定能量下限过滤弱教师信号。

默认全局batch16，其中25%为文本回放。原writer冻结时，文本回放只做保持检查，不给视觉模块提供梯度；开放writer后才产生保持训练梯度。新checkpoint保存视觉模块、writer、独立教师、优化器、随机状态及数据哈希；旧C1/C1-read通过严格接口检查迁移，旧C2入口保留。

## 8. 多轮训练与增量衰减诊断

用户观察到training-free中loop>1时Memory更新幅度不减，而最终v增量明显减少。本页将其记录为待定位现象；尚未核对对应原始增量日志，不能确定原因。

诊断按实际Memory K/V、GEN attention输出、各层hidden、最终归一化、输出头、conditional速度及CFG后速度依次定位。分别记录相邻轮和相对Base变化；UND block内部残差不等于轮间Memory增量。不同张量空间的原始范数不能直接作为衰减比例比较。

可能原因包括无效Memory漂移、读取不敏感、后续网络削弱、CFG变换，或有益收敛。只有结合语义和生成误差才能区分。不得以强制增大每轮速度增量作为训练目标。

先展开R1/R2，每轮完成真实GEN计算后监督出口：

$$
L_{\mathrm{multi}}=\sum_{r=1}^{R} w_r\|v^{(r)}_{\mathrm{cond}}-v^*\|^2.
$$

首版默认停止跨轮GEN反馈梯度，逐轮保留本轮Memory到GEN输出的梯度，两个出口等权。`cross_round_gradients=true`启用跨轮反向传播，捕获及writer同时保留反馈来源梯度。多出口损失不保证逐轮质量单调提升，须独立评测。

关键对照是R2新反馈与R2重复旧反馈。还需比较固定Memory重复计算、单轮和原生Base。只有新反馈提供额外语义收益，才支持闭环机制；完整采样还需与相同计算预算下增加普通采样步骤比较。

## 9. 推进顺序与通过条件

|阶段|主要问题|进入下一阶段的证据|
|---|---|---|
|视觉单轮写入|Memory是否包含GEN状态信息|同prompt状态变化可被识别，正确来源优于错配/关闭来源；文本接口保持|
|单轮生成联合|GEN是否有用地消费视觉Memory|正确Memory带来对应语义净收益，固定/错配对照不能解释该收益；完整采样检查退化|
|两轮闭环|更新后的GEN反馈是否带来新增收益|R2优于R1及旧反馈对照，质量与计算成本均有记录|

结果按实际t、任务类型和轮次分层报告。保留未参与checkpoint选择的测试集，按图像/场景族划分，避免同图不同噪声跨split。验收同时报告语义修复、原有事实损伤和画质变化，不用单一loss下降代替成功结论。

首版默认使用配对加噪训练，S1按当前学生每步真实重算。轨迹刷新是可选的训练族来源快照（默认关闭），不产生配对速度标签。完整采样采用50节点shift3网格，在前20次更新启用Memory；每个时间步重新初始化中性seed，最终轮之后仅执行一次Euler更新。成本日志包含checkpoint和梯度审计的实际重算层入口，不将轮数视为等算力。
