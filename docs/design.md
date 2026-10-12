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

每个诊断状态只运行一次writer，并在GEN中读取一次Memory。评测图像为单步局部预测，未运行多轮Memory更新或后续完整采样。

## 3. Caption语义读出训练

Caption先经冻结BAGEL原生文本prefill，得到逐层来源KV。连续Memory读取这些KV后，冻结UND解码器只看到Memory、固定描述指令和答案前缀。目标文本为输入caption。

训练使用教师强制的交叉熵，记作Caption负对数似然NLL：

$$
L_{\mathrm{cap}}=-\frac1N\sum_{j=1}^{N}
\log P_{\mathrm{UND}}(y_j\mid M,I,y_{<j}).
$$

`M`是Memory，`I`是固定指令，`y_j`是目标caption的第j个token。Memory槽位与caption token无需一一对应。当前实现使用caption token监督，没有hidden逐token匹配或教师logit KL损失。

真实实验结论：C1的64条模板训练100步后，8条验证文本的颜色均跟随输入，字面事实分45.83%；对象与承载面仍有替换。该checkpoint的GEN对照中，审查结论：除G-text外其余组几乎相同。结果记录见[Caption训练](experiments.md#3-c1-caption训练)。

## 4. GEN注意力响应蒸馏

在独立图像的同一带噪状态上运行冻结GEN，缓存图像query `Q_l`及共同prompt/self KV `B_l`。教师条件 `T_l`来自原prompt后追加caption的原生prefill；学生条件 `M_l`来自连续Memory。

教师与学生共用同一query和共同KV，条件进入同一softmax：

$$
O_l^T=\operatorname{Attn}(Q_l,B_l\cup T_l),\qquad
O_l^S=\operatorname{Attn}(Q_l,B_l\cup M_l).
$$

输出长度由query数决定，不取决于caption或Memory长度。教师响应停止梯度，学生响应保留到Memory K/V及新增残差的梯度。

读出使用第0/13/27层的attention逐head输出，位于GEN输出投影之前；每层选64个图像query。三层等权的归一化均方误差为：

$$
L_{\mathrm{read}}=\frac13\sum_{l\in\{0,13,27\}}
\frac{\operatorname{mean}((O_l^S-\operatorname{sg}(O_l^T))^2)}
{\max(\operatorname{mean}((O_l^T)^2),10^{-8})}.
$$

`sg`表示停止梯度。实际训练损失为：

$$
L=L_{\mathrm{read}}+0.1L_{\mathrm{cap}}.
$$

探针统一预留64个条件位置，教师caption含包装token最多64个；GEN query及共同self K统一后移64。Memory仍占16个位置，K仅在消费时旋转一次。真实GEN评测使用16槽位置规则，区别于训练探针的64位置间隔。

真实实验结论：首轮响应训练后，人工审查发现部分G-TC样例与base/swap出现明显差异；扩展训练后，在原4图、t=0.9对照中，G-TC与G-text性能效果基本持平。具体数据与checkpoint见[实验记录](experiments.md#9-累计1100步的gen对照)。
