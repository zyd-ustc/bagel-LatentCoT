# Design

BAGEL-LatentCoT 在生成专家与理解专家之间建立连续反馈通道。生成专家提供当前状态的内部表征，理解专家结合目标形成 memory，生成专家再消费 memory 更新预测。

## 理解与生成的交互

记目标文字为 P，图像潜变量为 x_t，时间为 t。生成专家 GEN 先给出基础速度 v_base 和内部特征 H_base；理解专家 UND 将 P 与 H_base 转换为 memory M；生成专家再次计算完整速度 v_loop：

\[
(v_{\mathrm{base}},H_{\mathrm{base}})=\mathrm{GEN}(x_t,t,P),
\]

\[
M=\mathrm{UND}(P,H_{\mathrm{base}}),
\qquad
v_{\mathrm{loop}}=\mathrm{GEN}(x_t,t,P;M).
\]

两遍 GEN 共享 x_t、t 和 P。交互结束后，外层采样器使用 v_loop 推进图像状态。新的采样状态触发新一轮特征提取与 memory 写入。

理解专家直接读取生成内部特征。连续 memory 承载与当前状态相关的信息，其表示由生成目标共同塑造。

## Memory 写入

键值表示（KV）由注意力的 key 与 value 组成。第 l 层目标文字 KV 记为 P_l，GEN 来源 KV 记为 G_l。Writer 的输入来源为：

\[
S_l=[P_l;G_l].
\]

Memory 从固定中性种子初始化，连续经过 UND 各层。记第 l 层 memory hidden 为 h_l，UND block 为 F_l：

\[
h_{l+1}=F_l(h_l;S_l).
\]

各层 hidden 经同层归一化和 K/V 投影导出 memory bank。GEN 来源包含当前图像 query 与图像边界 token 的特征。时间条件通过 GEN 特征进入 writer。

每次写入从中性种子开始，反馈内容随当前生成状态更新。

## Memory 消费

GEN 的条件上下文由目标文字、memory 和当前图像 query 组成：

\[
K=[K_P;K_M;K_G],\qquad V=[V_P;V_M;V_G].
\]

目标文字提供全局内容条件，memory 提供状态相关的反馈。Memory 使用独立的位置，图像 query 随附加上下文后移。导出的 memory key 在消费时应用对应位置的旋转位置编码（RoPE），来源 key 保留其原有位置。

条件分支预测反馈后的速度，再与无条件分支组合，形成无分类器引导（CFG）速度。

## Observe 初始化

Observe 让 UND 从 GEN 特征中恢复图像描述，学习主体、属性、数量和关系的读取。其 writer 来源为 GEN KV，生成来源保留目标文字条件。

文字读出与 memory 写入共享 UND。描述监督通过文字读出返回 writer，建立生成特征到理解表征的映射。交互生成阶段继续在这份初始化上学习反馈。

## 交互监督微调

监督微调（SFT）从正确图文对构造训练状态。设 x_0 为目标图潜变量，epsilon 为高斯噪声，时间从 t=1 的噪声端走向 t=0 的数据端：

\[
x_t=(1-t)x_0+t\epsilon,\qquad v^*=\epsilon-x_0.
\]

第一遍 GEN 提供停止梯度的基础特征。UND writer 与第二遍 GEN consumer 分别通过低秩适配器（LoRA）接受生成监督：

\[
\mathcal L_{\mathrm{SFT}}
=\mathbb E\left[\|v_{\mathrm{loop}}-v^*\|_2^2\right].
\]

梯度从完整速度预测进入 consumer，经 memory 返回 writer。原始模型权重、目标文字编码和基础提案保持固定。Writer 适配器作用于 memory 写入与 bank 导出，consumer 适配器作用于第二遍条件 GEN。

对应的干净图预测为：

\[
\hat{x}_0=x_t-t\,v_{\mathrm{loop}}.
\]

生成监督共同塑造反馈表示与消费方式。目标图提供状态构造和速度目标，UND 的状态输入来自 GEN 特征。

## 终图奖励优化

Flow-GRPO 从当前策略生成同一目标的候选图像，按终图满足目标的程度给出奖励。固定 SFT writer，以保持状态到反馈的映射稳定；consumer 根据同组候选的相对奖励更新。

记候选奖励为 r_i，组内均值与标准差为 mu_r、sigma_r，其优势为：

\[
A_i=\frac{r_i-\mu_r}{\sigma_r+\varepsilon}.
\]

策略在采样轨迹中选择随机转移，其余转移使用确定性更新。随机微分方程（SDE）的采样与重放共享高斯转移均值和方差。对下一状态 x_s，按潜变量维度平均的转移分数为：

\[
q_\theta(x_s\mid x_t,P)
=-\frac{\operatorname{mean}[(x_s-\mu_\theta)^2]}{2\sigma^2}.
\]

记旧策略分数为 q_old，概率比代理量为 rho=exp(q_theta-q_old)。策略目标采用裁剪后的组相对优势，并以固定 SFT 参考约束更新：

\[
\mathcal L_{\mathrm{RL}}
=-\min\left(\rho A,\operatorname{clip}(\rho,1-c,1+c)A\right)
+\beta\,\frac{\operatorname{mean}[(\mu_\theta-\mu_{\mathrm{ref}})^2]}{2\sigma^2}.
\]

其中 c 为裁剪范围，beta 为参考约束权重。采样时的旧策略用于计算比值，SFT 参考用于约束策略偏移。

## 显式文字反馈

显式路径先生成预览图，再由图像理解产生观察与修正文字。Writer 将目标和修正文字压入固定 memory，GEN 从相同初始噪声重新生成。

连续路径在每个采样状态读取内部特征并更新 memory。两条路径共享 memory 消费接口，分别提供文字反馈与内部特征反馈。
