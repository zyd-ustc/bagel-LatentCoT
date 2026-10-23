# 为什么当前 MoT 编辑比原 UND 输出速度的图片变化小

## 结论与比较对象

比较旧 guidance-safe UND 的 Step1000 与当前 MoT v2 的 Step2000。两者都完成1000次纠错/edit optimizer更新；当前前1000步为caption。采用相同32条prompt、seed17、512px和CFG4；逐像素检查发现32张native完全相同。数据、学习率、batch、结构和作用时间不同，下面区分实测变化和机制解释。

当前结果更接近native，主要问题集中在介入时间、反馈的信息路径和学到的修正方向。当前实际状态提交已经生效，旧v1的“预览大、实际只走一个Euler小步”和第二轮覆盖第一轮问题已修复。局部状态位移总量没有随最终图像变化同比缩小。

## 实测幅度

|量|旧UND guidance-safe Step1000|MoT v2 Step2000一轮|MoT v2 Step2000两轮|
|---|---:|---:|---:|
|相对native的RGB平均绝对差，像素归一化到0–1|0.089267|0.012765|0.020921|
|每次修正的平均raw delta RMS|0.181918|0.057226|0.058121|
|每次修正的平均applied delta RMS|0.180386|0.053287|0.052829|
|每张图各次局部提交位移RMS之和|0.041905|0.037098|0.073572|
|介入时间t，1为噪声端|1.000至0.758，连续25步|0.692/0.545/0.403各一次|同三节点各两次|

旧UND的最终像素差约为当前两轮的4.27倍、一轮的6.99倍。当前两轮比一轮变化更大，第二轮覆盖并缩小第一轮的旧机制已经消失。

局部提交位移之和不是最终latent差：各次修正方向不同，后续原生去噪还会改变扰动。当前两轮该数值为旧UND的1.76倍，说明把最终变化小归结为总注入量不足不成立。差异要结合时间、方向和后续动力学解释。

重新查看00/08/15/23四组：旧UND明显改变羊的数量、吹风机方向及霓虹背景、马的质感、跑步机房间布局；当前保留native主体与布局，主要改变边缘/表面，08两轮新增烟状背景。旧UND的变化也包含风格漂移和错误重构。

## 1. 纠错从早期连续介入移到了中后期稀疏介入

旧采样在49次Euler更新的前25次持续预测delta，作用于t=1至0.758的高噪声段。当前在index28/35/40介入，此前28次更新完全由native完成。两段时间不重叠。

这改变了任务：旧UND在图像布局形成过程中反复改变速度场；当前UND先看到更清楚的proposal，再编辑已经形成主体关系的状态。后者适合读取，但数量、位置的更改需要更有针对性的结构反馈。当前图片保持native构图，与这个时间变化一致。各节点到最终图像的扰动传播尚无逐节点测量，因此不把全部4.27倍差距单独归于时间。

## 2. 原生GEN头与空间残差头有不同的适配路径

旧UND读取完整空间patch，注意力每层可访问prompt及GEN图像特征，再用可训练input_projection和可训练FP32 dense head直接产生空间delta。当前路径为：

GEN clean特征 → observation UND每层16-slot KV → 结合目标的feedback UND每层16-slot KV → GEN editor → 冻结原生llm2vae → w → w-vc。

保留的是每层KV，不是只有一个末层16-token张量；但空间信息仍经过两级16-slot接口。当前editor保留实际带噪图像输入，可依靠GEN预训练能力完成普通去噪；目标的新增语义和图像错误反馈需要穿过该接口。仅取消editor原prompt cache，并未消除实际latent已携带的native语义状态。

实测当前w-vc RMS0.0581，应用后0.0528，仅约guided基础速度RMS的3.39%。这与editor主要保持既有去噪预测、反馈尚未产生有效方向的解释相符。现有日志没有分离“恢复prompt信息”和“根据当前画面诊断错误”的各自贡献。

不应把完整速度loss本身说成压小残差的原因。记r为投影后的目标残差，则：

    ||w-(vc+r)||² = ||(w-vc)-r||²。

vc在当前loss中detach。固定状态下两种写法对w的梯度相同。差别来自模型参数化、可用信息和优化状态，而非MSE被基础速度的数值“淹没”。

## 3. 当前UND读取尚未给出可靠的主体信息

observation Step1000/2000各8张proposal的主体清楚；开放caption没有读准主体。Step2000仍描述暗矩形、阴影、手指或纹理，遗漏羊/鸟、面包、猪/雨伞、兔子/自行车等主体。记录和图册见observation_002000。

caption阶段用真实图像latent训练，推理读取生成过程的clean estimate；输入分布不同。进入edit阶段后observation虽继续更新，loss已换成速度MSE，不再保留caption CE。解冻解决了参数可更新性，没有直接建立或维持准确的图像描述。

caption失准直接说明当前语言读出未对齐。连续KV是否仍携带部分有用信息，需要与它的实际输出贡献区分。当前32题结果未出现清楚的数量、关系修复，因此现有证据支持把反馈内容列为首要问题。

## 4. 训练强度也明显下降，原先的大梯度主要来自新输出头

纠错更新次数相同的两个窗口：旧Step901–1000，当前Step1901–2000。

|量|旧UND|当前MoT v2|
|---|---:|---:|
|learning rate|1e-4|1e-5|
|LoRA rank|64|32|
|global batch|256|64|
|1000次纠错更新的图文对抽样数，含重复|256000|64000|
|loss|0.414594|0.441918|
|本窗口同目标zero_correction_mse|0.453764|0.440695|
|UND/observation梯度范数|0.011785|0.010654|
|可训练input_projection梯度|0.258439|无此模块|
|可训练输出head梯度|1.279668|原生head冻结|
|feedback/editor梯度|不适用|0.005542 / 0.005576|

旧UND对自己的zero baseline拟合更好；当前与自己的zero baseline接近，仍稍高。两组目标状态与数据不同，表中绝对loss不作跨run质量排名。

原先1量级的head梯度对应一条可以直接改变输出的可训练通路。当前删去了该通路，只靠LoRA经冻结原生GEN头改变输出；observation梯度与原UND梯度本身在相近量级。总梯度缩小约百倍不表示UND梯度消失。当前全局loss除batch后执行SUM all-reduce，没有重复除64；总范数约0.0133，clip1未压缩它。AdamW也不使参数更新量简单正比于原始梯度范数。

lr、rank、样本曝光和数据均一起变化，这些足以让原生接口适配进度不同。它们影响多少，目前没有单独分离。

## 5. 当前重加噪系数与guidance-safe各自缩小了多少

当前提交：

    delta = H(w-vc)
    x_new = x - t*(1-t)*delta

三个t*(1-t)约为0.213、0.248、0.241；相对clean preview的-t*delta，保留比例1-t约为0.308、0.455、0.597。这是保留同一估计噪声时，clean改动映射回当前flow状态的混合系数。相比完整clean位移确有缩小；但旧UND乘的是Euler dt，这几个数并非相对旧方案的净衰减。

当前两轮平均raw/applied delta RMS为0.058121/0.052829，即平均范数下降约9.1%。guidance-safe没有消掉大部分向量；原始输出已经较小。旧UND guidance-safe为0.181918/0.180386，下降约0.84%。两者均使用同一负guidance投影移除规则。

跨轮editor状态梯度保留；native重算、图像特征提取和目标均detach。两轮速度MSE训练局部动作，没有沿完整后续采样轨迹优化最终数量/关系。局部loss下降和最终语义改善之间仍存在这层距离。

## 判断与下一步优先级

当前机制更接近“已有GEN去噪能力上的弱连续条件适配”，尚未形成准确读图、识别缺失内容并驱动结构修改的闭环。最终变化小同时包含风格更稳定和有效语义编辑不足。

优先处理observation对生成proposal的读取和向feedback传递的语义，再决定如何让反馈在仍能改变结构的时间段介入。保留原生输出头与UND→GEN叙事；单独放大现有delta会放大当前表面变化，未直接补足缺失的对象/关系反馈。此次只分析，没有修改代码、训练配置或运行进程。

## 可复查证据

- 数值：`artifacts/mot_state_edit_8000_20261023/amplitude_analysis/metrics.json`
- 四例实际对照：同目录`comparison.jpg`。
- 当前完整32题图册：http://127.0.0.1:8775/semantic_002000/index.html
- 当前caption：http://127.0.0.1:8775/observation_002000/index.html
- 旧实现：`qwen_latent_cot/correction/{model,attention,runtime,sampling,training}.py`；以上核心model/sampling相对旧run8d8004d未变。
- 当前实现：`qwen_latent_cot/mot_edit/{model,roles,sampling,training}.py`，部署revision4b23d8b。
- 配置：`configs/correction_guidance_safe.json`与`configs/mot_edit_8000.json`。
