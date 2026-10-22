# UND corrector：预训练能力继承审计

2026-10-22。范围：核对BAGEL原生训练和本地guidance-safe corrector，给出机制判断与实验建议；不修改训练代码、模型、数据或正在进行的评测。

## 结论

当前corrector继承UND初始化和部分文本/跨专家注意力能力，但与已验证的GEN-hidden caption接口不同（见用户澄清后的方向）。它把GEN的VAE输入嵌入交给UND patch stream，再用全新head预测速度残差。因此当前实验首先检验的是UND参数能否适配一个新回归任务，并没有单独验证原生视觉理解对纠错的贡献。UND作为速度回归器没有架构上的不可替代性；其最有价值的角色是通过原生视觉输入建立图像—文本的语义对应，提供可供生成侧消费的纠错信息。

## 原生训练与现有路径

原生依据：BAGEL论文§2.2–2.4、§4、§6，以及本仓库`qwen_latent_cot/bagel/modeling/bagel/bagel.py`。

- 文字和ViT token走UND专家；VAE token（包括clean VAE上下文）走GEN专家。
- SigLIP2/NaViT、connector、ViT二维位置嵌入构成原生视觉语义入口。clean VAE是另一条图像上下文表示，不能因此把VAE patch直接处理说成UND的原生输入方式。
- 文字位置由lm_head承担next-token CE；带噪VAE位置由GEN/llm2vae承担Rectified Flow MSE。共享attention使跨专家梯度能够传播，所以不能说UND仅受CE训练、完全未接收生成损失。
- 原生智能编辑依靠联合训练的理解与生成交互；论文报告ViT+VAE对Intelligent Editing优于仅VAE，不等价于证明UND能直接回归速度。

当前路径：

`proposal=x_t-t*v_g -> frozen vae2llm -> identity-initialized trainable input_projection -> time(t)+VAE 2D positions -> copied UND blocks -> copied UND norm -> new zero-initialized head -> delta_v`。

每层额外读取冻结prompt UND KV和同一次GEN前向的GEN patch KV。它没有调用vit_model/connector/forward_cache_update_vit，没有经过原生图像理解提示和文本回答路径。当前patch block是全双向patch注意力，不是自回归文本推理。没有输出推理文字不等于完全不能推理，但不能据此认定继承了原有VQA/推理计算流程。

冻结范围：native runtime整模冻结；复制的UND基础参数也保持冻结。UND的Q/K/V/O LoRA、input_projection和新head训练。称为“UND训练”时应指这些适配器，而非完整UND专家更新。GEN上下文原本参与跨专家attention，不能笼统称它们与UND不兼容；真正新增的分布偏移是让UND直接处理GEN VAE投影、proposal及生成时间嵌入，并改成稠密速度输出。

## 四个主要不足

1. 输入接口错位：继承了UND block，绕过了视觉语义编码器。冻结只保存参数，不保证新输入能激活原有能力。
2. 输出与任务错位：新速度head没有预训练的图像错误定位/纠正接口；UND的原生语言读出未用。零初始化保证初始不改图，但不提供速度预测能力；首步梯度先进入head也是该初始化的直接结果。
3. 监督不区分语义作用：当前loss为`MSE(raw_delta,H_g(epsilon-x0-v_c))`。H_g删负guidance投影。普通像素/latent误差修正、全局偏置和语义纠正均可能降低该loss；它没有逐项标明缺失物体、计数或关系。此处不是MSE不能学语义，而是当前证据不能把下降归因为UND语义能力。
4. 状态分布错位：训练在真实x0与独立噪声之间取x_t，部署在模型滚动轨迹上。高t的proposal本身可能不具可识别内容；它被作为近clean图像表示，却叠加原生成t嵌入，这属于新增条件设计，不是原生clean-image UND分布。guidance-safe解决单一向量投影约束，不能修复上述视觉语义接口。

已有可用证据：guidance_safe_20261022/RUN.md的Step100/1000人工对照记录了残留风格变化、计数/空间关系不稳定，以及部分变化同样出现在global基线。它支持优先调查语义接口与UND贡献；还不能取代同预算初始化/输入路径消融。

## 用户澄清后的方向（纯T2I，不能依赖ViT）

用户明确：必须维持基本文生图叙事，没有ViT输入；过去实验已证明GEN hidden可支持caption任务。采用该已知结果作为当前方向的依据。此前“优先恢复ViT输入”的建议不满足研究约束，撤回；原生ViT训练事实仍用于解释初始化背景，不能作为系统输入前提。

当前问题应改为：已验证可从GEN hidden读取语义的通路，为何没有有效传递到速度纠错？caption成功支持GEN内部表征含有UND可读的语义，不自动等于任意VAE投影或任意读出都继承相同能力。

当前corrector的主token输入是vae2llm(proposal)，GEN逐层信息以K/V上下文被读取；这与已成功caption的具体GEN hidden->桥接/序列->UND->语言读出路径不同。下一步优先恢复既有caption实验的层、特征处理、位置/注意力结构与已训练参数，再从同一UND语义状态接速度读出；不另建ViT路径，不要求生成外部输入图像，不自动增加caption CE。

首选可检验假设：在相同纠错训练预算下，继承已验证caption接口与权重，比当前UND初始化+VAE proposal主输入获得更好的语义改善/画质保持。caption能力可作为同一接口的诊断，但不能把caption损失降低直接当作纠错成功。

以下旧消融中C组属于ViT替代方向，不再是当前推荐实验；当前应将C组替换为“复用已验证GEN-hidden caption语义接口并接纠错读出”，并先找回原caption实验的实际配置与checkpoint。

最低限度的判别组，固定训练样本、采样seed、优化预算和可训练容量：

- A 当前UND初始化/VAE proposal路径。
- B 同接口、同冻结与适配预算，换为GEN专家初始化。
- C 恢复原生ViT/UND视觉入口，再接相同任务读出（报告额外视觉计算成本；另做等计算比较）。

A对B衡量UND初始化优势；C对A衡量恢复原生接口的价值。以native/global/corrected的同seed语义结果和画质一起判断，不以回归MSE决定胜负。随机冻结7B不是公平主基线，因为它同时破坏了预训练特征。

纯T2I SFT先改善生成器，但GEN-only训练不自动修复UND视觉入口。若之后换成新的SFT GEN底座，旧corrector看到的proposal、GEN KV、残差目标和global校准都改变；必须在新底座上重新匹配训练和比较，不直接把旧Step2000分数归到新系统。

## 来源

- Deng et al. Emerging Properties in Unified Multimodal Pretraining. https://arxiv.org/html/2505.14683v3
- 本地论文：`/Users/zyd/Documents/LCoT-codex/bagel.pdf`。
- 原生路由与训练损失：`qwen_latent_cot/bagel/modeling/bagel/bagel.py`，forward、forward_cache_update_vit、forward_cache_update_vae。
- 当前corrector：`qwen_latent_cot/correction/{model,runtime,attention,objective,training}.py`。
- 观察记录：`docs/guidance_safe_20261022/RUN.md`。
