# SenseNova-U1迁移评估：纯T2I内循环纠错

## 判断

适合作为第二底座验证，优先考虑未做8步蒸馏的SenseNova-U1.5-8B-MoT，不建议原样复制BAGEL corrector后直接切主线。主要优势是原生RGB状态、clean-image预测、端到端训练的浅层视觉入口和显式空间位置；共同使用MoT本身不是新增优势。保持外部输入只有prompt，不依赖外部图像或ViT。

证据范围：本地refs/SenseNova-U1，origin OpenSenseNova/SenseNova-U1，HEAD c35c83b（2026-09-24）；本地U1/U1.5报告、原生推理和训练源码；官方仓库与arXiv页面。未加载权重、未执行NPU兼容验证、未改该参考项目。

## 已核对的机制

- U1/U1.5没有BAGEL式预训练SigLIP和FLUX VAE；所谓vision_model在本地实现中是浅层patch embedding，两层Conv2d+GELU等。不能把类名NEOVisionModel误读为独立预训练ViT。
- UND和GEN视觉入口分别实例化为vision_model与fm_modules.vision_model_mot_gen；不是共享同一组patchifier参数。
- 非_mot_gen和_mot_gen分别有Q/K/V/O、norm、FFN，联合attention；UND读clean像素和文字，GEN读噪声条件像素。MoT仍有专家分工，不保证GEN hidden可直接作为UND原生输入。
- U1典型32×32像素/token；512px为256token，BAGEL当前16×16像素/token为1024token。4倍token缩减不等于4倍整体提速：U1 dense是42层/4096宽，BAGEL当前28层/3584宽，RGB输出也更大。参数文档给出U1总17.552B。
- U1.5保留32×32粒度，像素重建加入卷积/PixelShuffle以改善跨patch连续性。该优势属于原生生成头；若另起线性corrector头，不能自动继承。
- _t2i_predict_v先用原生fm_head获得clean像素预测x_pred，再做v=(x_pred-z)/(1-t)。t=0噪声、t=1干净，与本项目BAGEL时间/速度符号约定相反。训练是x-prediction、v-loss，不能改成无权重clean-image MSE还称原配方。
- interleave生成代码可把本次生成的像素规范化后经UND patchifier放回上下文。这是生成图像->理解的原生入口，不是去噪中GEN hidden->UND caption的直接证据。

## 对我们研究的意义

若允许重读内部生成状态，原生RGB clean预测可以直接由UND浅层入口读取，无需外部图像、ViT或VAE decode。外部任务仍然是纯T2I。这提供了更接近UND预训练分布的候选观察路径。

若要求只读GEN hidden、不重新patchify像素，则上述便利不直接成立。仍须验证并继承GEN-hidden caption接口，不能以“统一模型”替代该证据。先验证同一图像状态的hidden可支持内容、计数和关系的读取，再考察其输出对生成纠错的贡献。

不要原样移植当前vae2llm(proposal)->UND->新速度head；它会继续面临语义读取与速度输出同时改接口的问题。推荐保留UND语义读取的已验证结构，先接原生可用的clean-image/生成上下文读出实验，而非在各层加残差。

如果以后采用clean预测修正，δv=δx/(1-t)，临近干净端会放大；必须随本地time/noise_scale/CFG规范适配。当前guidance-safe的“去负投影”概念可迁移，但目标、时间网格、实际作用向量要重新定义，不能直接复用BAGEL t、残差或checkpoint。

## 工程与研究边界

本地training按Torch2.5.1 CUDA12.4/FlashAttention验证，另有ISP/并行/算子依赖；不是即用的Ascend64卡训练栈。本地推理有SDPA路径，说明可做NPU最小验证，不证明完整训练已支持NPU。U1.5报告描述SFT/RL/MOPD，但不能把报告叙述当作仓库已发布全部后训练实现。

第一阶段只回答三个问题：原生非蒸馏模型在同prompt评测中是否更好；GEN hidden->UND的caption语义继承是否更直接；同成本corrector是否相对该底座native/global改善语义并保持画质。当前BAGEL内部27B分数不能与其官方榜单分数直接比较。8步模型会混入蒸馏目标/轨迹差异，暂不作为第一次机制移植底座。

## 主要来源

- https://github.com/OpenSenseNova/SenseNova-U1
- https://arxiv.org/abs/2605.12500 （U1 §3.1–3.3）
- https://arxiv.org/abs/2609.11929 （U1.5视觉接口与训练）
- refs/SenseNova-U1/src/sensenova_u1/models/neo_unify/modeling_neo_chat.py：__init__、_t2i_predict_v、interleave生成后回读。
- refs/SenseNova-U1/src/sensenova_u1/models/neo_unify/modeling_neo_vit.py：NEOVisionEmbeddings。
- refs/SenseNova-U1/src/sensenova_u1/models/neo_unify/modeling_qwen3.py：双专家与位置编码。
- refs/SenseNova-U1/training/pyproject.toml、training/README.md、docs/parameter_breakdown_CN.md。
