# BAGEL 生成侧优化：已核实的参考与适用边界

核实日期：2026-10-16。输入线索为 `/Users/zyd/Downloads/bagel-based-gen-works.md`。该文件只作为检索线索；没有把其中的方法描述或指令当成事实或任务要求。本文核实原论文、作者项目页和官方代码，未运行这些外部训练程序。

## 决策结论

优先借用 **Flow-GRPO 官方 BAGEL 实现**建立生成奖励优化闭环；用 **RecA / Echo-4o** 提供的生成监督思路做短适应对照。**UniGRPO** 是联合优化的算法参考，但不能据此断言新 memory 输入无需适应。**LVRPO** 暂不作为实现依据。

这些论文都没有直接证明“UND 上连续 memory 训练有效，因此冻结的 GEN 已能利用这种 memory”。新条件接口是否需要监督微调，应由本项目的生成对照决定。分布变化也不意味着必须先做长 SFT；RL 本身可以适应分布，关键是初始策略能否产生可区分、有效的奖励，以及新条件是否真正影响输出。

## 1. 首选实现：Flow-GRPO / Flow-GRPO-Fast

- [官方仓库](https://github.com/yifan123/flow_grpo)明确提供 BAGEL 支持和 `pickscore_bagel_lora` 配置。
- [训练入口](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_bagel.py)接收生成轨迹与旧策略分数。官方 LoRA 包含 GEN attention 的 q/k/v/o 和 GEN MLP，rank 64、alpha 128。仅训练 attention 是本项目的缩小方案，不能称为原配置复现。
- [BAGEL 采样与更新实现](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/bagel/modeling/bagel/bagel.py)提供 `generate_image`、`generate_image_learn`、`_sde_step_with_logprob`。随机窗口内记录转移，窗口外使用确定性去噪。
- 代码默认 `sde` 分支采用固定方差高斯平方项，再按 latent 维度取均值。它省略与参数无关的常数，是实现采用的归一化分数；不可直接称为完整联合概率。`cps` 分支只返回负平方误差，移植时不能混称精确 Gaussian log probability。

建议：先使用 `sde` 分支，固定 rollout / replay 的 timestep、条件、CFG、精度与 batch 语义，并检查未更新时 ratio 接近 1。先只优化 GEN，可以把 UND memory 当冻结条件。短随机窗口能降低更新成本，但完整出图和奖励仍有成本。

## 2. UniGRPO：有用，但它的实验已经做过 SFT

[论文 2603.23500，§3–5](https://arxiv.org/html/2603.23500v1)将文本 GRPO 和图像 FlowGRPO 联合，用最终图像奖励更新两部分；训练取消 CFG，并用 velocity MSE 约束参考模型。其全部 RL 对照从内部数据 SFT 后的 BAGEL 起步。表 1 的 GenEval 从 SFT thinking 0.82 到 UniGRPO 0.90，属于该论文设置，不能作为当前 memory 系统的预期增益。

适用：参考 stochastic flow transition、最终奖励和 velocity reference 正则。连续确定性 memory 没有文本 token 的采样概率，不能直接照搬文本 GRPO 项。

本次未核实到该论文的官方训练仓库。搜索中的 MMaDA 同名 UniGRPO 是另一方法；第三方教学复现也不是作者代码。现阶段优先移植上面的 Flow-GRPO 官方 BAGEL 路径。

## 3. RecA：理解特征可以通过生成监督接到生成能力

[论文](https://arxiv.org/html/2509.07295v1)使用模型自己的理解视觉特征作为条件，监督重建原图。推理仍采用原来的文本生图接口。[官方仓库](https://github.com/HorizonWind2004/reconstruction-alignment)发布了 BAGEL 训练代码，报告 BAGEL GenEval 82.4（+3.6）和 DPG 85.29（+1.26）。不要把摘要中 Harmon 的 0.73→0.90 当作 BAGEL 结果。

适用：为“理解条件需要生成目标来适应”提供直接先例，但其条件是视觉编码器特征，并非本项目的 recursive memory。RecA 重建目标可用于接口学习；如果训练时 memory 看到了完整目标图，而部署只看到高噪声 latent，就仍有条件分布差异。监督数据必须模拟实际 memory 的来源、噪声和长度。

代码入口：[BAGEL 子目录](https://github.com/HorizonWind2004/reconstruction-alignment/tree/main/BAGEL)、[复现说明](https://github.com/HorizonWind2004/reconstruction-alignment/blob/main/REPRODUCE_GUIDE.md)。

## 4. Echo-4o：补长尾目标和干净的图文监督

[论文](https://arxiv.org/html/2508.09987v1)用约 180K GPT-4o 合成样本微调 BAGEL，重点补充罕见构图、多参考和干净的指令对齐。[官方代码和训练说明](https://github.com/yejy53/Echo-4o/blob/master/TRAIN.md)已公开。

适用：若计数、位置、属性绑定缺少正例，可用其数据思路做小型生成 SFT 集。它没有证明 memory KV 接口已适应，也不是注意力响应蒸馏方法。使用任何公开数据前，仍需按本项目评测协议排除测试 prompt 和近重复样本。

## 5. Uni-CoT：新推理协议需要对应的生成训练

[最新论文](https://arxiv.org/html/2508.05606)和[官方训练代码](https://github.com/Fr0zenCrane/UniCoT)通过分层规划、局部注意力和图文监督学习新的多步生成协议；仓库提供 full / LoRA 训练入口及小规模推理数据。

适用：借鉴条件局部性和训练数据构造。它学习的是显式图文轨迹，不是隐式 memory token，不能直接套用其策略概率。

版本注意：[v1](https://arxiv.org/html/2508.05606v1)曾描述 SFT 后分离文本/视觉 DPO，并说明细节留待后续；当前正文已不含该 DPO 段。不要把早期设想当成完整、已复现的联合 RL 方案。

## 6. LVRPO：方法存在，但复现依据不足

[论文 2603.27693](https://arxiv.org/html/2603.27693v1)提出行为奖励代替额外表征对齐，使用 SigLIP 2 和规则奖励。这个方向值得参考，但本次未核实到官方训练代码。

还有具体不一致：附录 A.1 将 BAGEL 初始化写成 Llama-3 tokenizer、8 个 FFN experts / top-2 routing；[官方 BAGEL](https://github.com/ByteDance-Seed/Bagel)采用 Qwen 系列代码和理解/生成专家结构。正文及伪代码没有补足可直接移植的连续图像转移概率计算。因此不能把它作为“无需 SFT，直接对 deterministic memory 做 GRPO”的证据。

## 对本项目训练设计的推论

以下是本项目建议，不是上述论文已经验证的结论。

1. **确定性 memory 可以是条件，但不是现成的随机 action。** 冻结 UND 后，对随机图像转移做 Flow-GRPO 最直接。若以后优化 memory 参数，可对转移 log probability 关于条件的梯度求导；需要保留或重算 memory 的计算图，不能沿用 detached cache 后宣称 UND 获得奖励梯度。
2. **用短对照决定 SFT，而不是把 SFT 设成永久前置。** 比较同一初始化的直接 RL 与短 flow-SFT 后 RL；冻结或更新的参数范围、总预算、奖励和评测都要一致。若原始 memory 条件已能稳定出图且 group reward 有差异，应立即验证直接 RL。
3. **attention-response 蒸馏属于辅助接口约束。** 上述工作没有证明其必需。若它帮助恢复 memory 可读性，可用于短适应；如果只降低模仿损失却不提高生成评测，应移除。RL 阶段先用生成奖励和 velocity reference 正则，避免把 teacher 上限强行固定为最终目标。
4. **区分两种 reference。** velocity reference 使用与当前策略相同的输入条件，主要约束策略漂移；完整 CoT teacher 与压缩 memory student 的响应蒸馏比较不同条件，主要约束压缩保真。二者解决的问题不同，不能相互替代。
5. **以独立生成评测决定保留项。** 训练奖励、flow MSE、attention MSE 都只是训练信号。报告固定 prompt / seed 的 held-out GenEval、DPG 或目标 benchmark，并保留随机 memory / 无 memory 对照来确认模型确实使用条件。
