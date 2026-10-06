# 分层 Memory KV：冻结 BAGEL 文生图内部循环

日期：2026-10-06。状态：training-free 代码已实现；新路径真实权重的图像质量和语义收益尚未验证。旧 hidden 方案存档在 history/20261005_memory_hidden_plan.md。

## 目标与实现依据

在同一次 denoiser 调用内增加理解与生成的交互，让 Memory 提供与当前生成状态有关的信息，并保留原生文生图能力。`x_t,t` 固定，只有最后一轮的输出进入原生采样器。ViT 仅用于离线观察或今后的理解监督。

原生 BAGEL 的 UND/GEN 使用各自的 norm、QKV 和 MLP，随后进入共享 attention。原生 `forward_cache_update_vae` 已支持写出 GEN KV；`PackedAttentionMoT.forward_inference` 写出的缓存是**当前层输入**经过原生 norm、投影、K norm 和 RoPE 得到的 KV。本实现直接使用这些接口，不将更新后的 hidden 重新投影成当前层 KV。

阅读 [Looped-DiT 官方实现](https://github.com/OpenSenseNova/Looped-DiT/blob/92a9c1914258361f78e03426588c7c215944b504/looped_dit/model.py) 的 prefix/body/suffix 与最终 decode 控制流程，借鉴连续 body 共享计算与普通 inference 只读出一次的结构。其训练后的图像 hidden recurrence、XSA、gate 不移入 BAGEL，也不作为 training-free 成功的证据。[第一篇 hidden-state refinement 论文](https://arxiv.org/abs/2608.29160) 的匿名补充代码未找到可验证的公开仓库；没有声称使用其开源实现。此前已失败的 `1/N` 不再使用。来源、版本和文件 hash 见 LAYERWISE_MEMORY_SOURCE.json。

## 状态与计算顺序

`s,e` 是 body 的半开层区间。`r=0` 是第一次 GEN 运算，R 是额外轮数。`P_l` 是不可修改的原 prompt KV；`GKV_{r,l}` 是 GEN 在原生层 l 的输入 KV。`M_{r,l}` 是上一轮 writer 写出的同层 Memory KV。所有箭头均表示当前运算读取哪个状态。

1. 按原生 UND 路径 prefill prompt。选取至多 K 个不同内容 token，排除特殊 token，保留原位置。在层 s 入口捕获它们的原生 hidden。短 prompt 不复制 slot。
2. GEN prefix 原生执行一次，保存 GEN 与边界的固定 body 入口。原 prompt KV 始终保持不变。
3. 第 r 轮 GEN 从固定入口开始。层 l 读取 `P_l + M_{r,l}` 和当前原生 GEN query；r=0 尚无 Memory bank，因此整个 body 是原生计算。每层完整执行原生 attention、MLP 与残差。
4. 如果还有下一轮，从固定 prompt hidden 入口启动 UND writer，按 s 到 e−1 的正常层序计算。层 l 读取 `P_l + M_{r,l} + GKV_{r,l}`，并保存其原生输入 KV 为 `M_{r+1,l}`。writer hidden 仅向更深层传递，不回送入口。
5. 下一轮 GEN 仅从对应层的 Memory bank 读取反馈。第一层 s 的 writer 输入尚未观察 GEN，故不将这一静态副本加入 bank；有效读层是 `[s+1,e)`。
6. 最后一轮 GEN 完成后，suffix 原生执行一次，不增加 Memory overlay。原生 norm、llm2vae 和 CFG/采样器计算最终 velocity。最后一轮不运行无人读取的 writer。

writer 的层 l 输入只能包含更浅层已经读到的 GEN 信息；同层 attention 输出形成层 l+1 的 hidden。这正是原生深度关系。每轮重置 writer hidden，但上一轮 KV 可被新 writer 读取，因此历史只通过分层 KV 保留。

所有 Memory bank 都局限于一次 denoiser 调用。prompt seed 仅在当前 prompt prefill 与生成间保留，生成结束后清除。无内容 prompt 不创建 Memory；text-removed CFG 是原生分支。批内不同图像和 prompt 长度使用 packed 索引，样本之间不读对方 KV。

## 旧对照与限制

`MEMORY_LOOP` 完整保留旧 hidden recurrence。它使用边界均值加 noise、在 body 入口回传末端 M hidden、suffix 读取 M，且 null CFG 也自建 M。新路径改变了 slot 来源、反馈类型、suffix 与 null CFG 行为。因此，新旧对比是方案对比，不能宣称仅改变 KV 格式。

GEN 读取额外 UND KV 会改变 attention 分布；保持原生算子和层序不能保证图像质量。BAGEL 训练 mask 会阻止其他区段读取 noisy image，因此 UND writer 能否从 noisy GEN 提取可用于语义编辑的信息仍待验证。slot 具有文本含义只说明 seed 的来源，不能证明反馈携带正确的对象、数量或空间关系。

## 用户执行与判定

先运行真实权重 E0：R0/native、no-read/native、prompt cache 不变、packed 分辨率，并报告 Memory/native 的 conditional 与 CFG velocity 差异。velocity 非零只证明读取路径有影响。

再运行同 prompt/seed 的 Base、旧 Memory loop、分层 KV 三组配对生成。默认 `[0,8)`、K≤8、R1、全部采样步。报告 GenEval2 atoms/GM、quality proxy、invalid、Repair/Damage 和按 prompt 聚类的区间，并检查配对图像。已有的 384 prompt×3 seed 确认集不重新列为待办。

可选 Memory QA 读取 GEN 实际使用的分层 bank。DYNAMIC 与 SEED/EMPTY/VIT_IMAGE 使用相同问题和答案协议。新 SEED 是所选原 prompt 内容的同层 KV，因此它是明确的文本回声参考；不导出或读完整 prompt cache。按观察到的 x0 图像标注，重点报告 prompt 与图像不一致时的准确率，不能只问“是否符合 prompt”。ViT 不进入生成。

本次不实现训练。只有 training-free 图像结果支持语义收益和质量保持，才讨论理解侧格式/隐式适配及 deep supervision。质量 proxy 和 QA 都不能单独证明成功。代理只运行 CPU 合成与接口检查；正式 GPU 任务由用户启动。
