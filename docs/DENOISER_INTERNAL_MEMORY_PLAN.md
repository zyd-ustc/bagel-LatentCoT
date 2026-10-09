# 原生早期图像观察＋完整 UND Memory 更新

NPU分支的运行适配与检查状态另见[NPU_RUNTIME.md](NPU_RUNTIME.md)。本分支还隔离条件编码的RNG，保留原生随机VAE后验，确保静态／观察配对使用相同视觉KV。

日期：2026-10-09。NPU分支状态：29项CPU／NPU数值测试通过。CPU模拟流程通过分片生成／评分／HTML、续跑和配对／文件完整性检查。NPU真实权重的原生上下文／velocity parity检查通过；正式质量评测未执行。此文件描述当前实现，不沿用10月5日旧方案的同层1/N循环、压缩或训练安排。

目标是在固定 x_t/t 下，通过可检查的早期视觉证据更新条件，然后重算 GEN。先只更新一次，冻结 BAGEL 全部参数，不引入 adapter、gate、alpha、压缩或训练模块。

## 执行路径

1. 按原生50点、shift3采样走到预定 step。先求原生速度 v0。
2. 依 BAGEL 的 x_t=(1-t)x0+tε、v=ε-x0，构造 x0_hat=x_t-t*v0。
3. 用原生 VAE decode 得到早期预测图。保存原始预测 latent；图像仅作原生显示范围裁剪。它不是最终图像，也不是已知正确的观察。
4. 调用原生图像接口：VAE＋ViT，完整保留图像 token，构建编辑用视觉上下文。
5. 原始 prompt 通过原生 tokenizer／embedding，从第0层连续经过全部28层 UND，以因果 attention 读取视觉上下文。保留完整 prompt、BOS/EOS、token顺序和原生追加位置。每层 KV 在 attention 输入处原生写入，禁止把 block 输出重新投影回同一层。
6. 条件使用完整视觉上下文＋文本 Memory。通过原生 prepare_vae_latent 和两个 CFG 准备接口重建 query 位置／索引；丢弃准备接口生成的随机噪声，始终使用同一个当前 x_t/t。
7. 原生 GEN 重算得到 v1，仅用 v1推进一次 Euler 更新。
8. 观察缓存在该调用结束后释放，下一采样步恢复原始文生图条件。不跨时间步缓存，不回送 H28 到第0层，不执行第二次 writer。

文本 Memory 容量和 hidden 维度不变，但完整视觉上下文增加了 token 和开销。新方案不声称对当前浅层循环等计算。

基线采样 CFG text4；本次重算使用 BAGEL 原生编辑 CFG text3、image1.5、interval(0.4,1]、global renormalization。静态／观察两组使用相同 CFG。相对 Base 的变化包含图像条件及 CFG 改变，不能全部归因于 Memory。

## 静态／观察对照

- STATIC：相同早期预测图，保留全部 GEN 图像上下文。prompt 从原生 embedding 经完整 UND，但编码时不读取图像 KV；保留与 OBSERVED 相同的 shifted文字位置。编码结束后将完整视觉前缀与文本 KV按原生顺序组合。
- OBSERVED：原生图像→文本 prefill，文本读取视觉前缀。这条上下文构造与原生编辑接口数值对齐。
- STATIC 是显式读取干预，不能称为完全原生的联合上下文。
- 两者的 token IDs、容量、位置、视觉前缀、重算噪声状态、t和GEN参数相同，仅 Memory writer 的图像读取不同。

text-removed CFG只保留源图像VAE＋ViT；image-removed CFG独立prefill原始prompt，使用该分支自己的原生位置。不跨分支复制条件。

## 首轮实验：40张图

8个既有prompt、seed0，Base＋step9 STATIC／OBSERVED＋step19 STATIC／OBSERVED。每个非Base arm只更新一次。step9的t≈0.9302，step19的t≈0.8257。它们是两个分别生成完整图像的配对arm，不是在一张图上更新两次。

只保留两项直接机制比较：同step OBSERVED对STATIC。仍报告各组对Base的结果，但不将其全部归因于Memory。不能凭变化或个例启动训练。

## 问答与追踪

每个arm取原有vqa_list前三个问题，仅传问题，不传目标答案。没有生成完整反馈文本。短答预算16 tokens；达到上限记录 incomplete，不把它当已验证回答。

四种只读probe：

1. native_vit_image：早期预测图的原生ViT-only理解上下文，校准观察能力。
2. native_edit_visual：原生VAE＋ViT图像上下文，没有文本Memory。
3. full_edit_context：视觉上下文＋Memory；答案可能直接来自图像，不足以证明Memory内容。
4. memory_only_diagnostic：移除视觉KV，只保留各层文本Memory及原生shifted位置；上下文发生改变，明确标为非原生诊断。

问题和生成的回答仅写入深拷贝。它们不进入GEN，不改变 canonical cache，不携带到下一采样步。标签待人工依据保存的早期预测图填写；不使用prompt目标或最终图像作当前观察标签。Memory初始化仍含prompt，因此存在间接复述风险。

每次更新保存early_prediction.png、event.json及state.pt。state包括x_t、t、v0、v1、未裁剪x0_hat、全部28层文本Memory K/V。event记录token IDs、位置、长度、图像及state文件hash、x_t/t hash、CFG、velocity变化、四种probe原始回答、完整性及probe时间。续跑和报告会校验文件。静态／观察配对会检查相同step的更新前状态、预测图、文字布局及CFG。完整图像KV未保存，可从绑定模型及早期图像重建。

原始latent和原生显示图有明确区别，人工问答只针对显示图；不能把高噪声预测图中的伪影当作可靠事实。

## 正确性与准入

CPU检查连续UND上下文与原生prefill一致、静态读取干预、容量／位置保持、probe不修改缓存、只在step9／19重算一次、x_t/t不变、失败后恢复原生方法。真实权重检查由用户启动，检查原生编辑上下文parity、缓存／权重不变、有限且可重复的velocity；不据此声称语义收益。

数值检查→人工观察/probe校准→OBSERVED对STATIC的Repair／Damage和质量→判断是否有信息增量。若图像本身不可判读，记录unknown；若full-context能答而Memory-only不能答，先排查读取接口，不直接宣布Memory无信息。没有成功参照前不蒸馏，不训练。

Looped-DiT借鉴仅为条件状态随深度连续传播、视觉信息参与条件更新。官方65a7705的DoubleStreamBlock拼接两路Q/K/V联合attention，并携带img、txt穿过body和各轮。本方案保持BAGEL原生的图像prefill→因果文本读取→GEN编辑顺序，不复现其对称joint attention、XSA、gate或GEN hidden跨body回送。

## 源码与入口

- BAGEL固定revision：056b5fd51a88c1eb4547318609e25d40080fcf87；来源hash见modeling/native_source.json。
- bagel.py：prepare_prompts／forward_cache_update_text、forward_cache_update_vit、forward_cache_update_vae、generate_text、generate_image／_forward_flow。
- 新核心：qwen_latent_cot/bagel/observation_memory.py。
- 配置：configs/observation_comparison.json；数据：data/observation8.jsonl。
- 入口仍只有scripts/compare_windows.py和compare_windows_8gpu.sh。默认配置已改为新方案。
- 历史两个配置仍可显式指定，用于复现旧评测；不覆盖远端已完成的7200张结果或冻结源码快照。
