# 从历史读图接口到跨时间步反馈：修改建议

状态：2026-10-23只读分析与方法建议。未改变当前8000步run、数据、预算或配置。本记录是基于现有实现的优化方案，不作新方法的论文新颖性主张。

## 历史查证：C1和C2-observe要分开

anchored-dual-state-loop分支diagnostics/caption.py的prepare_source明确规定：普通C1调用runtime.text(row['caption'])，source为UND_native_text_oracle；反馈C1读取prompt_and_correction。它们不是从GEN图像hidden观察画面。C1说明文字语义可通过连续KV传递给UND/GEN。历史记录：100步NLL0.458495、字面事实45.83%；响应扩展1100步response0.018171、NLL0.163620。文字压缩与响应成功应保留在这个层面。

后来的C2-observe确实读取图像。phase1a-opd的diagnostics/runtime.py:training_state在observe且无保存state时使用neutral_prompt；visual_objectives.py调用conditional_gen取图像source，然后student.write(source,t)，在拼接goal prompt之前返回。训练为28层UND完整Q/K/V/O及Q/K norm，t∈[.05,.25]，20000步。writer与caption decoder共享被训练的原生UND参数；lm_head保持原生。

旧结果observe_nativeund_results_20261016：Step19750 NLL2.038061，关闭source为2.718276；113条记录中112条source NLL更好。开放caption出现市场/红气球等图像相关内容，也存在把加油机叫消防栓等错误。source依赖有实证，精确对象/关系读取仍有错误。早先observe_full_20261015_153700的1000步则多为人物套话，应与后来的nativeund20k分开。

因此“历史成功只是重建prompt”对C1成立，对后来的C2-observe不成立。

## 当前接口遗漏的继承

当前mot_edit/model.py只在observation role上训练QKVO LoRA；caption_loss/caption_text调用roles.native_und_forward，后者直接使用冻结native UND。因此writer输出连续KV后，冻结的原生读出器必须直接理解新KV分布。旧C2的writer和caption decoder会共同适配。当前还缺少旧caption读出器的显式任务instruction，只从BOS续写。

建议统一observation writer与caption reader的角色参数：同一组observation UND适配器分别用于非因果读图和因果caption读取，保持各自attention mask，保留冻结原生lm_head。先用当前已授权图文对的真实图像，在中性GEN上下文中训练实际图像描述；目标P只进入后续feedback比较模块，不进入observer图像来源。保留caption任务以维持语义接口，不在固定1000步后完全切为速度MSE。所需训练配比/预算属于新方案确认项，未修改当前run。

保留16 slots作为起点：旧C2同样使用16 slots，现有历史不支持把槽数本身判定为根因。当前两个串联接口仍增加适配负担；先恢复已有效的共享读写参数和任务格式，再决定是否改变容量。

caption标签要描述实际输入图像。对真实配对x0可用其caption；对于生成失败proposal，原目标P是要求，不是实际画面描述。已有历史observed标签只用于其对应图像/状态。自然图文对本身没有提供所有生成失败状态的准确caption，这一监督缺口仍需在后续方案中处理，不能把P复述当作读图成功。

## 跨时间步：优先选保存状态回退，再持续带反馈重走

候选比较：
1. 同时间放大delta：增加扰动，不解决结构形成时机与反馈持续性，不选。
2. 从晚期clean estimate估计噪声并重加噪到早期：可实现，但额外依赖clean/noise估计，容易引入分布偏差；保留为后续备选。
3. 保存早期真实latent，在较晚节点读取反馈，再恢复早期latent并带反馈重走：推荐。省掉反演与噪声估计，保留原seed、prompt与已执行状态来源。

采用现有50节点shift3网格的一个具体起点：保存index12的t≈.9024状态，到index33的t≈.5926读取。精确节点仍是待实施选择，非当前run改参。

1. 从同一seed正常去噪，保存x_early。
2. 在t_read从当前状态形成proposal并抽取GEN特征，不经ViT或图像解码。
3. observation读出实际状态O；feedback以P、O以及上一轮反馈形成连续计划M。P不替换、不删除。
4. 恢复保存的x_early，保留M。M来自刚刚执行过的较晚状态，是算法工作记忆，而非尚未计算的未来信息。
5. 从t_early重走到t_read的每个采样步，让GEN同时读取原生P与M，使用原生llm2vae输出速度。M跨步持久；不采用三个节点的独立一次性delta。
6. 到达t_read后UND重读实际修改结果，更新M。若执行第二轮，携带更新后的完整反馈再次回退重走。最终保留最后一轮轨迹，继续到t=0。

原prompt继续作为用户目标，M提供当前尝试哪里做错、需要保持什么。建议保留当前guidance-safe部署语义：在同一state/t计算原conditional vc、guided vg与反馈条件速度w，v_used=vg+H(w-vc)。这样原生CFG存在且反馈每一步都改变速度场；它没有在模型各层增加新的残差输出头。连续M通过原生注意力条件接口进入GEN，与旧C1响应路径接轨。

回退后若仍使用原P和完全相同的native计算，确定性Euler会重走原路径；所以反馈必须改变重走区间的条件速度。单独回退或单独重加噪都没有解决语义纠错。

两个不同的时间必须分开：t_read控制何时可以读懂；当前t_write控制GEN输入时间嵌入与各步速度预测。M缓存未旋转语义KV，消费时按当前合法前缀布局计算RoPE；每步重算image query和当前时间，禁止把t_read速度张量用于t_write。

## 训练对应的改变

现训练把读图与执行绑定到同一个t，且主要是配对图像加噪的局部两轮MSE。跨时间版本需要在t_read形成反馈，再在较早t_write的状态上训练其GEN响应；采样端持续使用这份反馈，训练也要覆盖这一时间配对。目标继续由配对x0与实际visited state构建，保持原生头。真实回退轨迹的噪声端条件和单纯x0线性加噪并不相同，状态来源要随实现一起对齐。

历史C1可以复用为“连续计划能被GEN执行”的接口初始化，C2-observe则负责“计划来自实际画面”。两者各自承担其已有证据支持的任务。仅靠自然配对MSE仍可能学习普通去噪；准确读取与实际修正方向必须通过样例确认。

## 计算与叙事

单用户prompt、单seed、无外部图像输入、无推理期文字反思解码、最终一次VAE解码。时间表非单调，但仍是在一个采样调用内实现GEN→UND→GEN反馈。

49次原始Euler更新，index12到33每重走一轮增加21次；一次回退约70次，两次约91次，另计UND/条件GEN调用。这个版本用显式计算换取布局修改窗口；后续压缩计算是另一阶段，不把它记成原49步成本。

“早期更容易改变布局”与历史图像变化一致；变化更大不等同于更正确。CFG4也不证明原生动力学一定抹平全部扰动，seed并非对全局布局的不可改变约束。

## 文献增量与已有历史

复用：本项目C1连续文字压缩/GEN响应蒸馏、显式反馈early10/20重启、C2 native UND observe20k、当前v2状态提交与32图结果。

新增针对性查阅：
- Meng et al., SDEdit, ICLR2022，https://arxiv.org/abs/2108.01073 。加噪后重新去噪用于编辑；支持“重开结构修改区间”的邻近机制。本方案优先恢复保存状态，而非从解码图像加噪。
- Lugmayr et al., RePaint, CVPR2022，https://openaccess.thecvf.com/content/CVPR2022/papers/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.pdf 。扩散时间跳转/重采样用于修补；说明非单调采样时间已有先例，任务是已知区域条件的inpainting。

尚未展开：跨时间连续语义反馈与同类测试时自纠错的完整论文重叠。当前只确定工程修改方向，不提交新颖性结论。

## 证据位置

- 历史C1源码：anchored-dual-state-loop:qwen_latent_cot/diagnostics/caption.py的prepare_source。
- 历史显式早期反馈：同分支diagnostics/explicit_feedback.py的compare/sample_fixed。
- 图像observe源码：../bagel-LatentCoT-phase1a-opd/qwen_latent_cot/diagnostics/{runtime,visual,visual_objectives}.py。
- 图像observe原始记录：../artifacts/observe_nativeund_results_20261016/{report.json,training_summary.json,validation.jsonl}。
- 当前接口：qwen_latent_cot/mot_edit/model.py:caption_loss/caption_text与roles.py:native_und_forward。
