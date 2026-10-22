# 原生 T2I 后训练入口核对

实现前核对（2026-10-22）；现有入口与验收状态见[纯T2I使用指南](t2i_sft.md)。以下记录实现前的差距。

建议先做 GEN-only 图文对 Rectified Flow 微调。解冻全部 `*_moe_gen`（Q/K/V/O、QK norm、MLP、层 norm、最终 norm）及 `vae2llm`、`time_embedder`、`llm2vae`；冻结文本 UND、VAE、ViT/connector。依据：[GEN 路由](../qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py#L395)、[层参数](../qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py#L700)、[生成入口](../qwen_latent_cot/bagel/modeling/bagel/bagel.py#L75)。这会更新原生生成器，不是另加纠错头。

原生训练 `forward` 已返回速度 MSE（[L190–222](../qwen_latent_cot/bagel/modeling/bagel/bagel.py#L190)），但项目尚无原生 T2I 训练脚本。加载器默认全冻结；现有 `Runtime.predict` 与 `_forward_flow` 均为 no-grad，不能直接训练。

冻结 UND 时，文本 KV 可在 no-grad 下计算；GEN 的注意力及后续计算必须保留梯度。建议独立可微速度入口复用原生 GEN 模块，勿全局删除推理装饰器。原生 train attention 仍使用 CUDA efficient SDPA / compiled FlexAttention（[L455–488](../qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py#L455)），现有 NPU 适配在推理 SDPA 路径；需要实测原生 GEN 的前反向和显存，再选全量参数分片或 LoRA。当前已跑通的 UND 纠错反传不等于原生 GEN 训练已验证。
