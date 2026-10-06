# 全量替换的代码检查

2026-10-06，main。远端 CPU 检查目录为 `/private/yida_workspace/bagel-LatentCoT-main-full-memory-20261006`。提交和源码 hash 记录于 `SYNC_COMMIT`、`SYNC_SOURCE_SHA256`。

全量 CPU 测试：82 passed，6 skipped。6 项均为 CUDA 检查，其中 2 项专门检查新全量路径。CLI help、Python compile、shell syntax、git diff --check 通过。

CPU 小模型证据：

- 全量静态 R1/R2/R3 输出与原生 decoder 完全相等。memory_slots=0 或 1 均不限制全量容量。
- 完整 token 数、packed sample split、原始顺序与 RoPE 位置不变，特殊 token 没有被删除。
- 全量动态 R1/R2/R3 保持与 P 相同的读取长度；内容 token KV 改变，特殊 token K/V 精确保持原生值。
- 仅含特殊 token 的样本保留原生输出；其他样本不会影响它。null CFG、R0、状态清理与 seed policy 检查正常。
- 模型权重、原始 prompt cache 的对象与数值不变。
- 七路深度报告包含动态 vs 同深度静态、动态 vs 较浅动态、静态 vs 较浅静态与各路 vs Base。
- 静态 PNG 校验能识别图片 hash 不一致和 noise seed 输入不一致，不会将不匹配数据报告为相等。

没有加载真实 BAGEL 权重，没有执行 GPU 生图、真实权重 E0、评分或训练。用户运行的 driver 会先做真实权重静态速度 parity，再生成图片，并在评分前检查全部静态 PNG 是否与 Base 相同。CPU parity 不替代这些检查。

本次没有修改 BAGEL vendor 层、模型参数结构、采样器或噪声分布。新增 generator.prompt_lengths 仅保存日志所需的原生长度。旧子集/append/legacy 模式的已有测试通过。未来静态或动态图像结果不能由这份检查报告推断。
