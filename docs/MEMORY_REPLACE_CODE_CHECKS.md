# Prompt 替换路径的代码检查

2026-10-06，main。代码快照：`/private/yida_workspace/bagel-LatentCoT-main-memory-replace-20261006`。提交与源码 hash 记录在远端 `SYNC_COMMIT`、`SYNC_SOURCE_SHA256`。

执行的是远端 CPU 小模型测试，CUDA_VISIBLE_DEVICES 为空。没有加载 BAGEL 实际权重，没有执行正式生图、评分、真实权重 E0、训练或 QA probe。

全量结果：70 passed，4 skipped。跳过的是既有 CUDA 检查；替换路径的真实 CUDA 行为仍待用户验证。CLI help、Python compile、shell syntax、git diff --check 通过。

检查覆盖：

- GEN 第 0 轮读取原 P；R1/R2 的额外 body 和最终 suffix 只读取 K 个 Memory KV，没有混入 P，也没有逐轮积累 KV。
- Memory KV 精确来自同层 writer 输入；suffix writer 读取完整 P，GEN suffix 只读对应 M。
- 静态替换不执行 writer；GEN 使用对应原生 prompt 子集，位置和长度与动态替换相同。
- 原 P 对象与数值、所有模型参数不变；R0 原生绕过输出相等。
- packed 不等长样本互不影响；无 content seed 的样本和 null CFG 保持原生输出；状态不跨 denoiser call 累积。
- 在 s=0 和 s=1，改变 GEN 输入会改变后续 body 与 suffix 的 writer KV，第一层 seed KV 保持固定。
- CPU 小模型的 writer seed 经动态 Memory KV 到冻结 GEN 输出有有限非零梯度。它不证明训练 API 或 CUDA backward 已可用。
- 报告支持动态替换与同深度静态替换、同深度 append 的配对比较。

没有据此宣称有效语义编辑、画质保持或开销优势。这些结论依赖正式图像结果。
