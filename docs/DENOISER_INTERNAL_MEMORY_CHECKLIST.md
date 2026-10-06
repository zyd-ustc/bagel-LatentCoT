# 当前检查状态：分层 Memory KV

日期：2026-10-06。

- [x] 实现原生 UND 层输入 KV，在下一轮对应原生层读取。
- [x] GEN/边界与 UND writer 每轮恢复固定入口，不回传窗口末端 hidden。
- [x] 原 prompt 内容 slot、不同原位置、短 prompt 不复制 slot。
- [x] 第一层不暴露静态 seed 为反馈；suffix/readout 一次且不加 Memory overlay。
- [x] prompt cache 只读；无文本 CFG 分支原生绕过；状态不跨调用和样本。
- [x] packed 不同长度、KV 来源、GEN 依赖、固定入口、no-read/native、旧路径 parity 合成检查。
- [x] 51 项 CPU 测试通过，4 项 CUDA 合同检查跳过。
- [x] 三组配对图像评分、真实权重 E0、分层 Memory QA 导出与用户执行命令。
- [ ] 用户执行新路径真实 BAGEL 权重的 E0 和 CUDA packing。
- [ ] 用户执行分层路径的配对生成、语义/质量评分与人工观察。
- [ ] 图像证据确认质量保持和有效语义编辑。

本次没有执行正式 GPU 生成、QA 或评分。此前已执行的旧 hidden loop 评测不作为新路径证据。训练不在本轮范围内，384×3 确认集和默认 shuffle 不属于待办。
