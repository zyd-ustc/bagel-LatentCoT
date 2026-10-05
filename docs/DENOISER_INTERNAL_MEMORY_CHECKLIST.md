# 当前检查状态

- [x] 删除逐层 `1/N` 运行路径，恢复只回传 M 的 body loop。
- [x] GEN/边界每轮恢复入口，完整原生层更新，prefix/suffix 各一次。
- [x] 边界均值加 `1e-4` slot noise；M 使用 UND。
- [x] prompt KV 只读，Memory 在每次 denoiser/CFG 调用内创建。
- [x] 合成 packed 输入、不同 token 数、同权重旧路径 parity、no-read/native parity。
- [x] 配对 training-free 生成、质量/语义评分、用户执行入口。
- [ ] 用户执行真实 BAGEL 权重的 E0 一致性检查。
- [ ] 用户执行恢复路径的生图及结果评分。
- [ ] 图像证据确认质量保持和有效语义编辑。

训练、分层 KV 反馈和独立 UND writer pass 均未实现。固定 384×3 确认集和默认 shuffle 不属于待办。
