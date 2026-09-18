# Within-step repeated-block probe (BAGEL) — 结论

## 实验设置

- 探针：`scripts/evaluate/within_step_loop_probe.py`
- 模型：frozen `Bagel-7B-MoT`（NPU, Ascend 910），零样本、无任何训练
- 结构：把 `layers[10, 18)` 作为共享 block，在**单个去噪步内**重复 L 次，
  `h ← h + α·(body(h) − h)`
- 对照：同一 prompt、同一初始噪声；`L=1` 为参考
- 配置：`L ∈ {1,2,4,8}` × `α ∈ {1.0, 0.5}`，prompt 取自 GenEval2-hard

## 结果（4 prompts 均值，均以同 prompt 同噪声的 L=1 为参考）

| 配置 | rel-L2 vs L=1 | pixel MAE vs L=1 |
|---|---|---|
| L=1, α=1.0 | 0.000 | 0.00 |
| L=1, α=0.5 | 0.000 | 0.00（等价，无中间残差） |
| **L=2, α=1.0** | **0.609** | **34.9** |
| L=2, α=0.5 | 0.466 | 23.5 |
| L=4, α=1.0 | 0.736 | 42.7 |
| L=4, α=0.5 | 0.616 | 35.6 |
| L=8, α=1.0 | 0.849 | 47.1 |
| L=8, α=0.5 | 0.759 | 45.1 |

逐 prompt 明细见 NPU `/home/ma-user/work/outputs/within_step_loop_probe_v1/`。

## 结论

1. **Parity 成立**：`L=1` 与原生前向逐位相同（探针内含 assertion，4/4 通过）。
2. **`L=2, α=1.0` 即明显崩坏**（rel-L2 0.61 / MAE 34.9），随 L 单调加重（L=8 → 0.85/47.1）。
3. **阻尼 α=0.5 在全部 L 上一致优于 α=1.0**（L=2: 0.466 vs 0.609；L=4: 0.616 vs 0.736；
   L=8: 0.759 vs 0.849）。说明退化确有"残差无节制累积"的成分，但**阻尼不能挽救 naive loop**。
4. 与 SenseTime Looped MMDiT 博客一致：naive loop 不是免费的模型深度；其归因
   （反复更新侵蚀 token 局部位置信息，位置预测 R² 下降）与我们观察的"数量/空间先坏"吻合。
5. 推论：**loop 不能作为零样本 inference 技巧**，必须配合训练侧设计
   （Deep Supervision + Loop Distillation + Self-Modulating Attention）。

## 待补

- 人工/VQA 层面确认退化是否集中在"数量与空间关系"（与位置侵蚀假设对齐）。
- 输出目录：`/home/ma-user/work/outputs/within_step_loop_probe_v1/`
