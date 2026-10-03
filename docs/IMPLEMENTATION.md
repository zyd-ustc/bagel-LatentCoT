# 文档方案与实现对应表

方案来源：用户提供的 `UMM_T2ILoop_Anchored_Loop_Design.docx`。
实现对象是文档的第一版 Anchored GEN Loop + Memory Scratchpad。
本文区分代码实现、数值验收与需要真实模型实验的研究结论。

| 原文要求 | 实现 | 验收证据 |
| --- | --- | --- |
| 2.3、3.1 prompt KV 与 `x_t` 固定 | 所有 body/suffix 使用 `update_past_key_values=False`；sampler 在 inner loop 后只更新一次 | 真 MoT 小模型逐 tensor 比较 KV 与 `x_t` |
| 3.2 原生 base pass | prefix/base 使用原生序列，不插入 memory，也不经过 gate/adapter | off、R=0、α=0 的 velocity 与 native 一致 |
| 3.3 anchored re-entry | `RMSNorm → down → up`；up zero-init；可选 memory mean projection | correction 相对于同一 body exit `G_base`；固定映射测试捕获 entry |
| 3.4 GEN 为主循环状态 | 保存 `ΔG=G_r−G_base`，重新注入固定 `G₀` | GEN-only 无 dummy memory 完整运行 |
| 3.5 可选 memory workspace | extra body 每个 sample 追加 K 个 UND tokens；body exit memory 在本 timestep 内传递 | K=0/8/16 与真实 packed MoT 测试 |
| 3.6 loop-only write gate | 每层 sigmoid scalar gate，仅控制额外 body 的 residual update | base 不经过 gate；所有 gate=0 明确退回 native |
| 3.7 base residual merge | `G_base+α_r(G_r−G_base)`，per-loop α zero-init | α=0 exact parity；α 初始梯度非零 |
| 3.8 shared suffix | 每轮去掉 memory，以原生 endpoint boundary 和当前 GEN merge 执行共享 suffix | 同一 readout 供训练和推理使用 |
| 3.9 CFG 分支隔离 | 每次 branch 调用独立分配 entry/base/delta/memory；最后组合各轮 velocity | 三分支 workspace 存储独立，深度一致，α=0 CFG parity |
| 4.3 三模式 | `memory_only`、`gen_only`、`gen_memory_anchored` | 同一 runner，仅 persistent state 不同 |
| 4.4 负对照 | `direct_native` 直接以先前 exit 进入 body | 默认仍为 anchored；负对照单独选择 |
| 6.2、8.3 memory 干预 | correct、zero、frozen、batch-shuffled | zero/frozen 保持 body 中 memory 不变；shuffled 要求 batch≥2 |
| 6.3、6.4 functional diagnostics | velocity delta ratio、相邻 correction cosine、逐层 GEN write ratio、state norm；early/middle/late bins | 数值记录不是语义质量指标 |
| 7.1 Stage 1 参数边界 | 所有原生参数冻结，只开放 `t2i_loop.*` | backward + optimizer 后 native 权重逐 tensor 不变 |
| 7.2 direct flow deep supervision | final MSE + λ×mean intermediate MSE，同一 `epsilon−x1` 目标 | 显式损失数值测试；无 final-loop distillation |
| 7.6、7.7 数据与 curriculum | 普通/结构/easy/noop prompt-image JSONL；在分配上限内抽取 R | 同 timestep、同 clean image 与噪声目标监督每轮 |
| 7.4、7.5 后续阶段 | workspace 专项训练、body LoRA 尚未实现或启动 | 需先取得文档 M1–M3 的真实语义/质量证据 |

## 执行语义

- `loop_depth` 为额外 body 次数 R；总 body 执行为 `1+R`。
- 每个 branch 的 prefix 执行一次。base readout 和每轮 readout 复用 suffix，共 `1+R` 次。
- 当前 boundary query 每层使用 native base pass 在该深度的状态。它不参与跨轮 recurrence。
- memory 只存在于 extra body；suffix 与 sampler 的原生 token 布局不变。
- 同一个 batch 必须使用相同数量的 GEN/image tokens。不同 batch 可使用不同图像尺寸。
- GEN+Memory 的训练 re-entry 最后一层为零；training-free topology 对照使用固定小比例 correction mapping。
- 初始 α=0 时训练仍运行 loop 以计算 α 梯度。无梯度推理可直接使用 native route。
- 新 checkpoint 不兼容旧 loop/LoRA checkpoint，必须从原生 BAGEL checkpoint 加新 loop 参数加载。

## 本次验证与实验边界

本地测试使用真实四层 BAGEL MoT decoder，hidden size 32，CPU bfloat16。
它们验证控制流、缓存、tensor 布局、关闭等价性、梯度与 checkpoint。
完整 7B checkpoint 的 semantic gain、quality retention、Repair/Damage 和 FLOPs 尚未验证。
文档的 M1–M5 Go/No-Go 门槛属于真实实验验收，不能用单元测试通过替代。

重构前的全部文件快照（含已有未提交修改）保存在仓库外：
`../backups/bagel-before-umm-anchored-20261004.tar.gz`。
同目录的 `.patch` 保存 tracked diff，`.base` 保存重构前 HEAD。
仓库内的历史研究文档和旧实验入口已删除；已有实验产物目录未删除。
