# BAGEL：以 Memory 为轮间状态的 denoiser 内部循环

`main` 实现冻结原生 BAGEL 的 training-free 文生图。循环恢复旧 Memory loop 语义：只回传 Memory hidden，GEN 每轮恢复同一个 body 入口。当前代码没有 adapter、gate、α、shuffle 或训练入口。分层 Memory KV 是未实现的候选方案。

## 计算路径

同一次 denoiser 调用保持 `x_t,t` 不变。query 为每个样本的 `[SOI, M×K, GEN, EOI]`。Memory 使用原生 UND 专家，图像 token 使用原生 GEN 专家。共享注意力允许两者交互。Memory 初值为该样本 SOI/EOI embedding 均值加 `1e-4` 随机 slot noise；保留 GEN 和边界原始 RoPE，M 使用 SOI 的 RoPE。

1. prefix 执行一次。Memory 可读原生 query 和 prompt KV；所有非 Memory query 都禁止读 M，避免经边界 token 间接读取。
2. 保存 body 入口。第一轮执行完整原生 decoder body，保持上述严格 Read mask。
3. 每个额外轮恢复入口 GEN/边界，只替换为上一轮 body 末端的 Memory hidden。GEN 和 M 通过原生注意力交互；每层正常执行 attention、MLP 和完整残差更新。
4. 最后一轮执行一次 suffix、原生 norm 和 velocity head。suffix 保留旧路径的 Memory 读取语义。

`R` 是额外 body 轮数。`R=1` 表示一次 Read 加一次 Write；`R=0` 完全绕过 Memory，等于原生生成。默认窗口 `[0,8)`、K=8、在全部采样步启用；可显式缩小进度区间。Memory 不跨 denoiser 调用、样本或 CFG 分支保存。conditional 和 text-removed 分支各自创建并更新 Memory。packed 输入支持不同图像 token 数。

运行模式为 `BASE`、`MEMORY_LOOP` 和可选的 `MEMORY_NO_READ`。默认仅比较前两组。no-read 是实现诊断：所有阶段关闭 M→非M 读取，并调用原生 GEN 计算以消除 masked kernel 数值差异，其成本不能作为部署预算。原逐层 `1/N` 循环和 `GEN_LAYERWISE`、`MEMORY_STATIC` 已删除。

## 验证与执行

正式生成、评分、QA 和预算测量由用户启动。命令和远端路径见 [运行说明](docs/MEMORY_LOOP_RUNBOOK.md)。八卡入口也支持用户设置四张空卡；实际 worker 数等于 `GPUS` 中的卡数。默认执行 Base / Memory loop 的配对生成、语义和质量评分及合并，不默认执行问答 probe。

测试包含冻结旧实现与新路径在同权重、同输入上的 hidden/velocity 数值一致性。测试 oracle 只在 `tests/oracles` 和用户执行的一致性检查中使用，生成路径不加载旧训练代码。CPU 小模型测试通过不代表真实模型质量改善。

原生 BAGEL 来源见 [NATIVE_SOURCE.json](docs/NATIVE_SOURCE.json)。旧 Memory 来源及差异见 [LEGACY_MEMORY_SOURCE.json](docs/LEGACY_MEMORY_SOURCE.json)。decoder vendor 文件未增加架构改动；严格 Read 使用旧路径的 masked SDPA，其他层使用原生 FlashAttention。

方案见 [DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)。历史实验保存在 `docs/experiments`；已经出现的逐层 `1/N` 质量崩坏属于被替换的实现，不代表本次恢复路径的结果。当前真实权重的数值一致性和图像效果待用户运行。
