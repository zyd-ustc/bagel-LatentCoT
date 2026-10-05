# main 破坏性重构：工程验证记录

执行日期：2026-10-05。主机：`node-12` / H200。工作树：`bagel-LatentCoT-phase1a-opd`，分支 `main`，重构基点 `02edfb1`。远端代码：`/private/yida_workspace/bagel-LatentCoT-main-internal-20261005`。

## 结论

新架构与评测入口已实现，原生 bypass 与 no-read parity 通过。默认与一次预定备用窗口的工程结果均未支持正向语义收益。目前不能启动训练，也不能宣称动态 Memory 提供有效语义增量。质量 proxy 未下降不等于语义改善。

## 实现与删除

删除或替换旧 reader、OPD/GRPO、训练入口、旧配置、旧评测脚本和对应测试，共处理 256 个旧路径，最终净删除 231 个旧路径。原生模型代码重新取自 `refs/Bagel`，仅改命名空间和 attention dispatch。逐文件原始 hash 存于 `docs/NATIVE_SOURCE.json`；CUDA 使用原生 FlashAttention。保留历史结果记录，不将其作为新架构证据。

实现逐层 GEN 循环、临时 Memory KV overlay、原生 UND writer、原生 MoT expert 路由、延迟读取、同轮更新前状态读写、层间和 timestep 间重置，以及最后一次 writer 查询跳过。无新增训练参数。CFG 保持各自状态；空文本分支无 Memory。支持 packed 变长输入。

新增工程/确认入口、独立 GPU 分片与严格 resume 校验、权重/源码/配置/噪声/图像 hash、官方 GenEval2 函数绑定、TIIF 本地 yes/no 评测、quality proxy、Repair/Damage、prompt cluster bootstrap、盲评导出/导入及预算测量。scorer 失败保持未完成状态，不当作零分。非有限生成图计为 invalid。

## 数值与接口

H200 同原始权重、同输入，`N=1` vs 原生 velocity 最大绝对差 **0**；GEN-only vs Memory no-read velocity 最大绝对差 **0**。prompt cache 不变。packed 输入包含 256×256 和 256×384。默认工程运行的 8 张 GEN-only 图与 no-read 图全部 SHA256 一致。

28 项测试通过（含 CUDA/FlashAttention）。测试覆盖 CPU attention 与 float64 mask 参考、CUDA FlashAttention、GQA、变长输入、混合空 Memory 段、CFG cache 隔离、expert 路由、无新增参数、内容 token 选择、原始 RoPE、延迟读取、writer 跳过、诊断不改变结果、slot 塌缩检测、prompt cluster CI、数据配对与盲评 hash 校验。

## 工程图像与评分

固定 4 条历史 hard16 和 4 条 easy16 prompt，seed=0；每窗口五组，共 80 张 512×512 图。`N=2,K≤16`，采样进度 `[0,0.5]`；50 个时间点 / 49 次 native denoiser，shift=3，text CFG=4。所有 arm 同噪声。该数据仅是工程集，不能当独立确认集。scorer 使用本地 Qwen3-VL-8B-Instruct，官方 GenEval2 的 `soft_tifa` 函数；完整 source/模型 hash 见 provenance JSON。每窗口 61 个配对 atom。CI 为 prompt 聚类的 10,000 次重采样。

|窗口|Arm|Semantic GM|Quality proxy|Invalid|净 Repair vs Base|Repair / Damage|
|---|---|---:|---:|---:|---:|---:|
|共同 Base|BASE|0.4240|0.8125|0|0|0 / 0|
|[16,24)|GEN_LAYERWISE|0.2588|0.8438|0|-0.1148|2 / 9|
|[16,24)|MEMORY_DYNAMIC|0.2679|0.8438|0|-0.1148|2 / 9|
|[16,24)|MEMORY_STATIC|0.2828|0.8125|0|-0.0820|3 / 8|
|[12,20)|GEN_LAYERWISE|0.2584|0.8750|0|-0.0328|5 / 7|
|[12,20)|MEMORY_DYNAMIC|0.2902|0.8438|0|-0.0328|6 / 8|
|[12,20)|MEMORY_STATIC|0.4128|0.9062|0|-0.0328|3 / 5|

默认动态 Memory 净 Repair CI：`[-0.2029,-0.0333]`。备用动态 Memory CI：`[-0.1500,0.0909]`。备用窗口没有明确正向证据；其动态 Memory GM 低于静态 Memory。no-read 数值与 GEN-only 相同，评分也相同。

## 开销

默认窗口，batch=1，3 次 warmup 后每 arm 测量 20 次。包含 prompt prefill、完整采样和最终 VAE decode；不含模型加载、写盘和 scorer。关闭诊断。

|Arm|中位秒/图|p95 秒/图|中位耗时比|峰值分配 GiB|
|---|---:|---:|---:|---:|
|BASE|6.292|6.312|1.000|28.527|
|GEN_LAYERWISE|7.127|7.155|1.133|28.527|
|MEMORY_DYNAMIC|7.558|7.632|1.201|28.528|

本轮默认配置通过中位≤1.35×、p95≤1.5×、峰值分配≤1.2× 的开销门槛。这里只说明本机工程测量；确认集不同长宽比和分辨率需要重新测量。

## 同 prompt 两个 seed 的 Memory 诊断

另用同一 hard prompt、seeds 0/1 运行可选诊断，保存每个 seed 的 400 条 conditional 记录。平均 effective rank 约 6.37，slot cosine 约 0.879，update ratio 约 0.084。第二次 GEN 运算中，采样 query 对 Memory 的平均 attention mass 分别约 **0.161% / 0.134%**。第一次读取为 0，符合 mask 契约。低读入比例是可观测事实，不能单独证明语义内容无效。

seed=0 的 Base 和动态 Memory 最终图像均与前一版 runtime 的 SHA256 一致；可选诊断不改变图像。两种 seed 的状态不同，只能说明状态敏感性，尚不能证明 Memory 能正确判断实际计数或关系。完整摘要见 `diagnostics_summary.json`。

## 证据范围与待办

未执行独立 384 prompt×3 seed 确认集，也未运行等时延 Base。工程质量分数是同一个 VLM 的固定质量协议，不是人工质量偏好。匿名人工复核包已生成，尚未填写。E4 原生 Memory 语义问答 probe 尚未实现；slot/read-mass 诊断不能替代它。训练入口按任务范围删除，未重建或启动。

默认结果位于 `/private/yida_workspace/outputs/internal_loop_engineering_20261005`，备用结果位于 `/private/yida_workspace/outputs/internal_loop_fallback_20261005`。工程运行时的完整代码快照保存于默认输出目录的 `source.tar.gz`；对应 hash 存在各 worker 的 `run.json`。后续代码新增诊断和盲评字段，保留旧快照以精确复现这些结果。源代码版本不同时，resume 会拒绝复用旧输出目录。
