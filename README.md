# BAGEL: denoiser 内部循环

`main` 仅保留冻结的原生 BAGEL、层内 GEN/Memory 循环、免训练生成和离线评测。旧 reader、OPD/GRPO、Anchored adapter 和训练入口已删除。历史结果留在 `docs/experiments`，不代表新架构的结果。

## 架构

固定一个 denoiser 输入 `x_t,t`。在 `[0,8)` 的每个 decoder layer 内执行 `N` 次 GEN 运算，残差步长为 `1/N`。`N=1` 直接调用原生实现。采样进度 `s=i/(S-1)` 从噪声走向图像；默认仅在 `s∈[0,0.5]` 启用循环。

Memory 从该层原生 prompt 输入状态中选取至多 16 个不同的内容 token。它保留源 token 的 RoPE，用原生 UND 的 norm、QKV、O projection 和 MLP 更新。第一次 GEN 运算不读 Memory；随后 GEN 读取更新后的 Memory KV。GEN 和 writer 在同一轮均读取更新前的状态。Memory 不跨 layer 或 timestep 传递；最后一次 writer 的 Q/attention/MLP 不执行。

prompt KV 只读。Memory KV 是临时 overlay，物理追加位置不会改变 RoPE。SOI/EOI 使用原生 UND expert，图像 token 使用 GEN expert。CFG 分支使用各自的 prompt/Memory。text-removed 分支没有 Memory，但保留同一 GEN 循环策略。packed 输入支持不同图像 token 数。每次 denoiser 只执行一次最终 suffix/readout。

五组免训练对照为 `BASE`、`GEN_LAYERWISE`、`MEMORY_DYNAMIC`、`MEMORY_STATIC`、`MEMORY_NO_READ`。静态 Memory 仍计算 writer，但丢弃更新，以匹配 writer 开销。no-read 必须与 GEN-only 数值一致。

原生来源和逐文件 SHA256 见 [docs/NATIVE_SOURCE.json](docs/NATIVE_SOURCE.json)。decoder vendor 只改包名和 attention dispatch；原生 ImageTransform 只移除未使用的 cv2 导入。CUDA 必须使用 FlashAttention，CPU 慢速实现仅用于测试。原始代码各文件的许可证声明保留。

## 在 H200 执行

本轮把窗口提前到零起算的 `[0,8)`。用户已确认早期 loop 有效；旧 `[16,24)`、`[12,20)` 的小规模负结果仅约束那些配置。固定 384 prompt×3 seed 确认集已取消，既不是待办，也不是入口要求。

正式生成、评分、问答及预算评测均由用户启动。八卡完整命令、续跑方法和 probe 判读见 [EARLY_MEMORY_PROBE_RUNBOOK.md](docs/EARLY_MEMORY_PROBE_RUNBOOK.md)。八卡各一个进程，按 prompt/seed 分片；五组 arm 使用相同初始噪声。生成模型退出后才加载评分或问答模型。

语义问答 probe 只使用原生 UND 层、原生 LM head 和各层 Memory KV。它不读取原 prompt KV，也不加入 adapter。`DYNAMIC`、`SEED`、`EMPTY` 对照共享题目和位置规则；`VIT_IMAGE` 是离线读出校准。probe 使用第 8、16、24 步的 guided x0 图像估计，离线标签区分实际内容、prompt 要求及 unknown。问答准确率不能替代最终图像质量。

报告包含 GenEval2 soft-TIFA/GM、Repair/Damage、quality proxy、invalid、按 prompt 聚类的置信区间。质量 proxy 不能替代人工盲评。probe 捕获和中间 VAE 解码增加诊断成本；这类日志不能用于部署预算声明。独立预算入口为 `benchmark_budget.py`，须关闭 probe，执行 3 次 warmup 和至少 20 次测量。

环境使用已存在的 BAGEL Python、Qwen3-VL scorer Python 和 torchvision 0.20.1。只加载原生 BAGEL 权重；当前没有训练入口。`BASE_MATCHED_LATENCY` 仍须先标定原生采样步数。

## 测试与方案

```bash
python -m pytest tests -q
```

旧窗口的 H200 工程结果见 [docs/experiments/internal_loop_20261005/REPORT.md](docs/experiments/internal_loop_20261005/REPORT.md)：该报告保留旧窗口的负结果，不作为否定早期 loop 的依据。本轮新增 probe 的正式结果待用户运行。

完整研究方案见 [docs/DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)，协议见 [docs/DENOISER_INTERNAL_MEMORY_PROTOCOL.yaml](docs/DENOISER_INTERNAL_MEMORY_PROTOCOL.yaml)。删除清单见 [docs/REFACTOR_REMOVALS.json](docs/REFACTOR_REMOVALS.json)。
