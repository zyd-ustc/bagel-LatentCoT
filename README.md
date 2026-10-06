# BAGEL：分层 Memory KV 的 denoiser 内部循环

`main` 实现冻结原生 BAGEL 的 training-free 文生图。`LAYERWISE_MEMORY_KV` 只将上一轮的原生 UND 层输入 KV 传给下一轮的对应层。GEN 与 UND writer 每轮恢复各自的固定入口。没有增加模型参数，没有 adapter、gate、α、shuffle 或训练入口。

## 计算路径

同一次 denoiser 调用内，`x_t,t` 保持不变。prefix 一次，共享 body 在原生层序上执行，最终 suffix、norm、velocity head 各一次。GEN 不把窗口末端 hidden 回送入口。

Memory slot 取自原生 prompt 的不同内容 token，保留它们在 body 入口的 hidden 和原 RoPE 位置。UND writer 读取原 prompt KV、当前 GEN 的原生层输入 KV，以及上一轮的同层 Memory KV。在正常层序上计算时，保存原生 attention 实际写出的层输入 KV。writer 的最终 hidden 不回传。

第一层 Memory 输入尚未观察 GEN，因此反馈读取从窗口第二层开始。默认窗口 `[0,8)`，有效读取层为 1–7，K≤8，R=1，全部采样步启用。R 表示额外 GEN body 轮数；R=0/K=0 完全绕过循环。Memory 不跨采样步、样本或 CFG 分支传递。无 prompt 的 text-removed CFG 分支走原生计算。packed 输入支持不同图像 token 数。

`MEMORY_LOOP` 保留已有的 hidden recurrence，作为明确的旧对照：边界 embedding + slot noise、body 末端 M hidden 回传、suffix 读取 M、null CFG 自建 M。两条路径的差异不应归因于 KV 格式这一项。

`LAYERWISE_MEMORY_REPLACE` 保留第 0 轮原生 body 启动，随后 GEN body 和最终 suffix 只读取同层 Memory，移除完整 prompt 的直接读取。UND writer 仍读取完整 prompt；最后一次 writer 走过 suffix，提供对应层的 KV。第一层 Memory 是固定 prompt seed，其他层可随 GEN 和前轮 Memory 改变。`LAYERWISE_SEED_REPLACE` 使用相同位置和长度的静态 prompt 子集，检查反馈与条件压缩的区别。细节和小规模八卡命令见 [prompt 替换方案](docs/MEMORY_PROMPT_REPLACEMENT.md)。

## 运行与证据

正式生成、评分、Memory QA 和真实权重检查由用户启动。见 [新路径运行说明](docs/LAYERWISE_MEMORY_RUNBOOK.md)。八卡入口也支持通过 `GPUS` 选择四张卡。默认比较 `BASE / MEMORY_LOOP / LAYERWISE_MEMORY_KV`，使用同 prompt、同 noise seed。

CPU 小模型测试验证同层 KV、固定入口、原生绕过、缓存不变、样本隔离与旧实现数值 parity。这些检查不能证明真实图像语义改善或质量保持。append 路径的 hard16 / seed0 / R1–3 已完成配对评测，均未显示语义净收益；prompt 替换路径尚未做真实权重生图评测。原生 UND 读取 noisy GEN 的语义能力仍是待检验的假设。

实现依据与边界见 [当前方案](docs/DENOISER_INTERNAL_MEMORY_PLAN.md) 和 [来源记录](docs/LAYERWISE_MEMORY_SOURCE.json)。借鉴 Looped-DiT 官方代码的循环控制流程；它的训练结果不能作为 BAGEL training-free 有效的证据。BAGEL vendor 文件没有新增架构改动。旧路径运行说明保留在 [MEMORY_LOOP_RUNBOOK.md](docs/MEMORY_LOOP_RUNBOOK.md)。
