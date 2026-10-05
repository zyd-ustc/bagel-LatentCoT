# Denoiser 内部 Memory：执行与证据清单

状态：2026-10-05，架构、E0 和工程筛查已完成。28 项测试通过。默认窗口 4 卡、备用窗口 2 卡生成，共 80 张工程图；另有 4 张诊断图。独立确认与训练未执行。未勾选项尚未完成。主方案见 DENOISER_INTERNAL_MEMORY_PLAN.md；参数合同见 DENOISER_INTERNAL_MEMORY_PROTOCOL.yaml。

## 已完成的设计工作

- [x] 明确最终任务为 T2I，内部循环期间固定 x_t/t。
- [x] 核对 BAGEL native KV、专家路由、noise mask 与 flow 目标。
- [x] 核对两篇循环论文的方法及适用边界。
- [x] 核对 SLVR 的语义监督、实际训练开关与教师特征注入。
- [x] 核对 UNO 的训练方向、prompt 泄漏及论文局限。
- [x] 定义新的 N/R 语义、Memory 生命周期、初始化和位置规则。
- [x] 定义基线、证据层次、预算、训练准入与停止条件。

## W0：冻结运行合同

- [x] 工程运行已绑定原生 checkpoint、源码、环境与 scorer hashes；确认阶段仍需冻结合同。
- [ ] 绑定 development/confirmation manifests，排除历史与训练重叠。
- [ ] 确认 structural、ordinary/easy 实际数量，保留原题与 provenance。
- [ ] 绑定 native timestep shift、condition dropout 和其他 CFG 参数。
- [ ] 冻结 4 配置开发搜索上限、确认阈值与一次扩展规则。
- [ ] 记录协议版本；结果返回后不改成功门槛。

## W1：实现最小算子并通过 E0

- [x] 新增独立 mode，不沿用旧 Anchored α/gate/adapter。
- [x] prefill 捕获选中层 prompt 入口 hidden；按规则选取 Memory seed。
- [x] 正确实现按层临时 overlay 和一轮写读延迟。
- [x] N=1 真正调用 native bypass。
- [x] Memory 不跨层/跨 timestep 携带，不重复计入 K/V。
- [x] GEN RoPE 与原生一致；Memory 保留 source positions。
- [x] packed 变长图像、边界 token 专家路由、CFG 分支隔离正确。
- [x] no-read 与 GEN-only 对齐；P cache 不被写入污染。
- [x] 小规模高精度 mask 验证与真实 BAGEL BF16 parity 通过。
- [x] NaN/Inf、slot collapse 与无效缓存条目能够被检测。

## W2–W4：免训练证据

- [x] E1：完成默认纯 GEN 的 8 prompt 工程筛查，结果为负；完整开发集未执行。
- [x] E2：完成默认动态 Memory 的 8 prompt 工程筛查，结果为负；未达到训练准入。
- [x] 执行一次预定备用窗口的 E1/E2 工程筛查；未见正向净收益，停止扩大配置搜索。
- [x] E3：完成两窗口的 static-M 和 no-read 工程对照。
- [ ] E4：实际状态内容诊断；探针先做可靠性校准。
- [ ] 区分 prompt 意图、实际观察、unknown；检查间接复制。
- [ ] 开发集选定唯一 Memory 候选，冻结全部配置。
- [ ] 校准等时延原生采样 arm，延迟误差≤5%。
- [ ] E5：独立确认的全部 5 arms 配对齐全。
- [x] 工程结果已输出 prompt-cluster CI、Repair/Damage、语义、quality proxy 与 invalid。
- [ ] 固定抽样的 128 对盲审记录齐全；否则仅声称自动评测支持。
- [x] 默认窗口关闭诊断，完成 3 warmup + 20 measured generations 的完整延迟/显存测量。
- [ ] 若证据不足，仅进行一次预定扩展；到期停止。
- [x] 记录当前 stop/no-training；负结果和原始分数保留。

## W5–W6：仅准入后执行的理解侧训练

- [ ] 训练准入的每项证据具有文件路径与数值，不以 loss 替代。
- [ ] 复制 native UND writer 参数；原始 GEN/prompt/boundary 固定。
- [ ] 参数副本的显存和部署成本仍满足门槛。
- [ ] 每项 trainable 参数有 allowlist；验证 GEN 到 writer 的梯度。
- [ ] teacher 固定，labels/features 不出现在部署分支的 forward 输入。
- [ ] 对多深度分支完成合法全网络轨迹；不伪造中间出口。
- [ ] 训练只覆盖已验证的 N；N=1 不计成 writer 训练样本。
- [ ] teacher 标签分别保存 desired、observed、confidence 与来源。
- [ ] 语义监督读取屏蔽 prompt 直连，并检查 Memory 转存捷径。
- [ ] 视觉监督只有在位置与维度对应成立后才启用。
- [ ] 运行最多 200-step pilot，记录内容指标与最终图像指标。
- [ ] pilot 通过后运行最多 1000 steps；不自动追加训练。
- [ ] 对比免训练、depth-only、depth+semantic 三个必要结果。
- [ ] 用全新独立确认集评估，验证质量保留与成本。
- [ ] 对固定参数做 hash 对照；训练后 R0 保持原生。

## 关闭阶段时必须交付

- [x] 工程 manifest、run.json、初始噪声 hashes 与全部图像已保存。
- [x] 工程原始评分及逐项 Repair/Damage 已保存。
- [ ] prompt-cluster CI 与普通/easy 的非劣界限。
- [ ] native/GEN/static/dynamic/等时延 Base 的同口径比较。
- [ ] profiling 报告及真实 8 卡执行记录。
- [x] 接口正确性已证实；工程数据不支持正向收益；实际状态语义内容与泛化仍未解决。
- [x] 当前建议：不部署、不启动训练；保留本轮负结果和算子实现。

H200 运行记录与证据文件见 `docs/experiments/internal_loop_20261005/REPORT.md`。勾选代表该具体工作已完成，不代表语义收益通过。
