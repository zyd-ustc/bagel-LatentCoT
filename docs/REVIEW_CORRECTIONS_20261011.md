# 2026-10-11 审查修正与验证

修正前 main 为 `a9cdb0a`。本次更新覆盖用户列出的四项关键问题和六项较小偏差。真实权重实验仍未执行。

## 逐项处理

|问题|处理与证据|
|---|---|
|Memory K 梯度|修正前源代码的 consume_bank 已没有 no_grad 装饰器。新增 L/C 的 K-only 与 V-only 反传测试；仅末层 bank 连通时，C 的首层 adapter 仍收到梯度。首个训练微批检查每层 K/V 梯度并记录。临时加入 no_grad 后，两个 K-only 测试失败，两个 V-only 测试通过；已恢复源码。|
|两个 swap 等价|保留协议中的名称，但共用一次 donor-bank 解码，记录 equivalent_to 与共同 evidence group。报告明确不算两份独立证据。Source-off 与 Memory-masked 仍分别回答来源依赖与 Memory 通路依赖。|
|C0 指令|单独固定复述指令，teacher forcing 和自由解码都使用该指令。指令 ID 复用原生 prepare_prompts；真实 Qwen2Tokenizer 类配临时 byte-BPE 词表覆盖中英文和 BOS/EOS。尚未验证正式 tokenizer 文件与真实模型复述质量。|
|续采样时间网格|受控续采样取最近原生节点，并按实际 t 重建 x_t 与 writer 来源。全部 arm 使用相同状态与完整后续节点；拒绝网格外起点、缺少节点或轨迹 sampler 配置不匹配。局部诊断不启用续采样时仍使用精确协议 t。|
|纯噪声检查|按 noise_seed 分组，跨所有受控图像同时检查 x_t_hash 与输出 ID，不依赖 swap_id，不按待验证 hash 分组。|
|否定词事实分|v2 按标点和转折词划分局部断言；a red cube, not blue 获得 red 的事实分。保留未知事实检查。这仍是受控词表规则，无法解析任意语言的否定范围。|
|残差激活|改为无 bias 的 Linear–Linear；up 保持零初始化；线性性质测试通过。|
|固定 bank|G-fixed 改为全零可见 K/V，仍参与 softmax；中性 seed K/V 单列 G-neutral。两份固定 bank 随 GEN 张量保存。|
|C2 抽样|每个微批均匀独立抽 t，扩散 seed 使用另一随机流；样本顺序使用第三个流。逐微批记录 t/seed。有限次训练不保证各桶精确等量。|
|VAE posterior|保持原生 posterior 采样；由图像文件内容 hash 与基准 seed 派生 seed。不同图像不再共用 posterior 噪声；相同文件内容仍可复现。posterior 与扩散 RNG 分离。|

## 实际续采样节点

原生设置：sampler_steps=50、timestep_shift=3；原生实现使用50个节点、49个 Euler 区间。

|请求 t|实际 t|step_index|首步 dt|剩余区间|
|---|---|---|---|---|
|0.05|0.058823530|48|0.058823530|1|
|0.25|0.254237264|44|0.043710947|5|
|0.5|0.493150681|37|0.028361976|12|
|0.75|0.757575750|24|0.015307724|25|
|0.9|0.902438998|12|0.009877086|37|

t≈0.05 的最近原生节点仍只剩一个完整区间。这是该原生网格的末端结构；本次修正没有凭空增加步数，也没有从0.05截取一个非原生区间。

## 验证范围

- main：40项测试通过。包含随机初始化的真实 Qwen2MoT/BAGEL 小模型、C1→C2训练与 GEN 续采样流程。
- 包版本0.6.1；wheel构建和安装通过；仓库外 CLI help 通过；wheel中的41个 Python 文件与源码逐字节一致。
- 测试日志、梯度故障注入日志、时间节点和wheel保存于外层工作区 artifacts/memory_topology_review_20261011/。
- 旧 checkpoint 缺少新实验配置字段时拒绝加载；旧 C0通过文件缺少复述指令/token IDs时不接受。旧结果须重跑，不能和修正后的结果直接混合。
- 未加载正式 BAGEL 权重，未运行目标 NPU/CUDA 实验。工程通过不能证明 caption 接地、GEN 的语义使用或画质提升。

分支同步后的核对结果另存外层 artifacts/memory_topology_review_20261011/branch_sync.json。用户原有未提交删除与图示资产不包含在本次提交中。
