# 验证

状态：真实权重C0已完成；C1训练与生成质量评测未开始。更新：2026-10-12。

[返回 README](../README.md)

## 1. 已有证据

|记录|实际范围|
|---|---|
|[首次构建](archive/build.md)|本地macOS arm64；23项测试、wheel构建与安装后CLI检查|
|[审查修正](archive/review.md)|修正后40项测试；0.6.1 wheel与源码一致；梯度故障注入和原生时间网格检查|
|[历史清理](archive/cleanup.json)|旧模块/脚本/配置清单与备份位置；不是当前运行说明|

测试使用随机初始化的小型原生Qwen2MoT/BAGEL模块。流程中的VAE/tokenizer是替身；另有真实Qwen2Tokenizer类配临时词表的检查，仍未使用正式tokenizer文件。

这些是已有工程记录，本轮文档整理没有重新运行模型测试。日志和wheel在外层工作区`artifacts/memory_topology_build_20261011/`与`artifacts/memory_topology_review_20261011/`，不保证随仓库分发。

## 2. 已记录的修正

- 检查每层Memory K/V梯度与C跨冻结UND的梯度。
- 记录两个swap的等价关系，避免重复算证据。
- C0使用独立复述指令及原生tokenizer/BOS/EOS协议。
- 后续采样绑定原生actual t及完整时间网格。
- t=1同时检查跨x0输入与自由输出；修正局部否定事实规则。
- 使用线性残差；区分零可见G-fixed与中性G-neutral。
- 独立抽t/扩散seed，VAE posterior由图像内容hash派生。

## 3. C0真实权重结果

2026-10-12，`modelarts-job`，`loop-layer-NPU`提交`152e13a`。模型为 `/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT`；输入为 `data/c0_text16.jsonl`；输出为 `/root/outputs/c0_text16_20261012_111241`。16卡各处理一条人工编写文本，任务是原生caption KV的自由复述，不涉及图像或压缩Memory。

|指标|结果|
|---|---|
|完成样本|16/16|
|平均教师强制NLL|0.2711153743|
|严格文本匹配|7/16（43.75%）|
|逐条预设字面事实分|15条为1；`c0_08`为0|
|16卡端到端耗时|54.0313秒，含启动/加载/汇总；不是训练吞吐|

人工检查：主要对象、颜色、数量与关系保留。`c0_08`输出“single orange”，与参考“one orange”等价；字面词表漏掉同义词。`c0_15`保留五个紫色积木，但新增尺寸、地板材质与光照等无依据描述；事实分1不能排除扩写。其余多为同义改写或引导语。

结论：原生文本语义读取可用，允许进入C1小预算；严格复述未通过，存在无依据扩写。此结论不证明Memory压缩、视觉接地或GEN消费成立。校准文件初始`passed: false`是待人工核验状态，不是自动失败；放行时需保留上述范围和限制。

## 4. 未验证的结论

真实BAGEL的C1/C2训练、Caption接地、GEN语义使用、画质收益、训练吞吐和显存仍无结果。新增语义读出、响应蒸馏与语义去噪尚未实现。按[实验](experiments.md)推进文本桥与GEN读取，不因C0通过而扩大结论。

## 5. 联合训练入口检查

2026-10-12，本地41项测试通过（9.30秒）。新增双进程CPU测试验证：参数广播、梯度累积与全局batch缩放、同步梯度裁剪、连续三步AdamW更新，均与单进程全局batch参照一致。

`modelarts-job`的16卡HCCL检查通过：参数广播、累积梯度SUM、连续三步SGD更新、统计归约与记录汇总。使用一个单参数线性模型，不加载BAGEL；日志为 `/tmp/bagel_hccl_check.log`。这仅验证联合训练的通信与优化数学，不代表真实BAGEL训练已通过。
