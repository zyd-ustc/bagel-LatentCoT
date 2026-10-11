# 验证

状态：工程记录已归档；正式评测未开始。更新：2026-10-11。

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

## 3. 未验证的结论

未运行正式BAGEL权重的目标NPU/CUDA训练或质量评测。Caption接地、GEN语义使用、画质收益、真实吞吐和显存尚无结果。新增语义读出、响应蒸馏与语义去噪也尚未实现。

正式质量解释前先做16样本真实权重检查：冻结/梯度、原生text与GEN来源、mask/RoPE、目标后端重复误差、CFG与采样网格。通过后按[实验](experiments.md)分项评测；工程通过不等于效果提升。
