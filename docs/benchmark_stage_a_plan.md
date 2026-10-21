# Stage A benchmark 接入计划

## 路径与验收

复用 modelarts-job 已部署的 GenEval 原始 553 条标注、Mask2Former、OpenCLIP 和 `/root/venvs/geneval`。
保留官方 prompt、标签、原始顺序和评分规则。采样只调用新 Stage A `InferenceEngine`，不恢复旧 loop。
原生 BAGEL 与训练后 Loop 使用相同种子、分辨率、CFG；每个 Loop 时间点仍使用全部 28 层、前 25 次 Euler 更新。

先验证六类各第一条、seed 0，共 12 张图及其官方评分。小样本不得报作正式 GenEval 总分。
全量入口默认 553 条 × seeds 0/1/2/3 × 两组，导出官方目录布局并报告完整性。

关键风险：现有 CPU 评分环境可能漂移；运行真实 pilot 验证。
CPU 适配沿用已部署逻辑：允许无 CUDA，CLIP 使用 FP32、不使用 CPU BF16 autocast；阈值不改。
不改 64 卡训练，不使用付费 API，不改共享评测环境，不改冻结的其他 benchmark 任务。

工具记录：baseline skill 的 bash_exec / artifact 工具在本会话未提供；使用当前可用执行工具，保留命令、输出和验收文件。
