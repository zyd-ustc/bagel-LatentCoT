# Denoiser 内部 Memory：当前工作清单

状态：2026-10-05。原生循环已实现。用户确认早期 loop 有效；本轮将默认窗口移到 `[0,8)`。固定 384 prompt×3 seed 确认集已取消。旧窗口结果保留在 experiments，不作为否定早期 loop 的依据。

- [x] 移动窗口到 `[0,8)`，保留 N=2、K≤16、前半程循环。
- [x] 删除固定确认集入口要求与对应待办。
- [x] 导出 GEN 实际读取的原生 UND Memory KV，以及初始 Memory KV；不导出 prompt KV。
- [x] 原生完整 UND/LM head 问答，按层读取 KV；没有 adapter 或新训练参数。
- [x] DYNAMIC、SEED、EMPTY、离线 VIT_IMAGE 读出对照。
- [x] 离线标签分开保存实际内容、要求答案、unknown 与来源。
- [x] 报告 prompt/图像不一致时的复述率，以及同 prompt 不同 seed 的状态区分。
- [x] 生成、图像质量评分、Memory QA、图像标签均有八卡分片与严格合并入口。
- [ ] 用户运行正式评测，检查各 step/skill 的有效标签数与 VIT_IMAGE 读出可靠性。
- [ ] 用户核查最终图像的 semantic gain、quality retention、Repair/Damage。
- [ ] 依据已有早期证据和本轮 Memory 增量，决定是否需要理解侧格式/隐式适配。

代码检查和小模型接口测试仅证明实现合同。问答准确率不证明 T2I 质量提升；有序 Memory slot 或 attention 读取不证明语义内容。ViT 只离线运行，不能反馈到正式生成轨迹。

八卡命令与报告路径见 [EARLY_MEMORY_PROBE_RUNBOOK.md](EARLY_MEMORY_PROBE_RUNBOOK.md)。所有正式评测均由用户启动。代理不自行生成、评分、运行大模型问答或预算评测。
