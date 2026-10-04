# 本地重构验收

日期：2026-10-04。主机：本地 macOS，CPU。
分支：`codex/umm-t2i-anchored-loop`。

验证环境使用 Python 3.12.14、PyTorch 2.14.1、Transformers 4.57.6。
模型测试使用真实四层 BAGEL MoT decoder，hidden size 32，bfloat16；入口测试使用小型模拟 VAE 和 tokenizer。

| 检查 | 执行结果 | 证据范围 |
| --- | --- | --- |
| 完整测试集 | 96 passed | 包含模式、K=0/8/16、CFG、固定 KV/`x_t`、checkpoint、loss 与梯度 |
| 原生路径对照重构前快照 | passed，velocity 最大绝对误差 0.0 | 相同小模型权重与输入，velocity shape `[32,8]` |
| 新 workspace 关闭循环 / R=0 / α=0 | passed，velocity exact parity | 小型真实 MoT；全局、逐 channel 与 text-channel CFG |
| Current MemLoop vs legacy compatibility | passed，5 组 velocity 最大绝对误差 0.0 | 独立子进程加载冻结 parent `6f936b7`；K=8/16，R=1/2，batch=1/2，global/channel/text_channel CFG，含零 gate/α与训练参数污染检查 |
| Memory slot 对称性 | passed | 实际 BF16 初始化算术；K=8/16 唯一 slot，初始及循环后 variation 非零；collapsed slots 的 centered rank=0 |
| Stage 1 timestep 分布 | passed，逐值误差 0.0 | seed 固定，4096 个 normal draw → sigmoid → shift；shift=1/3，native training forward 接入检查 |
| GEN correction gate | passed | native transformation 为 reference；只 gate GEN；真实 MoT 的 memory full update 未被 gate |
| GEN-only 初始 parity 与训练梯度 | passed | α=0.01 + zero adapter 初始 velocity exact parity；bias 梯度非零；实际两步训练 bias 改变 |
| P1-1 shuffled causal control | passed | donor 固定；reader 的污染 M 被丢弃；3 轮 canonical M 与 correct 逐 tensor 相同；两层 MoT 的 R1 第二层 read delta>0 |
| P1-2 matrix legacy 参数隔离 | passed，velocity exact parity | 实际 CLI，α/gate=(0.1,0.02)、(0.9,0.8) 各对照 (0,0)；manifest 标记 parent readout |
| P1-3 Direct Native GEN-only / +M | passed | GEN-only K=0；完整 `B(G_previous)`，无 gate/α blend；实际 sampler 与 runner 一致 |
| P1-4 分阶段训练覆盖 | executed，10 optimizer steps passed | 新深度在 step 1/4/8 执行；R3 执行并更新共享 α；checkpoint 含 coverage；额外 R4 标记 unseen |
| P1-5 R0–4 elastic depth | passed | R3 checkpoint 在 runtime=4 加载并执行；R0 exact native parity；allocation=4；R5 被拒绝 |
| P1-6 native text dropout | passed，velocity 最大绝对误差 0.0 | full / mixed dropout 对应同一 text-removed layout；两步训练包含 dropped-condition 样本 |
| P1-7 variable-resolution packed batch | passed，velocity 最大绝对误差 0.0 | GEN token 数 16/8/12；批处理对逐样本运行；memory conditioning 映射与 backward |
| P1-8 suffix/readout 次数 | passed | R4 普通推理和关闭 DS 的 autograd 路径各 1 次；DS/诊断 5 次；最终 velocity exact parity |
| P1-9 bucket 配比与日志 | passed | 6000 batch，40/30/20/10% 误差≤2.5%；固定 seed 可复现；分 bucket loss/α/gate |
| P1-10 native 无裁剪 resize | passed | 横竖矩形均保留完整画面和角落像素；VAE 按样本编码后 packed |
| P1-11 memory 监控 | passed | signed cosine、effective rank、slot std、sigma1 ratio 和逐轮 update ratio；训练日志包含诊断 |
| P1-12 evaluator 的结果协议 | passed，13 项接口/计算测试 | 原题 GenEval2 CLI 接口 fixture、TIIF yes/no 解析、三数据集 CLI、Repair/Damage 分母、相邻 R、坏图、缺失分数、人工评审哈希；未执行真实 judge |
| checkpoint v3 / 拒绝 v1、v2 | passed | 共享 α 不按 R 分配；allocation/runtime 分离；拒绝 whole-layer gate 和 per-depth α |
| 新 workspace 所有 loop gate=0 | passed | 即使输出 α 非零也返回 native readout |
| 最终图片等价性 | passed，逐字节相同 | 小模型 sampler 与模拟 VAE，α=0 对 disabled |
| Stage 1 CLI | executed，GEN+Memory 与 GEN-only 各 2 optimizer steps passed | prompt/image JSONL、VAE patchify、直接 flow loss、保存 loop-only checkpoint |
| 模式与深度矩阵 CLI | executed，22 最终图导出 passed | Base + 5 模式 × 2 深度 × 2 样本；early/middle/late velocity 与 readout 图 |
| 代码静态检查 | passed | 新增模块/入口及修改的 native 核心 |
| 编译与 diff 空白检查 | passed | 当前 Python 代码及 Git diff |

测试命令在仓库根目录执行：

```bash
pytest -q
```

本次未运行完整 7B 权重的生成或训练，也未运行真实 GenEval2 / TIIF VLM judge 或人工质量评审。
结果 evaluator 的接口、计算和错误处理已验证；semantic gain、quality retention、Repair/Damage 的真实研究结论及 accelerator FLOPs 尚未验证。
CPU 小模型测试通过说明实现的数值与状态约束成立，不能证明真实图像质量或语义改善。
