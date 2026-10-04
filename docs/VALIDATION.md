# 本地重构验收

日期：2026-10-04。主机：本地 macOS，CPU。
分支：`codex/umm-t2i-anchored-loop`。

验证环境使用 Python 3.12.14、PyTorch 2.14.1、Transformers 4.57.6。
模型测试使用真实四层 BAGEL MoT decoder，hidden size 32，bfloat16；入口测试使用小型模拟 VAE 和 tokenizer。

| 检查 | 执行结果 | 证据范围 |
| --- | --- | --- |
| 完整测试集 | 55 passed | 包含模式、K=0/8/16、CFG、固定 KV/`x_t`、checkpoint、loss 与梯度 |
| 原生路径对照重构前快照 | passed，velocity 最大绝对误差 0.0 | 相同小模型权重与输入，velocity shape `[32,8]` |
| 新 workspace 关闭循环 / R=0 / α=0 | passed，velocity exact parity | 小型真实 MoT；全局、逐 channel 与 text-channel CFG |
| Current MemLoop vs legacy compatibility | passed，5 组 velocity 最大绝对误差 0.0 | 独立子进程加载冻结 parent `6f936b7`；K=8/16，R=1/2，batch=1/2，global/channel/text_channel CFG，含零 gate/α与训练参数污染检查 |
| Memory slot 对称性 | passed | 实际 BF16 初始化算术；K=8/16 唯一 slot，初始及循环后 variation 非零；collapsed slots 的 centered rank=0 |
| Stage 1 timestep 分布 | passed，逐值误差 0.0 | seed 固定，4096 个 normal draw → sigmoid → shift；shift=1/3，native training forward 接入检查 |
| GEN correction gate | passed | native transformation 为 reference；只 gate GEN；真实 MoT 的 memory full update 未被 gate |
| GEN-only 初始 parity 与训练梯度 | passed | α=0.01 + zero adapter 初始 velocity exact parity；bias 梯度非零；实际两步训练 bias 改变 |
| checkpoint v2 / 拒绝 v1 | passed | 不允许把旧 whole-layer gate 参数静默用作新 correction gate |
| 新 workspace 所有 loop gate=0 | passed | 即使输出 α 非零也返回 native readout |
| 最终图片等价性 | passed，逐字节相同 | 小模型 sampler 与模拟 VAE，α=0 对 disabled |
| Stage 1 CLI | executed，GEN+Memory 与 GEN-only 各 2 optimizer steps passed | prompt/image JSONL、VAE patchify、直接 flow loss、保存 loop-only checkpoint |
| 模式与深度矩阵 CLI | executed，18 最终图导出 passed | Base + 4 模式 × 2 深度 × 2 样本；early/middle/late velocity 与 readout 图 |
| 代码静态检查 | passed | 新增模块/入口及修改的 native 核心 |
| 编译与 diff 空白检查 | passed | 当前 Python 代码及 Git diff |

测试命令在仓库根目录执行：

```bash
pytest -q
```

本次未运行完整 7B 权重的生成或训练，也未验证 GenEval2、通用质量、Repair/Damage 或 accelerator FLOPs。
CPU 小模型测试通过说明实现的数值与状态约束成立，不能证明真实图像质量或语义改善。
