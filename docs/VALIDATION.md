# 本地重构验收

日期：2026-10-04。主机：本地 macOS，CPU。
分支：`codex/umm-t2i-anchored-loop`。

验证环境使用 Python 3.12.14、PyTorch 2.14.1、Transformers 4.57.6。
模型测试使用真实四层 BAGEL MoT decoder，hidden size 32，bfloat16；入口测试使用小型模拟 VAE 和 tokenizer。

| 检查 | 执行结果 | 证据范围 |
| --- | --- | --- |
| 完整测试集 | 43 passed | 包含模式、K=0/8/16、CFG、固定 KV/`x_t`、checkpoint、loss 与梯度 |
| 原生路径对照重构前快照 | passed，velocity 最大绝对误差 0.0 | 相同小模型权重与输入，velocity shape `[32,8]` |
| 关闭循环 / R=0 / α=0 | passed，velocity exact parity | 小型真实 MoT；全局、逐 channel 与 text-channel CFG |
| 所有 loop gate=0 | passed | 即使输出 α 非零也返回 native readout |
| 最终图片等价性 | passed，逐字节相同 | 小模型 sampler 与模拟 VAE，α=0 对 disabled |
| Stage 1 CLI | executed，2 optimizer steps passed | prompt/image JSONL、VAE patchify、直接 flow loss、保存 loop-only checkpoint |
| 模式与深度矩阵 CLI | executed，14 最终图导出 passed | Base + 3 模式 × 2 深度 × 2 样本；early/middle/late velocity 与 readout 图 |
| 代码静态检查 | passed | 新增模块/入口及修改的 native 核心 |
| 编译与 diff 空白检查 | passed | 当前 Python 代码及 Git diff |

测试命令在仓库根目录执行：

```bash
pytest -q
```

本次未运行完整 7B 权重的生成或训练，也未验证 GenEval2、通用质量、Repair/Damage 或 accelerator FLOPs。
CPU 小模型测试通过说明实现的数值与状态约束成立，不能证明真实图像质量或语义改善。
