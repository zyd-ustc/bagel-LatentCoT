# Training-free Memory body loop：代码验证

日期：2026-10-05。代码在 main 分支恢复只回传 Memory hidden 的旧路径，真实权重的评测由用户执行。

## 已执行

在 H200 主机的 CPU 上运行：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /private/software/conda/envs/lcot/bin/python -m pytest tests -q
```

结果：**40 passed，2 skipped**。跳过项为 CUDA kernel 一致性和 CUDA native QA packing。没有占用 GPU，也没有加载真实 BAGEL 权重执行生成或评测。

覆盖：

- 同权重、同输入的冻结旧路径和新路径 hidden/velocity 逐元素一致；R=1/2/4，单样本及不同长度双样本。
- R=0、K=0 和 Base 原生绕过；no-read/native 逐元素一致。
- body 每轮 GEN 与边界恢复入口，只有 Memory 回传；prefix/suffix 各一次。
- strict Read 阻断 Memory→GEN 与 Memory→边界；Memory 仍更新。
- prompt cache 不变、不同样本隔离、连续 denoiser 调用无状态泄漏。
- 不新增或修改模型参数，异常恢复原生方法，采样进度范围控制。
- 诊断与 KV 捕获不改变输出，slot collapse 指标和离线 QA 结构检查。
- 配对语义、质量、Repair/Damage、覆盖与 provenance 校验。

Python 编译、相关 shell 语法、CLI help 和 diff whitespace 检查通过。原生 vendor 文件的来源测试通过。

## 未执行

真实 BAGEL 权重上的 GPU velocity parity、正式 T2I 配对生成、最终图像评分、Memory QA 和预算测量均未执行。上述 CPU 结果只能验证实现，不能证明图像质量保持或语义改进。用户运行命令见 MEMORY_LOOP_RUNBOOK.md。
