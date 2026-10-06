# 分层 Memory KV：代码验证

日期：2026-10-06。分支 main，远端源码目录 `/private/yida_workspace/bagel-LatentCoT-main-layerwise-kv-20261006`；对应提交见该目录的 SYNC_COMMIT。

代理仅在 H200 主机的 CPU 上执行合成检查：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /private/software/conda/envs/lcot/bin/python -m pytest -q tests
```

结果：51 passed，4 skipped。跳过旧路径 CUDA parity、原生 QA CUDA packing、分层 writer 两个 CUDA packing 配置。没有加载真实 BAGEL checkpoint，也没有执行 GPU 生成、评分或问答。

验证了 R1/2/4 的同层 bank、原生输入 KV 来源、固定 GEN/writer 入口、第一层没有伪反馈、suffix 一次、prompt cache 对象及内容不变、原生 R0/K0/no-read、packed 样本隔离、null CFG 原生绕过、Memory KV 随当前 GEN 改变、捕获不干扰运算、QA seed 来源标注、三组配对统计及旧冻结实现的 hidden/velocity parity。原生 vendor 来源检查通过；模型参数没有新增或修改。

Python 编译、shell 语法、CLI help 和 diff whitespace 检查通过。CPU FlashAttention 替代计算是小模型测试 oracle，不能替代真实 CUDA kernel 与真实权重检查。新路径是否改善语义并保留图像质量仍未验证，执行命令见 LAYERWISE_MEMORY_RUNBOOK.md。
