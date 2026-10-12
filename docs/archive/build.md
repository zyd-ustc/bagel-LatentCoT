# 首次构建

状态：历史工程记录。记录日期：2026-10-11。

日期：2026-10-11（Asia/Shanghai）。分支：`loop-layer`。实现基点：`42e71148826011b74473a170c4ab62072d6c70ab`。

验证主机：本地macOS arm64，zyddeMacBook-Air.local。Python 3.13.7，PyTorch 2.14.1，transformers 4.57.6。环境位于工作区 `.venv-memory-topology`。

已执行：

- `python -m pytest -q`：23 passed。日志位于工作区 `artifacts/memory_topology_build_20261011/tests.log`。
- 生产代码、入口和新测试的 `compileall`：通过。
- `python -m build --wheel --no-isolation`：成功。
- 安装wheel后在仓库外执行 `memory-topology --help`：通过，导入路径来自site-packages；安装包在无Git checkout时仍可记录源码hash，不伪造commit或branch。
- wheel包含新diagnostics；未包含旧evaluation、internal_loop、full_depth_memory或observation_memory模块。
- `git diff --cached --check`：通过；新入口及运行模块无旧测试/诊断引用。

wheel位于工作区 `artifacts/memory_topology_build_20261011/bagel_latent_cot-0.6.0-py3-none-any.whl`。
SHA256：`1bd28f5a55d9af16c4b8e6d56d0a5124c131d8d01d3264fefcb62e0456260cbc`。

测试使用真实小型原生Qwen2MoT/BAGEL模块，覆盖L/C传播、native prefill种子和GEN KV等价、post-block读出、梯度与冻结、样本隔离、完整bank交换、因果移位、真mask、单次RoPE、pure-noise/RNG独立性、checkpoint模型来源、split泄漏、C1→C2→GEN、轨迹、报告及C0入口的原生VAE配置dataclass序列化。完整流程中的tokenizer和VAE是测试替身。

未执行真实BAGEL权重、目标NPU/CUDA全模型检查、正式训练或人工质量评测。本记录支持工程代码构建状态，不支持caption接地、GEN语义消费或画质改善结论。

[返回实验记录](../experiments.md#10-工程验证记录) · [返回 README](../../README.md)
