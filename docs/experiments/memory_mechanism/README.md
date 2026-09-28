# Phase 0.5：Memory mechanism vs content

> 历史协议（2026-09-27 已替换）。当前同名评测入口只运行 normal/R=2,4,6,8，
> 默认 8 条 prompt；请使用 [normal R 消融说明](../normal_r_ablation/README.md)。
> 下文六组命令不再对应当前代码，不应用于新运行。已有 hard64 结果与报告保持不变。

统一入口：`scripts/evaluate/bagel_memory_mechanism.py`。
这次重构实现 frozen BAGEL 的 zero-shot 机制实验，不加载训练 checkpoint，
不训练、不插入 LoRA、不 mask prompt KV，也不做 loop 位置搜索。

## 固定协议

K=8、R=2、body=[12,20)，每个 timestep 重新初始化 memory。
先 Read，再将非 memory hidden 恢复到 body 入口，最后执行一次 Write 和 suffix。
prefix/Read 中所有非 memory token 都不能读取 memory；Write 起 GEN 可读取 memory。
prompt KV 全程保持可见。初始化沿用模型原生 SOI/EOI 方案，不由 prompt 内容初始化。

| Arm | 模式名 | 干预 |
| --- | --- | --- |
| A | `native` | 无 memory 位置，原生单遍 forward |
| B | `static_null` | 保留位置；每层 hidden 和投影后 Q/K/V 强制为零，包括 suffix |
| C | `zero_dynamic` | Read 照常计算但丢弃；Write 入口清零，此后正常演化 |
| D | `normal` | Write 从正确的 Read memory 开始，正常演化 |
| E | `shuffled_dynamic` | 成对交换完整的 Read memory，然后正常演化 |
| F | `frozen_correct` | 正确的 Read hidden 在 Write/suffix 中固定，各层重新做自身投影 |

B 的零化发生在 bias、Q/K norm、RoPE 之后，防止投影重新引入非零内容。
E 按相邻两条不同 prompt 配对，配对跨 seed、分片固定；条件和无条件 CFG 分支
分别交换自身的 Read，不将条件 memory 传给无条件分支。

同状态探测与最终生图分开：每一步 A–F 使用完全相同的 native `x_t`，只由 A
推进探测轨迹；六组最终图片再各自从同一个初始噪声独立生成。
默认 512×512、seed=42、50 个时间点（49 次去噪）、timestep shift=3、text CFG=4。
A 的 transformer 计算量较小；相同去噪步数不代表 FLOPs 匹配。

## H200 / CUDA 执行

下面命令由用户在更新代码后的 H200 环境执行；本次开发没有执行 GPU 实验。
先进入 H200 上的 `bagel-LatentCoT-v1` 仓库根目录，激活已安装 CUDA 版 PyTorch
及项目依赖的环境。启动器默认使用 `python`，可用 `PYTHON_BIN` 指定解释器。
默认模型位置为仓库内 `models/Bagel-7B-MoT`，请用 `MODEL_PATH` 指向实际权重目录。
输出目录必须不存在，脚本拒绝覆盖旧结果。不需要安装 `torch_npu`。
FlashAttention 是可选依赖；未安装时使用已有 PyTorch SDPA 路径。

1. 在仓库根目录检查 CUDA，并运行核心测试：

```bash
python -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported(), "BF16 unsupported"'
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_memory_mechanism.py tests/test_bagel_memory_mechanism_cli.py \
  tests/test_memory_mechanism_runtime.py tests/test_mot_loop_phase0.py
```

2. 单卡 smoke：两条 prompt、三个时间点，验证完整输出链，不用于判断图像质量。

```bash
MODEL_PATH="$PWD/models/Bagel-7B-MoT" \
MAX_PROMPTS=2 NUM_SHARDS=1 NUM_STEPS=3 \
  bash scripts/evaluate/run_bagel_memory_mechanism.sh \
  "$PWD/outputs/memory_mechanism_h200_smoke_$(date +%Y%m%d_%H%M%S)"
```

3. smoke 成功后，在持久终端会话中跑 hard16。自动使用可见 GPU，最多 8 卡；
   只有 1 卡时依次处理 8 对 prompt，不需要修改实验协议。

```bash
MODEL_PATH="$PWD/models/Bagel-7B-MoT" SEEDS=42 \
  bash scripts/evaluate/run_bagel_memory_mechanism.sh \
  "$PWD/outputs/memory_mechanism_h200_hard16_$(date +%Y%m%d_%H%M%S)"
```

4. 多 seed 仅替换为 `SEEDS=42,43,44`。更大数据集通过 `BENCHMARK_DATA` 和
   偶数 `MAX_PROMPTS` 指定；每个 worker 必须处理完整的一对，因此 hard16 最多
   使用 8 个 worker，32 条 prompt 才能使用 16 个 worker。

启动器默认 `BACKEND=cuda`，尊重已有 `CUDA_VISIBLE_DEVICES`（数字编号或 GPU UUID），
将其中的 GPU 逐一隔离给 worker；每个进程内使用 `cuda:0`，一张卡放一个完整模型。
未设置可见列表时使用 PyTorch 可见的 GPU。可用 `NUM_SHARDS=1` 限制为单 worker，
显式指定的数量超过可见 GPU 或 prompt 对数会报错。调度器已设置可见列表时不要重写它；
未使用调度器时可自行设置，例如 `CUDA_VISIBLE_DEVICES=2,3`。
请先确保分配的卡空闲；这里不是跨卡模型切分或 tensor parallel。

加载权重前检查 CUDA 可用性、设备编号和 BF16；显式 CUDA 不会退回 NPU。
每个分片的 manifest 记录实际设备名、CUDA/PyTorch 版本、计算能力和显存容量。
运行时长和峰值显存尚未在 H200 测量，不能由 CPU 测试推断。所有分片成功后自动合并；
分片失败会停止合并并保留日志，不会把不完整运行标为完成。

需要回到 Ascend 时显式设置 `BACKEND=npu`、`PYTHON_BIN` 和 `MODEL_PATH`；
NPU 仍使用 `ASCEND_RT_VISIBLE_DEVICES` 隔离，不改变六组协议。

## 输出与解释

- 根目录 `index.html`：六列图像、逐步曲线和指标链接；`run_manifest.json`：
  协议、源码哈希、prompt 配对、seed、完成状态；`shard_*.log`：各卡日志。
- 分片内保存每一步 native `x_t`、逐 prompt `trace.jsonl`、`attention.jsonl`、
  `memory_checks.jsonl` 和六组图片。合并会检查全部 prompt/seed、图像和时间步产物。
- 速度差依次为 B−A（位置/拓扑）、C−B（动态 workspace）、D−C（内容）、
  D−E（特异性）、D−F（动态联合演化）。同时记录绝对范数和以 `||v_A||+1e-12`
  归一化的相对范数，以及 `cos(D−A,C−A)`；零向量余弦记为 null。
- 条件分支记录 body 末尾和归一化 suffix 末尾的 GEN hidden 差异；注意力仅在
  body 第 12、19 层逐 timestep 记录，对全部 query/head 分块精确求均值，
  不保存完整 attention map。最终独立生图阶段关闭这些注意力诊断。
- Pixel MAE 仅反映行为差异，不是语义质量分数；尚无真实 NPU 结果，不能据此
  宣称 memory 内容有效或无效。

## 兼容性与验证

旧 zero-shot / write-sensitivity / single-pass T2I 命令行入口已停用，会提示迁移。
历史底层接口仍保留，以免改变既有训练路径；新入口禁止混入旧 mask、M 注入等控制。

2026-09-26 H200 适配后，本地隔离 CPU 环境下，整个 `tests` 目录 **204 项通过**。
覆盖严格零 QKV（含非零 bias）、Read/Write 调用序列、非 memory reset、整块 shuffle、
固定 hidden、原生速度一致性、CFG/cache 隔离、同状态轨迹，以及多 seed/多分片产物链。
另覆盖 CUDA/BF16 检查、GPU 可见列表/UUID、单卡隔离、超额分片拒绝、失败后不合并，
以及安装 FlashAttention 时 CPU 单测仍选用 SDPA。新模块编译、启动器语法和无权重
dry-run 通过。该验证不替代真实权重/H200 smoke；目前没有实际 CUDA kernel 测试结果。

本地测试解释器：`/tmp/bagel-mechanism-tests.WOiF4o/bin/python`，未修改原有 Python 环境。
原入口备份位于 `/tmp/bagel-mechanism-backup.yDAIvs/`，属于临时目录，应按需另行留存。
本地副本的 `.git` 仍指向缺失的远端 worktree，未重写 Git 元数据、未提交或推送。
具体边界与状态见 `PLAN.md` 和 `CHECKLIST.md`。
