# BAGEL：冻结模型的独立 Memory 拓扑诊断

本页对应 `loop-layer` 分支。实现基点为 `loop-layer-NPU@42e71148826011b74473a170c4ab62072d6c70ab`。
本分支只保留新诊断路径。历史循环、prompt 替换、观察反馈、旧测试及旧启动脚本均已移除。

依据：[诊断协议](docs/memory_topology_diagnostic_protocol_20261011.md)。代码映射、数据格式和验证边界见[实现说明](docs/MEMORY_TOPOLOGY_IMPLEMENTATION.md)。

- L 每层从自己的固定中性种子开始。C 将更新后的 hidden 传给下一层。
- 两组各层执行一次完整冻结 UND block 和一个低秩残差。两组均使用更新后的 hidden 做同层读出。
- 固定 16 槽位、512 分辨率、全部模型层。仅 Caption 阶段训练新增残差；原 BAGEL、VAE 和 GEN 始终冻结。
- Caption decoder 只读 Memory、固定指令和已生成答案前缀。GEN 保留原 prompt，仅在选定 denoiser 调用读取一次 Memory。
- `Memory-masked` / `G-layout` 屏蔽 attention 中的对应位置。Memory K 保存为未旋转形式，消费时只应用一次 RoPE。

本地检查使用随机初始化的小型原生 BAGEL 层和测试用 VAE/tokenizer。未运行真实 BAGEL 权重、NPU/GPU 训练或质量评测。

## 安装与工程检查

在本仓库根目录执行。先安装目标设备的 PyTorch；Ascend 还需要与环境匹配的 `torch_npu`，CUDA 原生路径需要 `flash-attn`。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
memory-topology --help
```

唯一启动入口为 `memory-topology`。等价入口是 `python -m qwen_latent_cot.diagnostics.cli`；`scripts/memory_topology.py` 只是同一入口的薄封装。

## 按协议运行

先准备核验数据。`DATA` 指向同一份包含 train/val/test 的 JSONL，按原图/场景族固定 split，包含反事实配对。
下面的模型、数据和输出路径需要换成目标主机的实际路径。所有运行使用同一配置；缩减预算时，L/C 必须一起缩减。

```bash
export MODEL=/path/to/BAGEL
export DATA=/path/to/verified_memory_dataset.jsonl
export OUT=/path/to/memory_topology_outputs
export DEVICE=npu:0

# 1. 在验证集校准原生文本 oracle 的自由解码。
memory-topology calibrate --model "$MODEL" --data "$DATA" --device "$DEVICE" \
  --split val --output "$OUT/c0"
```

检查 `c0/captions.jsonl` 的自由文本与参考 caption。确认解码协议可用后，在 `c0/calibration.json` 记录 `passed: true` 和核验者。代码校验同一模型、tokenizer、VAE及解码指令，不通过时禁止训练和 GEN 结果解释。

```bash
# 2. 相同预算训练 C1-L/C；输出 init.pt、best.pt、last.pt。
for TOPOLOGY in L C; do
  memory-topology train --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --stage C1 --topology "$TOPOLOGY" --c0-pass "$OUT/c0/calibration.json" \
    --output "$OUT/c1_$TOPOLOGY"
done

# 3. 各自从 C1 的共同验证规则所选 checkpoint 迁移至 C2。
for TOPOLOGY in L C; do
  memory-topology train --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --stage C2 --topology "$TOPOLOGY" --checkpoint "$OUT/c1_$TOPOLOGY/best.pt" \
    --c0-pass "$OUT/c0/calibration.json" --output "$OUT/c2_$TOPOLOGY"
done

# 4. 测试集 Caption 对照；分别评估 init、best、last。
for TOPOLOGY in L C; do
  memory-topology caption --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --stage C1 --topology "$TOPOLOGY" --checkpoint "$OUT/c1_$TOPOLOGY/init.pt" \
    --output "$OUT/c1_init_test_$TOPOLOGY"
  memory-topology caption --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --stage C1 --topology "$TOPOLOGY" --checkpoint "$OUT/c1_$TOPOLOGY/best.pt" \
    --output "$OUT/c1_test_$TOPOLOGY"
  memory-topology caption --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --stage C2 --topology "$TOPOLOGY" --checkpoint "$OUT/c2_$TOPOLOGY/best.pt" \
    --output "$OUT/c2_test_$TOPOLOGY"
done

# 5. GEN 局部阶梯。原 prompt 保留，CFG 分支保持原生。
for TOPOLOGY in L C; do
  memory-topology gen --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --topology "$TOPOLOGY" --text-checkpoint "$OUT/c1_$TOPOLOGY/best.pt" \
    --visual-checkpoint "$OUT/c2_$TOPOLOGY/best.pt" \
    --c0-pass "$OUT/c0/calibration.json" --output "$OUT/gen_$TOPOLOGY"
done

# 6. 合并相同模型、数据、源码和配置的结果，生成四张表。
memory-topology report --runs "$OUT/c1_test_L" "$OUT/c1_test_C" \
  "$OUT/c2_test_L" "$OUT/c2_test_C" "$OUT/gen_L" "$OUT/gen_C" \
  --output "$OUT/report"
```

C2-scratch 使用 `train --stage C2 --scratch`，省略 `--checkpoint`，两组预算保持一致。`last.pt` 使用相同 Caption 命令单独评估。

`trajectory` 命令从原生 prompt/seed 运行采样，保存真实 `x_t/t/step` 和条件速度预测图；候选观察标签默认 `verified: false`。人工核验 caption/facts/unknown 后，才能作为观察数据进入诊断。

GEN 局部结果通过语义检查后，使用新输出目录加 `--continue-sampling` 做后续采样。每个 arm 只在选定节点读取一次 Memory，随后恢复原生。先按协议固定小规模数据及节点，避免默认展开所有 t×arm。匿名图像与 `blind_annotations.csv` 用于分别核验事实和画质；`blind_key.json` 不交给盲评者。

默认配置为 [memory_topology.json](configs/memory_topology.json)。训练上限是每组 5,000 optimizer steps、有效 batch 32，微批为 1；每 250 步验证。当前不实现分布式训练或断点恢复。输出目录必须为空，失败和已完成运行不会被覆盖。
