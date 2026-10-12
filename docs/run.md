# 运行

状态：适用于当前CE实现。更新：2026-10-12。

[返回 README](../README.md)

本页命令不包含拟议的响应蒸馏或语义读出损失。目标主机的真实权重实验由用户执行。数据格式见[实现](implementation.md)，实验顺序见[实验](experiments.md)。

Ascend 实验默认使用 `loop-layer-NPU` 分支和16卡并行。评测按样本分片；训练用 `torchrun --distributed` 同步writer梯度。`effective_batch`是全局batch，须被卡数整除；每卡微批1。

## 1. 安装与检查

在本仓库根目录执行。先安装目标设备的 PyTorch；Ascend 还需要与环境匹配的 `torch_npu`，CUDA 原生路径需要 `flash-attn`。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
memory-topology --help
```

唯一启动入口为 `memory-topology`。等价入口是 `python -m qwen_latent_cot.diagnostics.cli`；`scripts/memory_topology.py` 只是同一入口的薄封装。

## 2. 原生文本校准

### modelarts-job：16条文本、16卡

在同步后的 `/root/bagel-LatentCoT-loop-layer-NPU` 执行：

```bash
cd /root/bagel-LatentCoT-loop-layer-NPU
OUT=/root/outputs/c0_text16_$(date +%Y%m%d_%H%M%S)
/root/venvs/bagel-NPU/bin/python scripts/c0_npu.py \
  --model /data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT \
  --output "$OUT"
```

启动器让物理逻辑设备0–15各处理一条caption，进程内部均用 `npu:0`。它保留原生C0入口与默认配置，汇总16条结果，不自动判定通过。每卡加载一份冻结BAGEL。

输入为 [c0_text16.jsonl](../data/c0_text16.jsonl)。这些人工编写文本只用于复述校准；`verified`指文本事实核对，不代表核验了图像。不能用它们作C2视觉标签。

查看 `$OUT/captions.jsonl` 的 `caption/text`，以及 `$OUT/summary.json`。逐卡日志和manifest在 `$OUT/workers/`。核验全部16条后，在汇总的 `$OUT/calibration.json` 记录 `passed: true` 与 `reviewer`，后续使用这个路径。

### 通用入口

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

## 3. 训练与评测

### modelarts-job：C1首跑

C0已按[验证记录](validation.md)人工放行，范围仅为原生文本语义读取。首跑使用 [c1_pilot.json](../configs/c1_pilot.json) 与 [c1_text80.jsonl](../data/c1_text80.jsonl)：64/8/8条文本、100步、全局batch16、每50步验证。L/C依次各用16卡联合训练，只更新writer；每组处理1,600条训练样本。模板文本不代表图像核验数据。

```bash
cd /root/bagel-LatentCoT-loop-layer-NPU
OUT=/root/outputs/c1_pilot_$(date +%Y%m%d_%H%M%S)
nohup bash scripts/c1_npu.sh "$OUT" > "$OUT.log" 2>&1 < /dev/null &
echo "$OUT"
tail -f "$OUT.log"
```

默认C0路径为 `/root/outputs/c0_text16_20261012_111241/calibration.json`；可用环境变量 `C0_PASS`覆盖。`MODEL_PATH`、`DIAGNOSTIC_DATA`、`DIAGNOSTIC_CONFIG`、`MODEL_PYTHON`同样可覆盖。C0的16条数据全属val，不能直接作为C1训练集。

结果分别位于 `$OUT/L`、`$OUT/C`。`train.jsonl`记录全局NLL、16卡全局样本和同步后梯度范数；验证与checkpoint仅由rank0写入。检查自由输出与错配事实，再决定是否扩展训练。此入口没有断点恢复。

### 通用入口

```bash
# 2. 相同预算训练 C1-L/C；输出 init.pt、best.pt、last.pt。
for TOPOLOGY in L C; do
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    -m qwen_latent_cot.diagnostics.cli train --distributed \
    --model "$MODEL" --data "$DATA" --device "$DEVICE" \
    --stage C1 --topology "$TOPOLOGY" --c0-pass "$OUT/c0/calibration.json" \
    --output "$OUT/c1_$TOPOLOGY"
done

# 3. 各自从 C1 的共同验证规则所选 checkpoint 迁移至 C2。
for TOPOLOGY in L C; do
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    -m qwen_latent_cot.diagnostics.cli train --distributed \
    --model "$MODEL" --data "$DATA" --device "$DEVICE" \
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

## 4. 轨迹与后续采样

C2-scratch 使用 `train --stage C2 --scratch`，省略 `--checkpoint`，两组预算保持一致。`last.pt` 使用相同 Caption 命令单独评估。

`trajectory` 命令从原生 prompt/seed 运行采样，保存真实 `x_t/t/step` 和条件速度预测图；候选观察标签默认 `verified: false`。人工核验 caption/facts/unknown 后，才能作为观察数据进入诊断。

GEN 局部结果通过语义检查后，使用新输出目录加 `--continue-sampling` 做后续采样。每个 arm 只在选定节点读取一次 Memory，随后恢复原生。先按协议固定小规模数据及节点，避免默认展开所有 t×arm。匿名图像与 `blind_annotations.csv` 用于分别核验事实和画质；`blind_key.json` 不交给盲评者。

默认配置为 [memory_topology.json](../configs/memory_topology.json)。训练上限是每组 5,000 optimizer steps、有效 batch 32，微批为 1；每 250 步验证。支持writer数据并行训练；尚不支持断点恢复。输出目录必须为空，失败和已完成运行不会被覆盖。
