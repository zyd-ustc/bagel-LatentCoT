# 运行

状态：保留C；16卡GEN启动器已实现，颜色配对标签待准备。更新：2026-10-12。

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

### 已完成的C1首跑（仅供复现）

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

### 当前固定checkpoint与下一步

后续只使用C，不重跑上面的L/C首跑脚本：

```bash
export MODEL=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export C1_C=/root/outputs/c1_pilot_20261012_113553/C/best.pt
export C0_PASS=/root/outputs/c0_text16_20261012_111241/calibration.json
```

C为step 100，SHA256与选择原因见[验证](validation.md#4-c1结果与拓扑选择)。checkpoint保持不变，不覆盖原输出，不按GEN结果重选。

下一步先准备[GEN颜色校准](experiments.md#当前首跑c的颜色消费)的16张核验图像与8个场景对。每个worker的数据必须包含一对及双方 `swap_id`，图像路径使用绝对路径或按分片文件位置正确解析。与其对应的worker配置保留默认接口参数和 `train_t`，仅将 `eval_t`设为 `[0.5,1.0]`、`noise_seeds`设为该worker的 `[0]`或 `[1]`；GEN入口跳过t=1，不额外采样。按 `pair_id × seed`分配16个worker，输出目录分别独立，汇总时按原场景对聚合。

16卡总启动器已实现。数据根目录固定为 `/data/zyd_workspace/bagel-LatentCoT/datasets`，默认清单为该目录下 `memory_gen_color16.jsonl`，可用 `--data`覆盖。检查到的现有CoRT/结构化编辑清单不满足颜色配对格式：含编辑目标、未核验标记或旧图像路径。不得自动将target prompt当成图像真值；目前默认颜色清单尚不存在。

准备清单后，在仓库根目录执行：

```bash
OUT=/root/outputs/gen_color_C_$(date +%Y%m%d_%H%M%S)
nohup /root/venvs/bagel-NPU/bin/python -u scripts/gen_npu.py \
  --dataset-root /data/zyd_workspace/bagel-LatentCoT/datasets \
  --output "$OUT" > "$OUT.log" 2>&1 < /dev/null &
echo "$OUT"
tail -f "$OUT.log"
```

`--prepare-only`只写16份配对分片/配置与 `plan.json`，不加载BAGEL；输出目录必须全新。正常启动自动准备后执行，不能复用prepare-only目录。缺失或不合协议的数据在启动worker前报错。

汇总为 `summary.json/gen.jsonl/timing.jsonl`；逐卡日志、manifest与状态在 `workers/`。`local_blind/`、`local_blind_annotations.csv`与单独保留的 `local_blind_key.json`用于预测图审查；`observed_color`填写red/blue/other/unknown，其他字段记录对象保留、画质与伪影。启动器不自动判定语义成功，不将局部预测图记为最终生成图。汇总核对各卡模型、源码、checkpoint与配置，并要求224条完整arm记录。

下面是**启动器内部的单worker模板**，不是16卡总启动命令：

```bash
/root/venvs/bagel-NPU/bin/python -m qwen_latent_cot.diagnostics.cli gen \
  --model "$MODEL" --data "$PAIR_DATA" --config "$WORKER_CONFIG" \
  --device npu:0 --split test --topology C --text-checkpoint "$C1_C" \
  --c0-pass "$C0_PASS" --output "$WORKER_OUT"
```

启动器为每个worker设置对应的 `ASCEND_RT_VISIBLE_DEVICES`，进程内部设备为 `npu:0`。GEN不传 `--distributed`、`--visual-checkpoint`或 `--continue-sampling`。先用速度、预测图与7组既有对照判断消费接口，结果与继续条件见[实验](experiments.md)。当前C1文本数据没有图像，不可直接作为GEN输入。

### 后续C2（尚未启动）

只有文本Memory在GEN颜色任务中产生合理语义作用、预测图未出现明显损伤后，才准备独立图像-caption训练/验证数据。C2沿用C拓扑，以 `$C1_C`初始化，caption仅用于监督。使用16卡联合训练，暂不恢复L或scratch组；GEN首轮校准数据不混入C2训练后再当独立测试。响应蒸馏若先成为下一步，需要实现损失入口后另给启动命令。

## 4. 轨迹与后续采样

C2-scratch 使用 `train --stage C2 --scratch`，省略 `--checkpoint`，两组预算保持一致。`last.pt` 使用相同 Caption 命令单独评估。

`trajectory` 命令从原生 prompt/seed 运行采样，保存真实 `x_t/t/step` 和条件速度预测图；候选观察标签默认 `verified: false`。人工核验 caption/facts/unknown 后，才能作为观察数据进入诊断。

GEN 局部结果通过语义检查后，使用新输出目录加 `--continue-sampling` 做后续采样。每个 arm 只在选定节点读取一次 Memory，随后恢复原生。先按协议固定小规模数据及节点，避免默认展开所有 t×arm。匿名图像与 `blind_annotations.csv` 用于分别核验事实和画质；`blind_key.json` 不交给盲评者。

默认配置为 [memory_topology.json](../configs/memory_topology.json)。训练上限是每组 5,000 optimizer steps、有效 batch 32，微批为 1；每 250 步验证。支持writer数据并行训练；尚不支持断点恢复。输出目录必须为空，失败和已完成运行不会被覆盖。
