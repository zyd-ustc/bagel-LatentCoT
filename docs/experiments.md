# Experiments

当前训练路线为 Observe 初始化 → 交互 SFT → Flow-GRPO。本页集中记录配置、运行命令、阶段结果与下一轮训练安排。

## 训练入口

| 阶段 | Python 入口 | NPU 脚本 | 配置 |
|---|---|---|---|
| Observe / Feedback | `qwen_latent_cot.diagnostics.visual_cli` | `scripts/visual_training_npu.sh` | `configs/visual_memory_pilot.json`、`configs/visual_feedback.json` |
| 高噪声 Observe | 同上 | `scripts/visual_training_multinode.sh` | `configs/visual_observe_highnoise_full4.json` |
| 交互 SFT | `qwen_latent_cot.interaction.cli sft` | `scripts/train_interaction_npu.sh` | `configs/interaction_sft.json` |
| Flow-GRPO | `qwen_latent_cot.interaction.cli rl` | `scripts/train_interaction_npu.sh` | `configs/interaction_rl.json` |

训练环境为与设备匹配的 PyTorch、torch_npu 和项目依赖。安装追踪依赖后，使用 W&B 记录训练：

```bash
python -m pip install -e '.[dev,tracking]'
wandb login
```

## Observe 初始化

### 全层注意力训练

原生 UND 全部 28 层读取同层 GEN KV。训练 Q/K/V/O 投影与 Q/K norm，前向使用混合精度，选定参数以 FP32 优化。首次训练设置为 16 NPU、global batch 16、学习率 1e-5、t∈[0.05,0.25]、20,000 步。

在仓库根目录运行：

```bash
export MODEL_PYTHON=/root/venvs/bagel-NPU/bin/python
export MODEL_PATH=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export TRAINING_CORPUS=/absolute/path/to/observe_feedback_corpus_v2
export C0_PASS=/absolute/path/to/calibration.json
bash scripts/train_observe_nativeund_npu.sh /root/outputs/observe_nativeund_new
```

入口先运行两步检查，再恢复优化器与采样进度继续到 20,000 步。`NPROC_PER_NODE` 控制本机进程数，`ASCEND_RT_VISIBLE_DEVICES` 指定物理设备。

### 高噪声课程

本地已有短程、单机全程和多机全程配置：`visual_observe_highnoise.json`、`visual_observe_highnoise_full.json`、`visual_observe_highnoise_full4.json`。全程课程为：

| 起始步 | Observe 最大 t | 低噪声采样比例 |
|---:|---:|---:|
| 1 | 0.5 | 0.50 |
| 2,001 | 0.6 | 0.45 |
| 5,001 | 0.7 | 0.40 |
| 8,001 | 0.8 | 0.35 |
| 12,001 | 0.9 | 0.30 |

低噪声区间为 [0.05,0.25]，高噪声区间从 0.25 延伸至当前课程上限。验证使用上限 0.9、低噪声比例 0.4。全程配置预算为 20,000 步。

单机初始化示例：

```bash
export DIAGNOSTIC_CONFIG="$PWD/configs/visual_observe_highnoise_full.json"
export VISUAL_PARENT=/absolute/path/to/observe_parent.pt
bash scripts/visual_training_npu.sh observe /root/outputs/observe_highnoise_new
```

多机环境在每个 worker 的仓库根目录执行相同命令：

```bash
export VC_WORKER_HOSTS=worker0.cluster,worker1.cluster,worker2.cluster,worker3.cluster
export MA_NUM_HOSTS=4
export NPROC_PER_NODE=16
export RUN_DIR=/shared/outputs/observe_highnoise_full4
export VISUAL_PARENT=/shared/checkpoints/observe_parent.pt
export TRAINING_CORPUS=/shared/data/observe_feedback_corpus_v2
export MODEL_PATH=/shared/models/Bagel-7B-MoT
export C0_PASS=/shared/calibration.json
export MODEL_PYTHON=/absolute/path/to/bagel-NPU/bin/python
bash scripts/visual_training_multinode.sh
```

脚本根据主机名确定 node rank，使用 `MASTER_TRAIN_INDEX` 选择主节点，`MASTER_PORT` 设置通信端口。各 worker 使用相同配置和共享数据、模型、parent、输出路径。

## 交互 SFT

### 配置

| 参数 | 默认值 |
|---|---|
| Parent | format-4 `C2-observe` 或 `C2-feedback` checkpoint |
| Writer / consumer | 两组 attention LoRA；rank=16、alpha=16、B 零初始化 |
| 投影 | UND `q/k/v/o_proj`；GEN `q/k/v/o_proj_moe_gen` |
| 学习率 | writer 2e-6；consumer 1e-5 |
| 预算 | 500 optimizer steps；global batch 16；脚本默认 16 NPU |
| 时间采样 | 原生 50 节点、shift=3 的前 20 个更新节点，均匀抽取 |
| 损失 | 交互完整速度对配对 flow target 的 MSE |
| 优化 | AdamW；weight decay 0；clip 1；warmup 20 步 |
| 保存 | 第 100 / 250 / 500 步；默认生成 4 条 val 目标的对照图 |

每卡逐样本前向与反向，按全局 batch 归一化后求和梯度。Writer 与 consumer 参数使用 FP32，原始参数保持 parent dtype。适配器开关覆盖前向与激活重计算。基础提案、prompt 编码和无条件 CFG 分支冻结。

分辨率、CFG 和重归一化等采样设置从 parent 继承。现有阶段验证使用 512px、CFG=4、49 次 Euler 更新，前 20 次启用反馈。

### 启动与恢复

在仓库根目录设置实际模型、parent 和经过审核的[图文数据](data.md)：

```bash
export MODEL_PYTHON=/root/venvs/bagel-NPU/bin/python
export MODEL_PATH=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export TRAIN_PAIRS=/absolute/path/to/reviewed/train.jsonl
export VAL_PAIRS=/absolute/path/to/reviewed/val.jsonl
export OBSERVE_PARENT=/absolute/path/to/observe_parent.pt
WANDB_MODE=online bash scripts/train_interaction_npu.sh sft /root/outputs/interaction_sft_500 \
  --parent "$OBSERVE_PARENT" --train "$TRAIN_PAIRS" --val "$VAL_PAIRS"
```

`--parent` 接受具体文件或实验目录。目录解析选择最高编号的完整 checkpoint；实际路径、哈希与 parent 噪声课程写入运行记录。

同阶段恢复使用 `--resume`：

```bash
bash scripts/train_interaction_npu.sh sft /root/outputs/interaction_sft_500 \
  --resume /root/outputs/interaction_sft_500/step_000250.pt \
  --train "$TRAIN_PAIRS" --val "$VAL_PAIRS"
```

恢复使用相同数据路径与哈希、优化设置和 worker 数。可调整总步数、保存节点和验证图数。短程运行通过 `--steps`、`--global-batch`、`--val-images` 设置预算。

## Flow-GRPO

### 配置与奖励

RL 冻结 SFT writer，更新 GEN consumer。SFT 出口保存为固定参考，当前批 rollout 分数作为旧策略分数。

| 参数 | 默认值 |
|---|---|
| 候选 | 每 prompt 8 张；每批 7 prompts，共 56 张 |
| 预算 | 100 批，共 5,600 张候选终图 |
| 设备 | policy 卡 0–13；judge 卡 14–15 |
| 随机转移 | 从 i=1…19 抽一个 SDE 转移，其余为 Euler ODE |
| SDE noise level | 0.7 |
| Consumer 学习率 | 1e-5 |
| 策略裁剪 / 参考 KL 系数 | 0.001 / 0.01 |
| 更新 | 每批一次 optimizer 更新；gradient clip=1 |
| 保存 | 第 10 / 25 / 50 / 100 批 |

Judge 使用 Qwen3-VL-8B-Instruct，对[奖励问题缓存](data.md#rl-奖励问题)计算预期答案概率。奖励取原子概率算术平均；同组 8 张候选的奖励归一化为优势。

SDE 采样与 replay 共用 FP32 转移函数。转移分数按单图 latent 维平均，KL 使用实际转移方差。每批记录更新前后 ratio、clip fraction、固定参考 KL 和零方差组比例。

### 启动

准备生成与 judge 的独立环境，以及审核后的问题文件：

```bash
export JUDGE_PYTHON=/root/venvs/qwen3vl-npu/bin/python
export JUDGE_PATH=/data/zyd_workspace/models/Qwen3-VL-8B-Instruct
export GENEVAL2_SOURCE=/root/npu-eval-tools/GenEval2/evaluation.py
export REWARD_QUESTIONS=/absolute/path/to/reviewed/train_questions.jsonl
bash scripts/train_interaction_npu.sh rl /root/outputs/interaction_rl_smoke32 \
  --checkpoint /root/outputs/interaction_sft_500/step_000500.pt \
  --train "$TRAIN_PAIRS" --val "$VAL_PAIRS" --reward-questions "$REWARD_QUESTIONS" \
  --steps 1 --prompts-per-batch 4 --val-images 0
```

32 张候选完成生成、评分、replay 和更新后，启动正式预算：

```bash
bash scripts/train_interaction_npu.sh rl /root/outputs/interaction_rl_100 \
  --checkpoint /root/outputs/interaction_sft_500/step_000500.pt \
  --train "$TRAIN_PAIRS" --val "$VAL_PAIRS" --reward-questions "$REWARD_QUESTIONS"
```

脚本建立独立评分队列，并在退出时结束本次 judge 进程。RL 同阶段恢复使用 `--resume` 替代 `--checkpoint`。

## 推理、保存与日志

单图入口为 `python -m qwen_latent_cot.interaction.cli sample`，生成命令见[项目首页](../README.md#开始使用)。完整六组对照与评分命令见 [Benchmark](benchmark.md)。

交互 checkpoint 格式为 `interaction_lora_v1`，保存原生 UND parent、两组 LoRA、阶段、配置、optimizer、sampler、各 rank 随机状态、数据来源和固定参考。基础 BAGEL 模型文件在加载时提供。

| 输出 | 内容 |
|---|---|
| `run.json` | 实际配置、parent 身份、噪声课程、数据哈希与参数量 |
| `train.jsonl` | SFT flow、基础 flow、速度差、梯度、耗时与显存；RL 策略和奖励指标 |
| `step_*.pt` | 阶段 checkpoint 与恢复状态 |
| `validation_*/` | val 的 parent / correct / masked / swap 终图及记录 |
| `rollout_*/` | RL 选定转移、状态、旧分数、group、seed 与奖励 |

`WANDB_MODE` 支持 `online`、`offline` 和 `disabled`。交互脚本默认 online；`WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_NAME` 设置项目、团队和名称。Observe / Feedback 使用 `training.jsonl`、`validation.jsonl`、`training_summary.json` 及各 rank 日志。

## 阶段结果

### Observe：2026-10-16

主机 `modelarts-job`，输出 `/root/outputs/observe_nativeund_attn_20261015`。原生 UND 全层注意力训练完成 20,000 步，t∈[0.05,0.25]，呈现 320,000 次样本，耗时 43,734 秒。

| 步数 | 验证 NLL | Source-off NLL | 两者差值 |
|---:|---:|---:|---:|
| 1,000 | 2.5865 | 2.8402 | 0.2537 |
| 5,000 | 2.3280 | 2.7551 | 0.4271 |
| 10,000 | 2.2103 | 2.7328 | 0.5225 |
| 19,750 | 2.0381 | 2.7183 | 0.6802 |
| 20,000 | 2.0481 | 2.6919 | 0.6438 |

固定验证为 113 条。`best.pt` 对应 19,750 步，`last.pt` 对应 20,000 步。可训练参数为 822,219,776 个，框架峰值约 41.7 GiB。审阅样本中，角马群、浴室、鹦鹉和自行车等主体读取改善；细类、数量、颜色和文字分别记录。

W&B：[observe_nativeund_attn_full_20k](https://wandb.ai/zyd20040407-ustc/bagel-LatentCoT/runs/rx2nomt3)。

### C2 Feedback：2026-10-16

反馈监督从 Observe 初始化，Endpoint 占 85%、Observe 占 15%，训练原生 UND 注意力，以文字交叉熵学习修正与保持。

已保存的 C2 对照为 `/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt`，SHA-256 为 `cb8479c5671c01ea07dc5b88a5a7c78577aacd3aa3141974b462eb832f73750f`。该文件用于当日五组统一推理与评测入口验证。

### 交互 SFT / RL：2026-10-17

代码部署于 `modelarts-job:/root/bagel-LatentCoT-interaction_20261017`。初始化使用 Observe `step_020000.pt`，其训练噪声上限为 0.25。每组 LoRA 包含 10,092,544 个 FP32 参数。

| 运行 | 结果 | 输出目录 |
|---|---|---|
| 单 NPU SFT | 2 步；两组梯度有限且非零；保存 `step_000002.pt` | `/root/outputs/interaction_sft_single_20261017` |
| 16 NPU SFT | 20 步；第 2–20 步平均 1.569 秒/步；峰值 38.406 GiB | `/root/outputs/interaction_sft_16npu_20261017` |
| Euler 验证 | 一条 val caption 的 parent / correct / masked / swap 四张 512px 终图 | 同 SFT 输出目录 |
| 14 policy＋2 judge | 4 prompts×8 candidates；真实概率评分后完成 consumer 更新 | `/root/outputs/interaction_rl_smoke32_v2_20261017` |

RL 批耗时为 66.383 秒，计时覆盖 rollout、评分和更新。更新前 replay ratio=1.0，consumer 梯度范数为 6.2033e-5，零方差组比例为 0，平均训练奖励为 0.730856，框架峰值为 37.914 GiB。

保存后检查显示 writer 与 parent 保持一致，consumer 的 224 个 LoRA 张量更新，固定参考与 SFT 出口一致。重载后完成 49 次 Euler 更新，输出 `reloaded_euler.png`。保存转移的更新后探针 ratio=0.999999225、KL=6.8157e-7，记录位于 `verification.json`。

小型原生 BAGEL 与 CPU 分布式检查共 25 项通过，覆盖适配器范围、两组梯度、重计算、恢复、R2 CFG 和 SDE replay。

## 下一轮训练

| 顺序 | 工作 | 阶段输出 |
|---|---|---|
| 1 | 完成 8k–16k 图文配对审核，选择 Observe parent | 数据划分、审核索引与明确 checkpoint |
| 2 | 运行 500 步交互 SFT | 第 100 / 250 / 500 步权重和 val 终图 |
| 3 | 用训练 caption 建立奖励问题，完成 32 张候选更新 | 奖励分布、分组优势、更新后 ratio 与 KL |
| 4 | 运行 100 批 Flow-GRPO | 阶段权重与终图评测 |
| 5 | 比较正常、屏蔽、交换反馈及原生、parent 基线 | 生成质量、反馈贡献与实际计算成本 |

前 20 步检查梯度、吞吐和保存恢复；第 100 步检查 val 终图；第 500 步进入 RL。出现输出退化时，按适配器作用范围、数值和学习率顺序定位。奖励集中时，检查原子问题、概率与目标难度。

文字反思诊断按需从同一配对池派生：解码当前提案，teacher 根据 caption 与提案标注 observed、correction、preserve 和 status，再对 writer 做文字交叉熵辅助训练。该扩展单独管理标签、更新次数和成本。

## 历史结果

| 日期与阶段 | 主要结果 | 权重或输出 |
|---|---|---|
| 2026-10-12 C0 | 16 条 caption 复述；平均 NLL 0.271115，严格匹配 7/16 | `/root/outputs/c0_text16_20261012_111241` |
| 2026-10-12 C1 连续 memory | 100 步后验证 NLL 0.458495，字面事实分 45.83% | `/root/outputs/c1_pilot_20261012_113553/C/best.pt` |
| 2026-10-12 C1 响应训练 | 100 步响应蒸馏后，局部图像审阅观察到 memory 内容影响 | `/root/outputs/c1_response_20261012_134848` |
| 2026-10-12 C1 扩展响应 | 累计 1,100 步，验证 response 0.018171、NLL 0.163620 | `/root/outputs/c1_response_expand_20261012_141729/best.pt` |
| 2026-10-13 显式反馈 | 16 个 prompt、80 张终图；人工审阅观察到语义编辑 | `scripts/explicit_feedback_npu.sh` |

显式反馈运行入口为 `qwen_latent_cot.diagnostics.feedback_cli`，固定 memory 的窗口与产物检查入口为 `qwen_latent_cot.diagnostics.feedback_audit`。旧层循环、文字压缩与响应蒸馏作为连续反馈的研究参照。

## 代码位置与方法来源

| 文件 | 职责 |
|---|---|
| `interaction/adapters.py`、`interaction/student.py` | 双 LoRA、作用范围与 GEN→UND→GEN |
| `interaction/data.py`、`interaction/training.py` | 配对数据、SFT、验证与日志 |
| `interaction/checkpoint.py` | 保存恢复、随机状态和固定参考 |
| `interaction/sde.py`、`interaction/rl.py` | 随机转移、分组优势与策略更新 |
| `interaction/rewards.py` | 原子问题、概率评分和 judge 队列 |
| `diagnostics/inference.py` | 原生、parent 和交互统一推理 |

方法借鉴 [Correction Training](https://github.com/sunovivid/correction-training/tree/475e21464485c8c3b38730131d071042ad6481f1) 对当前提案施加生成监督的训练任务，以 UND＋第二遍 GEN 承担修正计算。[Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) 提供 BAGEL 随机转移与策略优化参考，本项目使用 rank-16 attention-only LoRA 和单随机转移训练。
