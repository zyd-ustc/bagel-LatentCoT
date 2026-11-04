# 目标驱动纠偏实现规范

本文档记录 2026 年 11 月 4 日定稿的目标驱动纠偏实现，接口标识为 `kind="u15_target_correction_v1"`。该方案支持当前版本的检查点与轨迹数据。

## 机制与实现约定

1. **确定性历史回退**
   训练和部署统一使用 `rho_deep = 0.0` 与 `rho_shallow = 0.0`。模型直接从真实历史状态回退，不混入新的高斯噪声。该设计消除了随机换噪导致的非编辑区域构图漂移。该设计不构成非编辑区域绝对不变的数学保证，但在采样中保留了原轨迹已建立的大部分主体结构。

2. **训练样本与条件解耦**
   Tier B 样本使用同种子的原生完稿图像作为强编辑器的输入源图，学生模型在训练中只读取原生前向保存的草图。外部编辑指令只用于离线生成目标图，不进入部署条件。训练输入条件仅从轨迹记录读取部署条件，不引入外部裁判针对完稿生成的诊断文本。

3. **图像优先的前缀结构**
   草图前缀使用图像优先的标记顺序，格式为 `<img>` 视觉标记后接条件文本。全部原生权重保持冻结状态。模型在生成器前 36 层添加秩为 64 的 LoRA 适配器，冻结最后 6 层生成层。起点时间嵌入与当前提案投影层均初始化为零。

4. **当前原生提案注入**
   原生生成器在当前时间步预测的干净图像外推结果，经图像编码后注入适配支。该提案输入在计算中截断梯度，实际采样状态链路保持可微。注入额外图像编码会产生单步图像特征提取开销。

5. **Stage 1 残差速度预训练**
   Stage 1 为残差速度预训练阶段，残差在窗口内不衰减，使用指向编辑目标的双点线性桥。该阶段检查点仅用于初始化，不可直接用于部署推理。

6. **Stage 2 混合网格展开监督**
   Stage 2 在前向过程中完整运行部署使用的 Heun 求解网格，终点步采用 Euler 法。反向传播时，纠偏窗口展开为 5 步 Euler，原生尾段展开为 3 步 Euler，以此提供代理梯度。代理梯度与细网格精确梯度存在离散化偏差。

7. **分区域多层结构与保持损失**
   结构损失提取冻结理解解码器第 14 层、第 28 层与第 42 层的图像标记特征距离，结合 4 倍下采样的 Charbonnier 损失。非目标区域约束图像特征与像素 L1 距离。模型还在粗网格窗口状态上计算低权重残差速度惩罚。画布填充区域单独计算损失，不稀释内容区域权重。

8. **离线独立验收**
   训练损失不包含理解模块单词概率的语义反传项，改由外部视觉语言模型进行离线验收打分。

上游视觉模块不保存中间隐层状态。结构损失提取冻结理解解码器的多层图像标记特征。该操作会增加理解模块的前向与反向显存开销，计算资源预算不能直接套用早期阶段的估计值。

## 代码与配置

核心代码位于 `qwen_latent_cot/u15_target_correction/` 目录下：
- `model.py`：条件前缀构建、适配层与速度残差网络。
- `field.py`：精确前向积分、粗网格代理反向回传与配置验证。
- `loss.py`：多层特征结构损失、非编辑区域保持损失与填充区域加权。
- `data.py`：Tier A、Tier B 和 Zero 样本加载器与确定性采样器。
- `train.py`：两阶段训练流程、权重指数移动平均与断点恢复。
- `sampling.py`：文生图完整推理闭环与原生轨迹记录。
- `judge.py`：外部视觉语言模型评分接口。
- `cli.py`：命令行工具入口。

训练配置文件位于 `configs/u15_proposal_correction/` 目录下：
- `target_stage1.json`：Stage 1 配置，训练 6000 步，全局批大小为 256，学习率为 0.0001。数据采样比例为 Tier A 占 80%，Tier B 占 15%，Zero 占 5%。
- `target_stage2.json`：Stage 2 配置，训练 4000 步，全局批大小为 128，学习率为 0.00003。数据采样比例调整为 Tier A 占 20%，Tier B 占 70%，Zero 占 10%。

Stage 2 中每个样本都完整展开至成图阶段。前向沿部署细网格积分，反向沿 5 步纠偏窗口和 3 步尾段回传梯度。Tier A 样本的起始状态基于双点线性桥构造。模型每 250 步保存一次检查点。系统采用底座与适配器全卡复制的分布式策略，各进程独立计算前反向并同步累积梯度。数据采样的随机数状态完整保存在采样器检查点中。

## 数据接口规范

语料索引文件为 JSON 格式的清单文件，顶层键为 `records`。列表内每条记录必须包含 `uid`、`family`、`split`、`approved` 和 `tier` 字段。其中 `split` 取值为 `train`、`val` 或 `test`，`approved` 必须为 `true`。

### Tier A 外部单轮编辑样本

```json
{
  "uid": "a1",
  "family": "source1",
  "split": "train",
  "approved": true,
  "tier": "A",
  "source_image": "src.png",
  "target_image": "edit.png",
  "raw_prompt": "完整目标场景描述",
  "instruction": "结构编辑指令"
}
```

源图和目标图放置在同一适配画布内，不进行中心裁剪。黑色填充区域单独计算损失。此阶段将源图作为参考上下文，不假设与真实前瞻草图同分布。

### Tier B 真实失败轨迹样本

```json
{
  "uid": "b1",
  "family": "prompt1",
  "split": "train",
  "approved": true,
  "tier": "B",
  "trace": "trace/00000.pt",
  "arm": "deep",
  "target_image": "edited_native_final.png",
  "raw_prompt": "原始生成提示词",
  "target_edit_instruction": "仅用于离线生产目标的强编辑要求"
}
```

轨迹文件必须由本系统的 `trace` 命令生成，记录底座标识、分类自由引导缩放因子、时间网格与回退协议。目标图像分辨率必须与原生生成画布大小一致。前瞻草图、历史状态、原生完稿与部署条件均直接读取轨迹文件，不从编辑目标反演。

### Zero 零残差保持样本

Zero 行格式与 Tier B 相同，`tier` 设置为 `"zero"`，不需要 `target_image` 字段。系统自动将目标设置为轨迹中记录的原生完稿 `final`。该类别仅收录经外部确认原生完稿正确的样本，用于训练模型在无需编辑时保持零输出。

评测集合所属的提示词组不可进入训练划分。

## 运行命令

以下命令在包含本仓库代码与依赖环境的机器上运行。设置 `MODEL`、`SOURCE`、`OUT`、`CORPUS` 和 `PROMPTS` 环境变量。

### 生成原生轨迹

```bash
python -m qwen_latent_cot.u15_target_correction.cli trace \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_proposal_correction/target_stage1.json \
  --prompts "$PROMPTS" --output "$TRACE_OUT" --device npu:0
```

### Stage 1 残差速度预训练

```bash
python -m qwen_latent_cot.u15_target_correction.cli train \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_proposal_correction/target_stage1.json \
  --corpus "$CORPUS" --output "$STAGE1_OUT" --device npu:0
```

### Stage 2 轨迹展开端到端训练

```bash
python -m qwen_latent_cot.u15_target_correction.cli train \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_proposal_correction/target_stage2.json \
  --init-checkpoint "$STAGE1_CHECKPOINT" \
  --corpus "$CORPUS" --output "$STAGE2_OUT" --device npu:0
```

### 部署采样

```bash
python -m qwen_latent_cot.u15_target_correction.cli sample \
  --model "$MODEL" --source "$SOURCE" \
  --config configs/u15_proposal_correction/target_stage2.json \
  --checkpoint "$STAGE2_CHECKPOINT" --ema \
  --prompts "$PROMPTS" --output "$OUT" --device npu:0 --compare-native
```

多卡分布式运行追加 `--distributed` 参数。`--resume` 用于同一训练阶段、同一配置与同一语料的精确断点续训。`--init-checkpoint` 仅载入指数移动平均权重，优化器与采样器从头初始化。

## 外部裁判验收

```bash
python -m qwen_latent_cot.u15_target_correction.cli judge \
  --requests "$REQUESTS" --output "$JUDGE_OUT" \
  --base-url "$GLM_BASE_URL" --model "$GLM_MODEL" --key-env GLM_API_KEY
```

请求文件为 JSONL 格式，每行包含 `index`、`uid`、`instruction` 和 `images` 列表。每个图像项包含 `role` 与 `path`。系统读取环境变量获取 API 密钥，并将评审结果保存为对应编号的 JSON 文件。
