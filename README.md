# Stage 1 等时状态编辑

本项目保留 SenseNova-U1.5 的 Stage 1 训练，以及两种推理：源图编辑和 UND 自反馈文生图。目标检查点是 `train_full490k/step_014723`，条件协议为 `isochronic_condition_v2`：编辑器读取中间态和短编辑指令，不读取预览图或目标描述。

## 准备

在已安装 PyTorch 和 `torch_npu` 的 Ascend 环境中执行：

```bash
pip install -e .
```

以下示例在 `modelarts-64` 上执行。`MODEL` 是原生 SenseNova 权重，`SOURCE` 是带 NPU 适配的 SenseNova 源码，`CKPT` 是训练得到的编辑器权重。它们不属于本仓库。

```bash
MODEL=/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT
SOURCE=/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b
CKPT=/data/zyd_workspace/isochronic_data_20261107/train_full490k/step_014723
CONFIG=configs/u15_isochronic_state/stage1_full490k.json
```

## 编辑源图

```bash
u15-target-correction edit --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --image /path/to/source.png --instruction 'Make the dog blue.' \
  --checkpoint-mode trained --checkpoint "$CKPT" --output /path/to/new-output
```

程序按训练方法将源图与噪声线性混合至 `t=0.36`，在固定 `t` 下编辑，然后用无条件原生 T2I 接续至终图。这不是对源图的数值反演。输出目录必须不存在，结果包括 `edited_state.pt`、`edited_preview.png`、`result.png` 和 `result.json`。

若要使用未训练的编辑器作为对照，将最后一行替换为：

```bash
  --checkpoint-mode native --output /path/to/new-output
```

`native` 不加载训练检查点：LoRA 更新和编辑进度嵌入为零初始化，但程序仍执行指令条件编辑。它不是训练后效果，也不保证编辑成功。

## UND 自反馈文生图

```bash
u15-target-correction t2i --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --prompt 'a lion beside seven zebras' \
  --checkpoint-mode trained --checkpoint "$CKPT" --output /path/to/new-output
```

要不加载检查点，同样使用 `--checkpoint-mode native`，并省略 `--checkpoint`。程序先生成 `t=0.36` 的状态和预览，再让 UND 按顺序描述图像、判断是否满足提示词、写出一条修改指令。UND 判定已满足时，程序直接接续；判定需要修改时，程序编辑状态并接续。解析失败时，程序报错而不是静默判定已满足。输出包含 `preview.png`、`result.png` 和记录完整反馈的 `result.json`；发生编辑时还包含编辑后的中间态和预览。

视觉特征已接入 UND 生成路径。UND 仍可能数错物体或误判关系，不应将反馈当作独立真值。

## 训练 Stage 1

全量数据清单为 `/data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json`。模型训练使用 64 卡、全局批次 64、单点 `t=0.36`，最多 14,723 步。下面的启动器从当前 ModelArts 作业的四个节点启动训练：

```bash
python scripts/launch_train64.py \
  --linear /data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json \
  --config "$CONFIG" --output /path/to/new-training-output
```

使用 `--resume /path/to/step` 恢复同一配置、数据清单、步数和 64 卡规模的训练。训练输出、数据、模型和现有检查点均在代码目录之外。
