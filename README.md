# BAGEL-LatentCoT

在冻结BAGEL上训练独立Memory，使压缩后的caption条件可被理解与生成两侧读取。

Memory以固定中性文本初始化，经过连续UND层更新，并在每层加入可训练的低秩残差。训练先使用Caption语义读出，再加入GEN注意力响应蒸馏。原prompt和全部BAGEL参数保留；只更新新增残差模块。

最新人工审查：累计1100步响应训练后，在原4张图、t=0.9的局部预测对照中，**G-TC与G-text效果基本持平**。G-TC使用16槽文本Memory，G-text使用原生caption条件。这是所测样例的定性结果，未进行最终采样画质或视觉Memory评测。

## 文档

- [设计](docs/design.md)：连续Memory、中性初始化、低秩残差及两种已验证的文本训练方法及视觉反馈实现。
- [实验](docs/experiments.md)：training-free探索、C0/C1、GEN对照与两轮响应训练的真实记录。

## 快速开始

正式Ascend实验使用 [`loop-layer-NPU`](https://github.com/zyd-ustc/bagel-LatentCoT/tree/loop-layer-NPU) 分支，默认16卡。先安装与设备匹配的PyTorch及torch_npu，再在仓库根目录执行：

```bash
python -m pip install -e '.[dev]'
memory-topology --help
```

16卡响应训练入口为 `scripts/c1_response_npu.sh`；扩展训练入口为 `scripts/c1_response_expand_npu.sh`；GEN对照入口为 `scripts/gen_npu.py`。实验使用现有BAGEL权重及已准备的数据清单，路径与checkpoint记录在[实验](docs/experiments.md)中。

## 视觉反馈训练（NPU）

新增阶段：`C2-observe`训练视觉语义读取，`C2-joint`加入响应和完整GEN速度损失，`C3-loop`训练两轮反馈。BAGEL、DiT、VAE及原文本writer默认冻结；只训练感知噪声时间的视觉query残差。代码实现与工程检查不代表视觉语义或最终画质已改善。

在modelarts-job的NPU分支仓库运行。`VISUAL_DATA`必须指向已审查的图像标签清单，不能将现有自然文本prompt当作配对图像caption。配置分别位于 `configs/visual_memory_pilot.json`、`configs/visual_gen_joint.json`、`configs/memory_loop_r2.json`。

```bash
# E0：复用已有4图，仅检查数值、梯度与冻结合同；单NPU。
VISUAL_DATA=/data/zyd_workspace/bagel-LatentCoT/datasets/memory_gen_recheck4.jsonl \
  bash scripts/visual_training_npu.sh check /root/outputs/visual_e0_new --split test --max-samples 2

# 已准备visual_memory640.jsonl后：16卡联合训练的20步资源检查。
VISUAL_DATA=/data/zyd_workspace/bagel-LatentCoT/datasets/visual_memory640.jsonl \
  bash scripts/visual_training_npu.sh observe /root/outputs/visual_observe_check --steps 20 --skip-validation

# 正式E1；默认1000步、global batch16、25%文本回放。
bash scripts/visual_training_npu.sh observe /root/outputs/visual_observe_full

# E2/E3必须从前一阶段选出的视觉checkpoint开始。
VISUAL_PARENT=/root/outputs/visual_observe_full/best.pt \
  bash scripts/visual_training_npu.sh joint /root/outputs/visual_joint_full
VISUAL_PARENT=/root/outputs/visual_joint_full/best.pt \
  bash scripts/visual_training_npu.sh loop /root/outputs/visual_loop_full
```

`VISUAL_RESUME`恢复同阶段的优化器与全局采样顺序。`eval`默认16卡分配场景族，支持 `--images`、`--rounds 2` 和 `--layer-diagnostics`；`sample`独立运行Base/R1/R2完整采样。`label-states`只导出待审查观察候选，不自动确认标签。

视觉响应教师是独立冻结的1100步C1 writer，压缩为16槽；师生探针与真实消费均预留16位置。观察标签必须绑定实际t、noise seed和状态SHA256，未审查代理不进入确认指标。新checkpoint保存writer、视觉模块、冻结教师、优化器、随机状态和数据哈希；旧C1/C2接口保留。

## 许可

代码采用[Apache-2.0](LICENSE)。模型权重与外部数据使用各自许可证。
