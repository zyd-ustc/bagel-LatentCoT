# BAGEL-LatentCoT

在冻结BAGEL上训练独立Memory，使压缩后的caption条件可被理解与生成两侧读取。

Memory以固定中性文本初始化，经过连续UND层更新，并在每层加入可训练的低秩残差。训练先使用Caption语义读出，再加入GEN注意力响应蒸馏。原prompt和全部BAGEL参数保留；只更新新增残差模块。

文本Memory人工审查：累计1100步响应训练后，在原4张图、t=0.9的局部预测对照中，**G-TC与G-text效果基本持平**。G-TC使用16槽文本Memory，G-text使用原生caption条件。这是所测样例的定性结果，未进行最终采样画质评测。视觉Memory来源对照见[实验记录](docs/experiments.md#12-c2-observe及固定来源对照)。

## 文档

- [设计](docs/design.md)：连续Memory、中性初始化、低秩残差及两种已验证的文本训练方法及视觉反馈实现。
- [推理](docs/inference.md)：来源捕获、连续UND写入、GEN消费、R1/R2循环、CFG及完整采样；含16卡命令和代码索引。
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

在modelarts-job的NPU分支仓库运行。默认数据为 `datasets/visual_memory_full.jsonl`：CoRT一致性数据的终态图像与文本配对，保留来源split和匹配标注；该来源标注不等同于新增人工caption审查。配置分别位于 `configs/visual_memory_pilot.json`、`configs/visual_gen_joint.json`、`configs/memory_loop_r2.json`。

`scripts/prepare_visual_data.py`构建全量JSONL及带哈希的summary，不复制图片、不截断caption。只取终态匹配图像，按图片内容去重，并排除此前16张GEN审查图。未生成事实标注，`target_facts=[]`，不报告虚构的事实准确率。学生GEN prompt固定为 `An image.`。

```bash
# E0：复用已有4图，仅检查数值、梯度与冻结合同；单NPU。
VISUAL_DATA=/data/zyd_workspace/bagel-LatentCoT/datasets/memory_gen_recheck4.jsonl \
  bash scripts/visual_training_npu.sh check /root/outputs/visual_e0_new --split test --max-samples 2

# 首次构建全量清单，已有文件不覆盖。
/root/venvs/bagel-NPU/bin/python scripts/prepare_visual_data.py

# 16卡联合训练的20步资源检查。
bash scripts/visual_training_npu.sh observe /root/outputs/visual_observe_check --steps 20 --skip-validation

# 正式E1；全量训练池，默认1000步、global batch16、25%文本回放。
bash scripts/visual_training_npu.sh observe /root/outputs/visual_observe_full

# E2/E3必须从前一阶段选出的视觉checkpoint开始。
VISUAL_PARENT=/root/outputs/visual_observe_full/best.pt \
  bash scripts/visual_training_npu.sh joint /root/outputs/visual_joint_full
VISUAL_PARENT=/root/outputs/visual_joint_full/best.pt \
  bash scripts/visual_training_npu.sh loop /root/outputs/visual_loop_full
```

`--steps`控制更新次数，不缩小训练池。训练期验证默认从val中固定选64个场景族，记录于 `validation_selection.json`；`--validation-max-families 0`使用全量val。独立 `eval`仍可读取全量split，或用 `--max-samples`控制评测数量。`VISUAL_RESUME`恢复同阶段的优化器与全局采样顺序。`eval`默认16卡分配场景族，支持 `--images`、`--rounds 2` 和 `--layer-diagnostics`；`sample`独立运行Base/R1/R2完整采样。`label-states`只导出待审查观察候选，不自动确认标签。

视觉响应教师是独立冻结的1100步C1 writer，压缩为16槽；师生探针与真实消费均预留16位置。观察标签必须绑定实际t、noise seed和状态SHA256，未审查代理不进入确认指标。新checkpoint保存writer、视觉模块、冻结教师、优化器、随机状态和数据哈希；旧C1/C2接口保留。

`source-review`只比较正确、关闭、交换视觉来源的Memory caption。固定状态分配给16个进程，不执行GEN注入或训练。运行后用 `python scripts/render_source_review.py <run目录>`生成盲审页；组别可揭示，标注可导出。原图与原生预测预览分别标明，预览不当作hidden真值。

## 许可

代码采用[Apache-2.0](LICENSE)。模型权重与外部数据使用各自许可证。
