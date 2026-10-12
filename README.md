# BAGEL-LatentCoT

在冻结BAGEL上训练独立Memory，使压缩后的caption条件可被理解与生成两侧读取。

Memory以固定中性文本初始化，经过连续UND层更新，并在每层加入可训练的低秩残差。训练先使用Caption语义读出，再加入GEN注意力响应蒸馏。原prompt和全部BAGEL参数保留；只更新新增残差模块。

最新人工审查：累计1100步响应训练后，在原4张图、t=0.9的局部预测对照中，**G-TC与G-text效果基本持平**。G-TC使用16槽文本Memory，G-text使用原生caption条件。这是所测样例的定性结果，未进行最终采样画质或视觉Memory评测。

## 文档

- [设计](docs/design.md)：连续Memory、中性初始化、低秩残差及两种已实现的训练方法。
- [实验](docs/experiments.md)：training-free探索、C0/C1、GEN对照与两轮响应训练的真实记录。

## 快速开始

正式Ascend实验使用 [`loop-layer-NPU`](https://github.com/zyd-ustc/bagel-LatentCoT/tree/loop-layer-NPU) 分支，默认16卡。先安装与设备匹配的PyTorch及torch_npu，再在仓库根目录执行：

```bash
python -m pip install -e '.[dev]'
memory-topology --help
```

16卡响应训练入口为 `scripts/c1_response_npu.sh`；扩展训练入口为 `scripts/c1_response_expand_npu.sh`；GEN对照入口为 `scripts/gen_npu.py`。实验使用现有BAGEL权重及已准备的数据清单，路径与checkpoint记录在[实验](docs/experiments.md)中。

## 许可

代码采用[Apache-2.0](LICENSE)。模型权重与外部数据使用各自许可证。
