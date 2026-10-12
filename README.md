# BAGEL-LatentCoT

在冻结 BAGEL 上研究独立 Memory：将图像生成状态压缩为可被理解与生成两侧读取的语义条件。

项目面向 BAGEL 后训练研究，重点比较每层独立更新（L）与连续 UND 更新（C）。原 prompt 保留，原始 BAGEL 参数冻结；新增 Memory 模块负责语义写入与条件表达。

最新方案以 **caption 原生 prefill 为教师**，通过冻结 UND 的语义读出与冻结 GEN 的注意力响应蒸馏对齐 Memory，无需与 caption token 一一对应。推理时使用从 GEN 状态预测的 Memory。

## 当前状态

|内容|状态|
|---|---|
|L/C writer、文本桥、不同噪声 Caption 训练|已实现；当前训练损失为文本交叉熵|
|GEN 的布局、固定、文本、视觉与错配 Memory 对照|已实现；所有参数冻结，单次读取|
|语义读出对齐、GEN 响应蒸馏、语义状态去噪|设计方案，尚未实现|
|工程验证|历史40项小模型测试通过；新增联合训练检查见验证文档|
|真实 BAGEL C0|16卡完成16条；文本语义读取可用，严格匹配7/16，存在扩写|
|真实 BAGEL C1|L/C各100步已完成；保留C，验证字面事实分45.8%，颜色8/8；完整语义仍有错误|
|当前下一步|16图GEN定性读取已完成，224张局部预测图待人工审查|

## 快速开始

在仓库根目录执行。先安装目标设备的 PyTorch；Ascend 需要匹配的 `torch_npu`，CUDA 原生路径需要 `flash-attn`。

```bash
python -m pip install -e '.[dev]'
memory-topology --help
```

当前保留C，L暂停；选择依据与checkpoint见[验证](docs/validation.md#4-c1结果与拓扑选择)，下一步见[实验](docs/experiments.md#当前首跑c的颜色消费)。首次复现从[原生文本校准](docs/run.md#2-原生文本校准)开始。完整命令、核验数据格式和结果解读见下表。

Ascend 实验使用 `loop-layer-NPU`，默认16卡并行。C0按样本分片；GEN按场景对×seed分片，启动器已实现，本轮使用16张异场景图像的定性清单；C1/C2通过 `torchrun --distributed` 同步新增writer梯度，原BAGEL冻结。

## 文档

|入口|内容|
|---|---|
|[设计](docs/design.md)|BAGEL 约束、L/C 拓扑、三条对齐路线与训练阶段|
|[实验](docs/experiments.md)|噪声、标签、消融、GEN 读取对照和验收标准|
|[运行](docs/run.md)|当前代码的安装、校准、训练、评测和汇总命令|
|[实现](docs/implementation.md)|模块映射、数据与输出格式、位置和冻结边界|
|[验证](docs/validation.md)|实际检查范围、历史修正与未验证事项|

建议先读设计与实验，再按运行文档执行。所有文档使用简短文件名；方案、实现和验证状态分别标注，日期保留在正文。

## 分支与许可

`main` 与 `loop-layer-NPU` 共享核心CE诊断；当前联合训练入口与最新实验记录维护在NPU分支。`auto` 在 main 优先 CUDA，在 NPU 分支优先 NPU；显式设备选择优先。历史实现保留在 `older-1/older-2/older-3`。

代码采用 [Apache-2.0](LICENSE)。模型权重与外部数据使用各自许可证。
