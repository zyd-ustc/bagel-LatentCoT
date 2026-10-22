# GEN-hidden caption：已恢复的历史接口与本次实现

## 历史来源

已从Git历史恢复`e539415`（2026-10-13）的相关实现：

- `diagnostics/runtime.py::visual_source`在一次条件GEN前向中capture每层源，返回完整层序列。
- `diagnostics/visual.py::write`用writer slots逐层读取源，形成MemoryBank。
- `diagnostics/runtime.py::caption_prefix`将MemoryBank转为冻结UND的KV上下文。
- `diagnostics/visual_objectives.py::semantic_result`通过固定指令产生free caption；答案未作为目标前缀。
- `diagnostics/source_review.py`有Correct / Source-off / 同prompt Source-swap对照。

这是对话中“GEN内部状态能够支持caption”的可检索代码依据。本次没有加载历史BAGEL caption权重：模型维度、层数和专家结构不同，且尚未定位到对应训练权重和逐题验收文件。不能把结构迁移写成已继承了其语义分数。

## 本次U1.5接口

`Runtime.predict(capture=True)`采集每层GEN `input_layernorm_mot_gen`之前的空间hidden。完整层序列传给UND reader；不是只取最后一层。

UND保留全空间网格，不压缩成历史16个writer slots。每个UND层读取相应GEN层源与当前空间query，使用原生T/H/W位置与视觉双向、文字因果mask。源码中reader-only LoRA扩展只在显式`und_lora`参数存在时工作；原生文本cache和GEN不启用它。

caption序列为图像源/空间query前缀、固定描述指令、teacher-forced答案。前缀不读未来答案。正式纠错不生成caption文本；空间状态交给目标文本cross-attention和单一RGB输出头。

这是迁移后的空间reader，不是历史MemoryBank接口的逐字节复刻。保留的核心是逐层GEN来源、UND语义读出、固定caption指令与来源对照。它需要先在新底座训练caption adapter，再供纠错阶段继承。

实际语义验收尚需完整U1.5权重和自然图文对。本次小模型测试验证计算、梯度、mask与续跑，不代表学会caption或语义纠错。
