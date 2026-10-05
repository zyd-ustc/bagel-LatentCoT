# 冻结 BAGEL Memory body loop：当前方案

状态：training-free 实现；真实权重评测待用户执行。唯一架构依据是原生 BAGEL 的 UND/GEN 专家与共享注意力，以及用户已有的旧 Memory loop 行为。这里不增加 DiT/RAE 式逐层循环，也不加入训练架构。

## 目标与证据边界

在 denoiser 内部通过少量理解侧 Memory 更新生成条件，实现有效语义修改并保持文生图质量。一次 denoiser 调用内 `x_t,t` 固定，多个完整 body 运算之后只输出最终 velocity。原生 ViT 仅允许离线诊断；生成路径不使用图像解码/ViT 反馈。

用户已观察到旧早期 Memory loop 有效。此次从该路径恢复，不能将已失败的逐层 `1/N` 实现与旧实现混为一谈。固定 384 prompt×3 seed 确认集不列为待办。

源码支持 GEN Q 读取 UND KV：二者分别使用专家投影，再进入共享注意力。BAGEL 原生 think 路径先生成规划文本，再将它写入上下文以条件化图像生成。不过，训练的 noise mask 阻止其他区段读 noisy image。UND 读取当前 noisy GEN 的有效性仍需图像结果验证，不能仅由接口存在推导。

## 已实现结构

- Memory 与 query 拼为 `[SOI,M,GEN,EOI]`，M 使用原生 UND，GEN 使用原生 GEN。
- M 初值为每个样本边界 embedding 均值加 `1e-4` slot noise；图像采样 seed 与 Memory seed 分开记录。
- prefix 一次，第一轮 Read 禁止所有非 M query 读取 M；这也关闭 SOI/EOI 中继。
- 保存 body 入口。每个额外轮恢复全部非 M hidden，仅回传上一轮 body 末端 M hidden。
- 每层执行完整原生 attention、MLP、norm 和残差。没有 residual damping、adapter、gate 或 α。
- suffix 一次，保留旧路径读取最终 M 的语义；最终原生 norm/llm2vae 输出 velocity。
- M 不跨采样步或 CFG 分支传递。text-removed 分支也从其自身边界状态创建 M。
- R 表示额外 body 轮数。R≥1 与冻结旧 kernel 对齐；R=0 明确定义为不插入 M 的原生 bypass。

BAGEL 保持一个固定 GEN body 入口，因此没有将 GEN 末端 hidden 直接送回入口。但 M 仍有深度不匹配风险：末端 M 被回传到入口。这是旧机制的已知边界。本次不同时引入分层 KV 和单独 writer pass；这两项均未实现。

## 用户执行顺序

1. 在真实 BAGEL 权重上执行 E0：R=0/native、no-read/native、恢复路径/冻结旧路径的 CFG velocity parity，验证 prompt KV 不变和不同长宽比 packed 输入。
2. 执行同 prompt/seed 的 Base / Memory loop training-free 配对生成。默认早期窗口 `[0,8)`，R=1，K=8，progress `[0,1]`（恢复旧路径全部采样步启用）。R=2 使用独立输出目录；不混入训练权重。
3. 评分 GenEval2 atom/GM、quality proxy、invalid、Repair/Damage，并按 prompt 聚类计算区间。查看固定配对图片；quality proxy 不代表人类偏好。
4. 仅在图像质量保持且出现有效语义修改后，再研究 M 内容与编辑的联系。Memory QA 可以作为离线证据，但此前读出有 unknown 和答案偏置，不能仅凭 QA 分数否定 Memory 内容。

不默认执行 shuffle 或额外消融。可选 no-read 只用于数值/实现诊断。所有正式运行由用户启动，代理只执行 CPU 合成测试和接口检查。

## 后续训练条件

本次不实现训练。只有图像结果支持反馈机制后，才考虑理解侧的格式/隐式适配；不得以降低 flow loss 代替有效语义编辑和质量保持的证据。所有状态转换、数据和权重版本必须分别记录。
