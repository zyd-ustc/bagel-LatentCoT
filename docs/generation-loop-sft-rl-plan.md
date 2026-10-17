# 理解—生成交互：SFT → Flow-GRPO 首轮训练方案

日期：2026-10-17。数据约束：仅使用 caption-image 配对作为外部训练数据。交互 SFT / RL 已有独立训练入口；单卡 2 步与 16 卡 20 步 SFT 已执行，运行记录见 [Experiment 第 7 节](experiment.md#7-交互生成训练)。本文替代 `generation-training-plan.md` 的首轮参数范围与训练顺序。memory 是交互载体，方法目标是理解反馈改善下一次生成。

## 1. 决策

从已完成 Observe 的明确 checkpoint 开始，保留原生 GEN → UND → GEN 的同状态交互。先用配对图文做 **500 optimizer steps 的交互 SFT**，同时训练 UND 写入侧 LoRA 和 GEN 消费侧 LoRA；随后冻结 UND 写入侧，使用终图奖励训练 GEN 消费侧的 Flow-GRPO。学生推理保留内部特征交互，不增加图像解码—视觉编码环节。训练时可解码生成提案供超大多模态理解 teacher 标注；teacher 不进入部署路径。不增加注意力蒸馏，不训练长链，不等待全部 benchmark 完成。

主线为 Observe 初始化 → 连续交互生成 SFT → Flow-GRPO。无需先训练文字反馈格式或证明 memory 可解码为正确修改指令。已有 C2 可作为对照初始化；启动主线优先使用已完成多噪声 Observe 的明确 checkpoint，并记录其实际训练噪声范围。文字反思与超大多模态 teacher 仅为可选诊断或后续有证据支持的辅助训练，不默认插入 CE 更新。

Observe 提供读取生成特征的初始化；它不保证获得与 ViT 输入路径等价的视觉能力，也不保证能生成正确文字反馈。连续反馈无需具有预先规定的文字语义。UND 文本训练是初始化手段。SFT 直接训练“理解信息如何改变生成预测”，RL 再根据真实采样终图优化修正行为。SFT 不承诺已解决真实采样轨迹的分布偏移。

此前五组推理对照已验证的 C2 parent 是 modelarts-job 上 `/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt`，SHA256 `cb8479c5671c01ea07dc5b88a5a7c78577aacd3aa3141974b462eb832f73750f`。此信息来自前轮部署记录，本轮未重新查询远端。该文件是现有 C2 对照，不自动作为新的 Observe 主线 parent。启动时在选定 Observe 实验目录内解析最高编号完整 checkpoint，并核实噪声范围、记录实际路径；不能把训练日志中的未保存步数当作 checkpoint。

## 2. Correction Training：核对到代码的事实

审阅官方仓库 commit `475e21464485c8c3b38730131d071042ad6481f1`，本地只读副本 `/tmp/lcot-correction-training-20261017`。

- [`src/models/denoiser.py`](https://github.com/sunovivid/correction-training/blob/475e21464485c8c3b38730131d071042ad6481f1/src/models/denoiser.py)：corrector 从 denoiser 深拷贝初始化；`losses` 对初始和修正后的速度分别计算同一 target 的 MSE。默认 base 和 corrector 各自获得训练，不是只训练 frozen-base corrector。
- [`src/models/correction.py`](https://github.com/sunovivid/correction-training/blob/475e21464485c8c3b38730131d071042ad6481f1/src/models/correction.py)：`refine` 固定 z/t，默认输入 clean prediction；每轮将修正结果转换为下一轮输入。`detach_input` 与 `detach_skip` 分别控制输入和残差支路的梯度。
- [`configs/train/sit.yml`](https://github.com/sunovivid/correction-training/blob/475e21464485c8c3b38730131d071042ad6481f1/configs/train/sit.yml)：SiT 输入 x、输出 v、标量 gate 从 0 开始，blend 组合，训练一次修正，base/correction loss 权重都是 1。不能把论文概念图的 x-MSE 当成所有公开配置的实际损失。
- [`src/samplers/sit.py`](https://github.com/sunovivid/correction-training/blob/475e21464485c8c3b38730131d071042ad6481f1/src/samplers/sit.py)：先修正，再推进采样；SiT 内部时间从噪声向数据递增，但 correction interval 用 noise fraction。移植必须转换方向。
- [`README`](https://github.com/sunovivid/correction-training/blob/475e21464485c8c3b38730131d071042ad6481f1/README.md)：当前公开 JiT/SiT/SiT-REPA；DiT 与 T2I 代码尚待发布。公开 `validate_guidance` 路径限制 correction 与 CFG 的组合，不能直接当成已实现的 BAGEL CFG 训练。

关键区别：其训练输入是“配对加噪状态经过当前 denoiser 得到的预测”，不是完整自由采样 rollout；其监督来自原始正确目标，不需要逐时刻文字纠错标签。我们借用该训练任务，不声称复现其网络或输出空间迭代。

### 2.1 独立 corrector 与本项目交互路径的对应关系

Correction Training 默认额外训练一个独立 corrector，由 denoiser 复制初始化。以下以其 SiT 配置说明，统一转换为本项目的时间约定：t=1 为噪声，t=0 为数据；P 是条件，x_t 是当前 latent。所有速度都采用 epsilon-x0 的方向。

\[
v_{\mathrm{base}}=D(x_t,t,P),\qquad
\hat{x}_0=x_t-t\,v_{\mathrm{base}}
\]

corrector 读取预测的干净样本 latent，而非原始带噪图像；SiT 配置输出完整候选速度：

\[
v_C=C(\operatorname{sg}(\hat{x}_0),t,P)
\]

使用可学习标量 g 混合，其中 sg 表示停止梯度：

\[
v_{\mathrm{corrected}}
=(1-g)\operatorname{sg}(v_{\mathrm{base}})+g\,v_C
=\operatorname{sg}(v_{\mathrm{base}})
 +g\bigl(v_C-\operatorname{sg}(v_{\mathrm{base}})\bigr).
\]

这是该 SiT 配置的 blend 形式，不代表所有模型的 correction 都使用门控。它的 corrector 输出 v_C，不是直接输出速度差。默认基础预测与修正预测分别对相同目标做生成监督；修正损失停止于基础预测，基础网络仍接受自己的 denoising loss。推理多次修正时，会根据修正结果重建下一次 corrector 输入。

本项目的功能对应如下；这是设计映射，不是已验证的算法等价关系：

| 组成 | Correction Training 默认实现 | 本项目首轮方案 |
|---|---|---|
| 初始提案 | denoiser 的 clean prediction | 第一遍 GEN 的速度与内部特征 |
| 修正输入 | 预测的 clean sample，加时间与原条件 | 当前 GEN 特征、caption 与同一 x_t/t |
| 修正计算 | 独立 corrector 网络 | UND 写入连续反馈，再由第二遍 GEN 消费 |
| 新训练参数 | 独立 corrector；默认也更新 denoiser | writer UND LoRA 与 consumer GEN LoRA |
| 第一遍来源 | 由自身 denoising loss 更新 | 首轮固定，来源停止梯度 |
| 生成监督 | 修正后的速度/样本对齐正确目标 | v_loop 对齐配对 flow target |
| 首轮门控 | SiT 配置有零初始化 gate | 不新增 gate，直接使用 v_loop |

**UND＋第二遍 GEN 共同承担 corrector 的功能。** memory 承载二者之间的连续信息，不是一个必须先生成正确文字的独立教师。修正目标将同时塑造 UND 的表示和 GEN 的使用方式，无需再复制一个完整 BAGEL corrector 网络。

### 2.2 速度差、显式残差与修正训练是三个不同概念

定义 `delta_v = v_loop - v_base` 只描述预测变化，任何两次速度输出都可计算该差值。它本身不能证明发生了有效纠错，也没有新增可训练机制。

本方案的可训练机制是 GEN→UND→GEN 交互，训练目标是交互后的生成误差。第二遍 GEN 输出完整速度 v_loop；delta_v 首轮用于分析修正幅度、方向和终图效果。只有另行增加残差预测头或混合门控，才改变网络参数化或部署策略。

交互与无交互的终图比较、反馈屏蔽/交换，以及相近计算成本的生成基线，负责验证该机制是否提供收益。首轮结果出来前，不将“增加计算”“速度改变”或“文字更流畅”写成成功纠错的证据。

## 3. 本仓库可以复用什么，缺什么

| 代码 | 已有能力 | 首轮处理 |
|---|---|---|
| `diagnostics/flow_training.py:conditional_gen` | 解开 native no_grad，保留原生时间/位置/输出头；复用推理注入；重计算闭包自持上下文 | 保留，增加调用范围内的 adapter 开关 |
| `diagnostics/loop_training.py:memory_loop` | 首次 GEN 无梯度；UND 写入；第二次 GEN 可微；可读取上一轮新特征 | 首轮固定一次反馈，即两次 GEN，避免多轮训练 |
| `diagnostics/memory.py:MemoryWriter` | UND 全深度读同层 GEN KV；输出可微 bank | 保留原始 slot/位置；仅在 writer 内启用 UND adapter |
| `diagnostics/visual.py:VisualStudent` | 登记 C2 UND 原始参数并转 float32 | 不能直接沿用为新 optimizer；新增独立交互 student，冻结 C2 原始参数 |
| `diagnostics/visual_objectives.py` | C2 提前返回文字 CE；通用 loop 分支已有 flow MSE | 新阶段直接使用 generation flow，避免误走 C2 分支 |
| `diagnostics/runtime.py:training_state` | 从正确目标图构造 x_t、epsilon 和配对 flow target | 复用；其 no_grad 与冻结 prompt 编码符合首轮范围 |
| `diagnostics/corpus.py` | Generation 视图 role=2 | 新增 caption-image 轻量读取入口；旧视图仅作导出来源，经图文质量审核后使用 |
| `diagnostics/curriculum.py` | Generation 已支持高噪声；现有采样为 60/20/20 三段 | 新阶段采样部署启用的前 20 个节点，独立配置，不修改 C2 规则 |
| `diagnostics/distributed.py` | global loss 除 global batch，再 SUM 梯度 | 可复用；不能再额外除 world size |
| `diagnostics/visual_record.py` | format 4 仅保存 UND 与旧训练状态 | 新增交互 checkpoint，保存两组 adapter、parent 和 optimizer；不能冒充 format 4 |
| `diagnostics/shallow_sampling.py` | 现有 R2 同状态生成；实际 CFG、early20 | 训练后统一推理入口加载 adapter；保持作用范围一致 |

细节：writer 的 `t` 当前作为来源元数据传递，没有额外的独立时间嵌入。它通过含时间条件的 GEN 特征接收状态信息；首轮不增加 time embedding。

当前循环每次重置 GEN 入口和中性 writer seed，后续理解读取更新后的 GEN 特征。它不是 Correction 项目中 clean prediction 的反复回灌。首轮只训练一次理解反馈，不把更深循环的收益预先写入结论。

## 4. SFT：最小可训练交互

### 4.0 连续修正的语义与可选参数化

本方案将 UND 输出解释为针对当前生成提案的连续计算结果，不假设它是压缩的正确文字反馈。生成目标同时塑造这份表示与 GEN 的使用方式；有用性由修正后的终图判断。可解码的文字仅是探针，不是有效生成修正的必要条件。

ViT 提供的视觉语义输入分布与当前 GEN KV 不同。Observe 结果支持新路径已学到部分读取能力，但不能据此认定它完整继承原生图像理解能力。直接进入生成训练可以让该路径学会生成所需的信息变换，无需先解决完整文字描述或编辑指令生成。这里不引入额外外部视觉证据，收益可能来自语义重组、跨专家计算和可学习的迭代细化。

首轮仍用完整修正速度 `v_loop` 对配对 flow target 监督，定义 `delta_v = v_loop - stopgrad(v_base)` 用于诊断；此定义本身不需要新增残差网络。也可以将同一目标写成 `delta_v` 对齐 `(epsilon-x0)-stopgrad(v_base)`，在相同停止梯度和权重下两者等价，不因改名而获得额外训练收益。

若直接交互使质量明显退化，再独立比较显式混合 `v_used = v_base + alpha*(v_loop-v_base)`，以控制修正幅度。它改变初始化与采样策略；alpha=0 时首步不会给交互分支有效梯度，因此不默认叠加零 gate。先用当前接口开跑，根据实测决定是否需要该控制。

“多阶段噪声 Observe”需按实际 checkpoint 配置确认。之前审阅的 C2 文档仍是 t=0.05–0.25；若最新 Observe 已扩展，应更新 parent 记录，而不是据旧配置否定进展。无论 Observe 覆盖如何，生成 SFT 都直接覆盖部署启用的时间节点。

### 4.1 固定输入与计算路径

令 P 为完整目标文字，x0 为正确目标图的 latent，epsilon 为独立高斯噪声；t 从 1（噪声）降到 0（数据）。

```
x_t = (1-t) * x0 + t * epsilon
(v_base, H_base) = GEN_base(x_t, t, P)       # no_grad，来源 adapter 关闭
M = UND_writer(stopgrad(H_base), P)          # UND adapter 开启，有梯度
v_loop = GEN_consumer(x_t, t, P, M)           # GEN adapter 开启，有梯度
L = mean((v_loop - (epsilon-x0))**2)
```

梯度进入 UND adapter 和 GEN adapter，停止于第一遍 GEN 来源与冻结 prompt cache。不能 detach M，也不能把 UND 写入放进 no_grad。

这直接监督交互后的生成预测。对应 clean proposal 为 `xhat0 = x_t - t*v_loop`；同一 t 下 clean MSE 等于速度 MSE 乘 t²。首轮沿用 BAGEL 的速度损失，避免无意改变时间加权。

目标图只用于构造训练状态和损失，不以干净目标特征注入 UND。新阶段不读取编辑指令或 A→B 数据结构；任何旧数据仅在提取出独立、完整且正确的 caption-image 对并通过审核后使用。

### 4.2 可训练参数与执行范围

加载选定 Observe（或对照 C2）原始 UND 权重后冻结所有原始权重；增加两组 rank=16、alpha=16、dropout=0 的 LoRA，B 矩阵零初始化：

| adapter | 模块 | 何时启用 | 学习率 |
|---|---|---|---|
| UND writer | 28 层 `self_attn.{q,k,v,o}_proj` | 仅 `MemoryWriter.forward` 内，包括导出 bank 的投影 | 2e-6 |
| GEN consumer | 28 层 `self_attn.{q,k,v,o}_proj_moe_gen` | 仅第二遍条件 GEN | 1e-5 |

冻结 prompt 编码、来源 GEN、无条件 CFG 分支、VAE、MLP、norm、embedding 和速度头。这样 UND 的生成适配不会顺带改变 prompt 编码，第一遍来源也不会在首轮漂移。学习率是首轮工程起点，没有最优性证据。

adapter 的作用范围必须封装在 forward/recompute 闭包内；不能只在外部临时打开，导致激活重计算使用错误权重。共享 decoder 下，模块包装不得破坏 `weight`、原生 dtype 和已有 hook 契约。

零 adapter 应复现选定 parent 的同配置交互推理，而非原版 BAGEL。首轮不加 Correction 的零 gate：那会将系统起点切换为另一条策略，并在初始更新屏蔽部分路径梯度。

### 4.3 数据与预算

- 只读取第 4.5 节规定的 caption-image JSONL。现有 corpus 可作为候选导出来源，但 `enabled`、可读、来源名称都不能替代质量审核。
- 首轮准备 8k–16k 对经过筛选的高质量图文；数量不足时先用已审核部分开跑，不补入低质量对。保留原 family 划分，排除正式 benchmark 重合和跨集合近重复。
- 512 分辨率；16 卡；每卡 microbatch=1；global batch=16；500 optimizer steps，即 8000 次样本呈现，不保证 8000 个唯一图像。
- t 均匀抽取 native shift=3、50 节点下的前 20 个更新节点，含 t=1。对应部署 early20；不对高噪声追加“当前画面已满足目标”的文字标签。
- 生成更新的 loss 仅 flow=1；attention/响应蒸馏/速度 teacher=0。首轮不做反思 CE；反思格式与大模型 teacher 作为可选扩展，见第 4.6 节，不把 teacher CE 误写成旧 `teacher` 响应损失。BF16 forward，FP32 trainable 参数与 optimizer，激活重计算，gradient clip=1，前 20 步 warmup。
- 第 100/250/500 步保存。用经过同样审核的 caption-image val 样本进行生成检查和评分，不另造固定 prompt 集。正式 benchmark 在预定阶段出口运行，不用其测试题制作奖励数据或调超参。

SFT 用无 CFG 的条件速度目标；部署保留已有 CFG=4。这与常规条件生成训练一致，但仍须检查完整 CFG 采样表现。不要把 CFG 放大后的速度强行拟合同一个无条件配对 flow target。

### 4.4 转 RL 的判断

前 20 步确认梯度、显存、吞吐、保存恢复正确；第 100 步检查终图是否明显退化；第 500 步结束首轮。

只要采样质量可用、奖励能区分同组候选，就可以转 RL，不要求 SFT 已显著提分。若质量明显退化或输出失控，先修复作用范围/数值/学习率，再启动昂贵 rollout。flow loss 下降本身不构成生成提点证据。

若正确反馈与屏蔽/交换暂时没有差异，也不无限延长 SFT；保留有限 RL 试验，并以同预算无交互训练评估实际贡献。

### 4.5 唯一外部数据契约与质量要求

外部训练数据采用 JSONL，每行仅需两个字段：

```json
{"caption": "Three red apples sit on a white plate on a wooden table.", "image": "/absolute/path/to/image.png"}
```

`caption` 是图中实际内容的完整描述，也是生成目标；`image` 是对应的正确图像。按 train/val/test 分文件保存。来源、family、内容哈希与筛选记录放在独立索引中，不要求数据提供者提交反思、编辑链、负例图、轨迹或原子问题。它们如有需要，由训练系统从配对数据派生。

“只使用 caption-image”指唯一外部数据输入；运行中产生的 latent、提案图、teacher 输出和奖励缓存属于训练产物，不是第二套外部数据要求。RL 每次只取其中 caption 生成候选，配对 image 用于数据审核，不要求候选复制参考图布局。

质量优先于规模。筛选按以下顺序执行：

1. **资产有效性。** 图片可完整解码；检测异常尺寸、损坏、重复、水印和严重压缩、拼图、明显结构伪影。默认要求短边至少 512 像素，例外记录在审核索引，不静默放宽。
2. **图文一致性。** 校验主体、数量、属性绑定、空间关系、可见文字和显著背景。caption 中每项明确断言必须得到图像支持；不要求 caption 枚举所有细节。
3. **训练目标完整。** 丢弃“再多一个”“改成蓝色”等依赖外部上下文的编辑文字，以及无法核实的隐藏属性。不能把指令原文直接当成终图 caption。
4. **去重与划分。** 精确/近重复图像和同一原始 family 放在同一 split；排除 benchmark 图片、题目及其已知派生数据。先划分，再产生 teacher 标签与 rollout。
5. **超大多模态 teacher 复核。** 输入 caption 与配对 image，逐项给出 supported/contradicted/uncertain。明确矛盾或关键事实不确定的样本先剔除。只缺少描述时不强制重写；确需修订则保留原文并重新审核。
6. **人工抽查。** 首批按来源与数量/关系/文字等难点分层抽查至少 200 对。记录错误类型与拒绝率。模型打分只是筛选依据，不能宣称自动保证正确。

图像质量与 caption 一致性分开记录。优先保留自然、结构正确、组合要求清晰的样本，不以单一美学分代替真实性和语义正确性。分辨率处理保留长宽比；裁剪若改变数量、关系或文字，必须重审 caption，不能直接沿用。当前 `runtime.encode_image` 是方形 resize，正式数据入口需明确采用保留长宽比的分桶或经审核的裁剪策略；首轮可以先选近方形样本保持 512 管线，避免拉伸造成结构失真。

### 4.6 可选反思诊断：不作为生成训练的前置或默认损失

本节保留可选实施方法。首轮500步仅训练生成目标，不运行以下格式训练和交错 CE。仅当需要检查语义读取能力，或生成对照显示明确的状态理解瓶颈时，再启用本节；超大 teacher 也可用于数据质量审核，无需训练学生输出反思。

反思监督的目标是识别当前生成提案相对 caption 的偏差，并给出可执行的正向修正。采用简短结构化输出，控制长度，不要求长篇推理过程：

```json
{
  "status": "revise",
  "observed": ["Two red apples are clearly visible."],
  "correction": ["Show three separate red apples on the plate."],
  "preserve": ["Keep the white plate and wooden table."]
}
```

`status` 取 `revise`（需要修正）、`hold`（当前可见内容无明确偏差）、`uncertain`（证据不足）。`uncertain` 不能当成 `hold`；未知项不强造事实。高噪声提案允许很多未知项，不强迫 teacher 判定对象数量。每条输出总长度建议不超过 128 tokens。

**A. 格式训练。** 先从同一 caption-image 池派生约 256–512 个短示例，最多 50 个 CE optimizer steps；只训练 writer UND adapter，GEN adapter 关闭。先覆盖 JSON 格式、三个状态及简短修正语法。可使用人工审核的格式示例，但不能给不可见的高噪声提案编造语义标签。若已有 C2 已满足格式要求，可跳过或缩短。格式通过率目标为独立小验证集至少 99%；这只证明格式稳定，不代表反思内容正确。

**B. Teacher 内容监督。** 从已筛选配对池取 caption，使用当前学生实际第一遍 GEN 的 `xhat0 = x_t - t*v_base`，经冻结 VAE 解码为提案图。teacher 输入为“目标 caption + 当前提案图 + 固定输出规范”。用于反思标注时不给 teacher 原始目标 image，避免把参考图特有布局、纹理或非 caption 要求当成生成错误。

- 配对 SFT 状态与从噪声出发的真实 rollout 状态都可生成 teacher 请求，缓存时区分来源。早期节点优先，保留少量中后期提案用于检查能力边界。
- 首批约 1000 个提案，再按质量扩到 2k–4k；按 parent、prompt、seed、t、轮次及实际提案内容保存关联。下一轮必须对应更新后的提案，不能复用旧状态的纠错文本。
- teacher 只根据可见证据输出观察、修正和保持项。对模糊提案输出 uncertain；不使用目标 caption 反推“当前已经画了什么”。
- 学生仍读取同一状态的 GEN 特征与 caption。冻结文本读出器，teacher 文本仅作为 UND 输出的 CE 标签。teacher 的反思答案不能作为 GEN 的额外输入条件，否则训练会用到部署时不存在的 teacher 信息。
- teacher 监督 CE 只更新 writer adapter；writer 之外的 UND 读出参数固定。这比同时训练写入与文字读出更有利于检查反馈是否保持可解释性，但仍须用终图验证生成价值。

超大 teacher 与现有 Qwen3-VL-8B 本地 judge 分工不同。前者负责高质量数据审核、反思标注和疑难样本复核；后者可承担高频 RL 奖励。teacher 的具体模型、版本、服务端点、图像分辨率、采样参数与成本上限在接入时记录，本方案不把当前本地 8B 当成“超大 teacher”。其服务尚未确认，不能声称已经部署或调用。

**C. 可选辅助试验。** 500 步主线仅指生成 flow 更新。只有明确决定启用文字辅助对照后，才在 teacher 缓存通过抽查的条件下，每 4 次 flow 更新插入 1 次 writer-only CE 更新，最多约 125 次；主线不等待 teacher，不自动追加文字训练。格式训练独立计数，所有更新计入成本。交错更新避免先训练一个只会反思的模型再长时间等待生成适配，也避免未校准的 CE 权重支配 flow。

现有 `semantic_loss` 使用固定 instruction 和当前 caption prefix，需要改为上述反思 instruction 与 teacher label。teacher 文本不填进基础 `caption` 字段覆盖生成目标；放入派生缓存，并在 CE 更新时显式读取。

CE 内容监督不替代生成损失。真实 rollout 没有配对 epsilon-x0 目标，反思标签只能监督 UND，终图收益通过 RL 优化。不得为自由生成轨迹强配原始 image 构造伪 flow target。

## 5. RL：使用终图奖励优化交互后的动作

参考 Flow-GRPO 官方 BAGEL，审阅 commit `879042cf5707f8b90daa98d147d7deac2317c5da`。

- [`scripts/train_bagel.py`](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_bagel.py)：官方 GEN LoRA 为 rank64/alpha128，含 attention+MLP；本方案 rank16 attention-only 是缩小训练范围，并非原配置复现。
- [`bagel.py:_sde_step_with_logprob`](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/bagel/modeling/bagel/bagel.py)：SDE 计算使用 FP32；均值、方差、采样与 replay 共享函数。当前代码将平方项在所有 latent 元素上取均值，省略常数；首轮 batch=1 时沿用这个归一化代理分数。不要称其为完整联合 log likelihood，也不要在 pack 多个样本后跨样本平均。

### 5.1 首轮参数与策略

冻结 SFT 后的 UND writer adapter，继续训练 GEN consumer adapter。初始 GEN 来源与 prompt 编码仍固定。对给定 x_t/t/P，反馈因此是确定性且与本阶段待更新参数独立的条件，可以按需重算，无需保留 UND 反向图。

动作是下一步随机图像 latent，不是连续 memory。对 R2 得到的条件速度，按部署规则与无条件速度做 CFG/renorm，再使用同一个 SDE 转移函数计算 rollout 与 replay。保存 x_t、x_s、t/s、旧分数、group ID 和条件身份。

- 每 prompt 8 个候选，独立初始噪声及转移噪声。
- 保留 49 次更新和 early20 交互；在 i=1…19 中抽一个随机转移作训练，其余为 ODE。避开 t=1 和 t=0 的转移奇异点。
- 首轮 100 个 rollout batches；每批 7 prompts × 8 candidates = 56 张，合计 5600 张终图。
- 16 卡预算：14 卡 policy workers，2 卡常驻 Qwen3-VL judge。每 worker 每批 4 张；按 group ID 跨 worker 聚合全部 8 个奖励后计算优势。
- 每批 rollout 只更新 1 轮。GEN LR=1e-5，clip=1e-3 为代理分数下的初始值，按 ratio/clip fraction 校准。不能套用联合 logprob 的 clip 标度。
- 参考策略固定为 SFT 阶段出口；旧策略为当前批 rollout 权重。分别管理，不能混用。固定方差 Gaussian 转移 KL 按 latent 维平均计算，首轮系数设 0.01，记录相对 policy loss 尺度；若惩罚完全主导或几乎为零，先做小批尺度检查再固定正式值。

这属于短随机窗口的策略优化代理目标，不是对整条混合 ODE/SDE 轨迹进行精确全路径求导。最终 benchmark 同时保留部署 Euler 设置，不能把采样器改变当成权重提点。

### 5.2 奖励与最短可运行数据路径

训练 prompt 只从已审核 train caption-image 对的 caption 中取，先准备约 1000 条对象、数量、颜色、关系明确的目标。超大 teacher 依据 caption 一次性派生原子 yes/no 问题，检查问题只覆盖 caption 的明确要求，抽查后缓存。原始数据不要求 facts 或 vqa_list 字段；不能直接把 benchmark 的 vqa_list 搬来训练。

复用已部署 Qwen3-VL-8B 与 GenEval2 的逐题概率评分能力，训练奖励使用原子满足概率的算术平均，避免硬全对分在首轮大面积为零。标为训练奖励，不冒充官方 GenEval2 总分。首轮不叠加 CLIP、美学和多个 judge。

对相同 prompt 的 8 个奖励计算归一化优势；记录零方差组比例、奖励分布和图像退化。正式 rollout 前用 4 个训练 prompts × 8 张跑通生成→judge→replay→更新；不能只验证导入成功。

若奖励无区分度，先检查原子问题、评分概率和训练目标难度，不以长 SFT 代替奖励修复。图像质量由 caption-image val 与阶段 benchmark 分开检查，避免满足任务分但产生伪影。

UND 在 SFT 已接受生成监督，RL 冻结它是首轮降低成本和保持条件稳定的选择。只有发现反馈内容成为明确瓶颈，才让 RL 梯度穿过 UND adapter；届时重算反馈且不能 detach M，所需改动另行限定。

## 6. 最短实现顺序与交付条件

先完成 caption-image 质量筛选与 SFT 可运行入口，做 NPU smoke 后启动500步。反思格式训练与 teacher CE 均不在首轮必做项中；teacher 接入与完整 RL 工程不阻塞生成训练开跑。

| 顺序 | 最少改动 | 完成标准 |
|---|---|---|
| 0 | caption-image JSONL 读取、筛选报告与 train/val 划分 | 首批高质量配对可读取，图文审核与人工抽查完成 |
| 1 | 新交互 student + 两组作用范围明确的 LoRA；新 checkpoint 保存/加载 | 零 adapter 与选定 parent 的同配置交互推理数值一致，冻结分支不变，加载恢复一致 |
| 2 | 新 SFT CLI/配置，使用 caption-image 入口，复用 training_state、conditional_gen、memory_loop、distributed、日志 | 正确训练两组 adapter；单卡真实2步，16卡真实20步通过 |
| 3 | 接入统一 InferenceEngine 与现有 benchmark 工具 | 同一 checkpoint 可执行正常/屏蔽/交换反馈，保留 native 和 C2 基线 |
| 可选 | 反思诊断、提案解码和超大 teacher 标注 | 不阻塞 SFT；不默认加入文字 CE |
| 4 | SDE rollout/replay 共用转移函数；冻结参考与旧策略 | 未更新参数时 ratio≈1，单样本/分组语义正确，NPU 梯度有限 |
| 5 | 训练奖励清单 + 2 卡 judge + group aggregation | 32张试运行完成一次真实更新后，启动100批RL |

LoRA 零初始化时，A 矩阵第一步可能没有非零梯度，这是正常现象；检查每组 adapter 有有效更新，不要求每个因子在第一步都非零。

首轮验证只覆盖数值和行为关键点：作用范围、生成梯度能否到 UND/GEN、重计算一致、checkpoint 恢复、训练/推理 R2 一致、SDE replay 一致。已有 UND 测试和推理 smoke 不能替代这些检查。

生成训练入口为 `python -m qwen_latent_cot.interaction.cli sft|rl`，checkpoint 使用 `interaction_lora_v1`。`visual_training_npu.sh` 负责 Observe / Feedback，原 `loop` 模式已移除；新入口和具体启动命令见 Experiment 第 7 节。

## 7. 评估交互贡献与成本

先训练一条主线，不在启动前铺开多组长跑。已有五组 C2 推理对照继续作为起点；SFT/RL 出口用相同推理协议评估。

必须区分：

1. 终图是否提高：原始 C2、SFT、SFT+RL 的预定阶段出口比较。
2. 理解反馈是否贡献：训练后正确反馈、同布局屏蔽、同目标异状态交换。
3. 交互是否值得计算：后续补同预算无交互 GEN-LoRA SFT+RL，以及相近实际推理耗时的无交互基线。不同专家和 token 数下，不能仅以 forward 次数宣称等算力。

当前已测 512px、49更新、单条 GenEval2 prompt 的 C2+memory 纯生成约17.7秒。若简单按14张policy卡线性外推，5600张约1.97小时纯生成；该值不含评分、训练反向、同步、加载和长prompt差异，只能作下限参考。SFT 尚无真实每步耗时，不能从 UND 训练耗时推算；16卡20步后按实际吞吐估计500步预算。

## 8. 本次结论与边界

Correction Training 最值得借鉴的是：让模型看到自己的提案，并通过修正后的生成预测接受监督。我们已有 GEN→UND→GEN 的结构和可微消费路径，缺的是对应训练参数、损失入口与 checkpoint 支持。

首轮优先实现“一次理解反馈的生成 SFT”，再用终图奖励优化真实采样。保留现有内部特征与 memory 接口能最快开始训练；输出空间反馈是后续有针对性的比较，不是首轮开跑的前置条件。
