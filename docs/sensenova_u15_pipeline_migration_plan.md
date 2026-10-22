# SenseNova-U1.5：纯文生图语义纠错 Pipeline 迁移方案

- 日期：2026-10-22
- 状态：待实施的设计方案。本次只交付文档，不切换底座、不启动训练、不修改正在进行的评测。
- 推荐底座：`sensenova/SenseNova-U1.5-8B-MoT`，首次实验不用8步蒸馏LoRA。
- 参考代码：`/Users/zyd/Documents/LCoT-codex/refs/SenseNova-U1`，已核对本地版本`c35c83b`。
- 当前项目：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU`。

## 1. 要迁移的结果：不是换模型，而是保留研究问题

外部输入只有文本prompt与随机seed。系统先生成，再从自己的GEN内部状态读取语义，用UND产生一次空间纠错，继续原采样轨迹。最终输出图片。

迁移成功需要同时满足：

1. U1.5原生文生图在Ascend上正确运行，成为独立、可复现的基线。
2. GEN hidden可通过UND读取内容；继承过去已成功的caption计算路径，而非仅复制UND权重。
3. 纠错相对同一底座的native与global基线改善语义，且不以明显破坏画质为代价。
4. 训练、采样、检查点、六榜评分和本地HTML形成同一条可追溯流程。

原生底座变强和纠错增益分别报告。不能用U1.5 corrected优于BAGEL corrected，代替“UND纠错有效”的验证。

### 1.1 固定约束

- 不引入外部参考图像，不依赖ViT，不把文生图改成必须提供图像的编辑任务。
- 主线只读取GEN hidden；不将内部图像重新patchify作为默认观察路径。
- 不在GEN各层增加残差注入、Memory门控或多层bridge。
- 保持GEN原生输出职责；暂不实现“把UND状态送回GEN再额外预测一遍新速度”的方案。
- UND corrector只在采样器输出处提供一个增量；不增加外层零门控、raw辅助回归或shuffled对比损失。
- caption用于能力继承阶段；正式纠错阶段默认不叠加caption CE。
- 不把BAGEL的Table 3配方、时间方向、50个节点、CFG renorm或checkpoint直接移植到U1.5。
- BAGEL现有run、checkpoint、评分结果保持独立；本方案不授权停止其他任务或覆盖共享环境。

### 1.2 三个范围不同的选择

| 内容 | 本方案选择 | 原因 |
|---|---|---|
| 研究观察接口 | GEN hidden → UND | 延续已有caption证据，保持纯T2I内部循环 |
| 纠错输出 | 单个RGB速度增量 | 保留当前post-CFG范式，不同时改变所有机制 |
| 图像重读 / GEN重新输出速度 | 不进入首版 | 前者改变观察接口，后者改变已暂缓的输出方案 |

U1.5原生RGB与clean-image头仍有价值：它们提供明确的生成状态、可检查的候选图像和空间重建基线。但本方案不以重新读像素代替GEN-hidden语义继承。

## 2. 底座差异与可复用内容

U1.5采用RGB空间flow matching。所谓`vision_model`是浅层patch embedding，不是BAGEL的预训练SigLIP；UND与GEN各有一套patch embedding。两侧Q/K/V/O、FFN和norm也有独立参数，通过共享attention交互。

U1.5典型每个token对应32×32像素；512px为256个图像token。其dense模型42层、hidden size 4096，与BAGEL当前28层/3584不同。token更少不等于全流程按比例加速。

原生`fm_head`先预测clean RGB，再换算速度。U1.5的卷积/PixelShuffle读出应保留，不能为了迁移方便替换成BAGEL latent线性头。

| 现有模块 | 迁移方式 |
|---|---|
| 图文对清单、train/val划分、数据质量检查 | 复用语义与划分；替换VAE编码为原生RGB预处理 |
| 同seed native/global/corrected对照 | 保留，重新生成U1.5基线 |
| guidance-safe负投影去除 | 保留概念，按RGB速度和U1.5时间重新实现 |
| 全局校准基线 | 按新底座、新分辨率、新时间网格重新估计 |
| 六榜manifest、seed、27B rubric/parser/聚合 | 复用；新arm、新结果目录、新缓存namespace |
| BAGEL UND/LoRA/速度head checkpoint | 不加载到U1.5，结构和表征均不兼容 |
| BAGEL native cache、RoPE、采样器 | 由U1.5原生实现替代 |
| 成功caption实验 | 先恢复其计算契约，再迁移结构并重新训练适配器；不硬搬3584维权重 |

## 3. 总体结构与模块契约

```text
外部：prompt P、seed
          │
          ▼
 U1.5原生文本上下文缓存（冻结）
          │
          ▼
 RGB状态 z_t ── 原生GEN条件前向 ──► v_c、clean预测、H_GEN
     │                   │                         │
     │                   │                         ▼
     │                   │               semantic_reader(H_GEN,t,σ)
     │                   │                         │
     │                   │                    UND空间状态 U
     │                   │                         │
     │                   │            prompt-aware readout(U,文本状态)
     │                   │                         │
     │    原生无条件前向 ──► 原生CFG速度 v_g         Δv_raw
     │                                               │
     │                          guidance-safe投影 ───► Δv
     │                                               │
     └──────── Euler：z_next=z_t+Δt·(v_g+Δv) ◄────────┘
```

这里的“循环”是连续采样步之间的反馈：当前生成状态影响本步UND，修正后的状态进入下一步。首版不在一个时间步内部反复迭代，也不把上一步UND KV永久累积成Memory。

### 3.1 `NativeRuntime`

负责严格加载原生模型、RGB预处理、文本cache、原生噪声尺度、时间网格、位置索引、条件/无条件前向与原生图像输出。

条件前向返回一个结构体：

```text
NativePrediction:
  conditional_velocity: [B,3,H,W], FP32用于目标和投影计算
  clean_prediction:    [B,3,H,W]
  observation:         L个[B,N,D]，实现阶段已改为每层GEN block输入的空间hidden
  image_grid:          (H/32,W/32)，以模型实际配置为准
  spatial_indexes:     原生T/H/W索引
  time, noise_scale:   当前真实采样条件
```

`observation`已根据恢复的历史代码改为逐层观察；具体继承范围和与历史MemoryBank的差异见`docs/u15_npu_20261022/CAPTION_INTERFACE.md`。它不是已训练BAGEL权重的跨维度迁移。不得无意捕获无条件分支的hidden。

只缓存prompt相关状态。每个采样步的图像状态和reader状态重建，避免把旧噪声状态当成当前上下文。

### 3.2 `SemanticReader`：先继承可读语义，再接纠错

先恢复过去caption实验的以下对象，记入`CAPTION_INTERFACE.md`：

- 实验代码版本、checkpoint、训练数据和验证结果位置。
- GEN hidden层号、取值位置、是否多层、是否detach及归一化方式。
- bridge结构、图像token顺序、位置编码、mask、caption query和可见文本。
- 哪些UND参数可训练，哪些参数实际载入了已训练权重。

找不到原checkpoint时可以根据代码重现接口，但必须标为“接口重现、权重未继承”，先完成caption阶段，不能宣称已恢复同一能力。

在U1.5上，reader从本模型预训练UND初始化，并迁移上述计算结构；BAGEL bridge只提供结构经验，不跨维度拷贝权重。新增输入适配器把GEN观察映射到UND输入尺度，并保留完整二维网格；不把空间token平均成一个向量。

caption读取序列采用图像观察前缀、通用描述query、caption文本的结构。图像前缀内部使用原生视觉双向mask，文字使用因果mask。直接复用原生T/H/W位置逻辑，不能套BAGEL的单轴位置或重复旋转已得到的Q/K。

reader默认不直接接收目标caption文字，防止通过文字输入完成caption任务。H_GEN本身已受prompt条件影响，因此还要用同prompt不同生成状态检查是否真正读取了状态，见第7节。

输出`U ∈ R[B,N,4096]`为空间图像token状态。teacher-forced caption文字位于其后，不允许未来caption答案回流到U；正式采样时不生成caption文字。

### 3.3 `CorrectionReadout`：单一空间输出头

输入为U、目标prompt的冻结原生文本状态、t和noise_scale。

- 一次目标文本cross-attention让空间query读取目标条件，放在输出读出内，不插入GEN各层。
- 将空间状态恢复为token网格，经过轻量卷积和PixelShuffle产生`[B,3,H,W]`速度增量。
- 起始结构建议：通道投影4096→256，3×3卷积输出`3×32²`通道，再PixelShuffle(32)。实际patch边长必须来自配置；不是硬编码所有版本为32。
- 最后输出层weight/bias为0，其余层正常初始化。初始增量严格为0；不再叠加零初始化乘法门控。
- head输出保留FP32，在采样器只相加一次。零输出时走原生更新路径，不通过新的BF16重算改变基线。

这是新任务head，不能声称继承了原生`fm_head`的像素重建能力。原生GEN读出保持不变。首版不同时比较多种head设计。

### 3.4 参数与梯度归属

| 参数 | 纯T2I SFT | caption继承 | 正式纠错 |
|---|---|---|---|
| 原生GEN专家及图像生成输入/输出 | 训练选定GEN参数 | 冻结 | 冻结 |
| 原生UND基础权重、文本embedding/lm_head | 冻结 | 冻结 | 冻结 |
| reader输入适配器、UND注意力LoRA | 不用 | 训练 | 从caption checkpoint继续训练 |
| 新纠错读出 | 不用 | 不训练 | 训练 |

优先共享冻结UND基础权重，采用独立的reader LoRA调用路径，避免再常驻一整份8B副本。原生prompt前向不应用reader LoRA；不得临时原地改权重来切换模式。FSDP中每个参数只有一个注册owner，adapter checkpoint只保存可训练状态和底座身份。

## 4. 时间、目标与guidance-safe：迁移时必须重写的部分

以下t只使用U1.5定义：0为噪声、1为干净。像素归一化、分辨率噪声尺度σ_R与原生实现一致。

### 4.1 原生轨迹与速度

\[
z_t=t x+(1-t)\sigma_R\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
\]

\[
v^*=x-\sigma_R\epsilon,
\qquad v_c=\frac{\hat x_c-z_t}{\max(1-t,t_{\rm eps})}.
\]

`t_eps`读取原生config。训练不取t=1计算除法。采样最后一个前向节点严格小于1，Euler终点可等于1。

原生CFG得到v_g。所有CFG interval、noise_scale、time shift和renorm都保留该固定U1.5配置。不能拿BAGEL的global-renorm默认值覆盖U1.5的`cfg_norm`。

### 4.2 迁移当前guidance-safe目标

以实际原生CFG速度与条件速度之差定义：

\[
g=v_g-v_c.
\]

对每张图像独立，将RGB通道和空间展平：

\[
H_g(a)=a-\min\left(0,\frac{\langle a,g\rangle}{\|g\|^2}\right)g.
\]

g为0时H_g为恒等映射；实现中用FP32和明确的近零分支/数值下界，不跨batch做投影。

训练目标沿用已选择的conditional-residual方案：

\[
r^*=H_g(v^*-v_c),\qquad
\mathcal L_{\rm corr}=\operatorname{mean}\left[(\Delta v_{\rm raw}-r^*)^2\right].
\]

部署：

\[
\Delta v=H_g(\Delta v_{\rm raw}),\quad
z_{t+\Delta t}=z_t+\Delta t(v_g+\Delta v),\quad\Delta t>0.
\]

该loss衡量conditional残差拟合，部署则叠加在guided速度上。它不是“guided速度到v*的最小二乘最优修正”；必须同时记录两种物理量，不能通过改名混用。

若以后改为clean-image增量，必须重新推导`δv=δx/(1-t)`及相应v-loss权重；首版不切换到该输出。

### 4.3 纠错作用区间

首版固定50次Euler更新，即51个时间节点；这是U1.5的`num_steps=50`，不同于BAGEL此前50节点/49更新。

纠错固定作用于更新序号0–24，后25次更新使用原生速度，保持已有“前半程纠错”的研究设置。训练从这25个实际时间节点采样，不能用BAGEL的t列表，也不能把25/50解释成t≤0.5：time shift会改变物理时间。

caption诊断另外报告早、中、晚三个噪声区间；不因高噪声caption较差而静默删去前半程训练。若证据支持改区间，作为独立的后续消融，不混入首次底座迁移。

## 5. 完整训练链：三个checkpoint边界

```text
原生U1.5 release
    ├── native基线（保留）
    └── 可选纯T2I SFT → 新GEN底座 G_SFT
                             │ 固定底座，不再同时更新
                             ▼
                  GEN-hidden caption适配 → reader checkpoint
                             │ 保留全部已训练reader参数
                             ▼
                   guidance-safe纠错 → corrector checkpoint
```

### 5.1 纯T2I SFT：建立能力，不抢先混入迁移结论

数据源沿用用户提供路径：

`/data/250010019/unimoe_r3/qwen_image_vllm_20261019/pairs/pairs-20261022.jsonl`。

先检查真实schema、图片可读性、尺寸/比例、caption空值、重复与family划分。保留原划分并排除六榜评测来源及同源派生样本。不能在未读schema前假定字段名；转换清单保存原record ID和source位置，不改原文件。

使用U1.5原生RGB归一化、分辨率噪声尺度、x-pred/v-loss与空间head。文本条件以0.1概率整段丢弃是拟采用的CFG维护设置，明确写入新config；不是声称复现了U1.5全部原训练配方。首版建立GEN-only SFT，UND基础权重冻结。

U1.5报告另含感知损失。若当前发布训练代码没有完全对应实现/系数，先以“GEN-only flow-loss SFT”命名，不称完整U1.5 SFT复现，不自行补猜LPIPS权重。BAGEL Table 3的15K步、EMA、LR不能直接成为U1.5论文配方。

顺序建议：先用原生release完成接口和基线；SFT作为独立底座版本。选定SFT底座后，重采GEN hidden、重训reader和corrector、重估global。不把旧corrector直接套到新GEN上。

### 5.2 Caption能力继承阶段

训练样本为自然图文对对应的GEN前向状态与caption；读取点、t与噪声seed都记录。可采用真实图像加噪的原生状态，但验证同时覆盖真实采样轨迹，区分两者。

只训练reader输入适配器与UND注意力LoRA，通过原生lm_head计算caption CE。目标caption不作为reader前缀输入。

验证至少包括：

- 与原native UND像素理解能力分开报告GEN-hidden caption结果。
- 同prompt、不同seed下，描述是否随实际生成内容变化。
- 交换GEN观察后输出是否跟随观察；空观察是否失去关键信息。这是诊断，不是新增训练loss。
- 对缺物体、计数、颜色绑定、左右关系单独记录可读性，不用流畅长caption掩盖错误。
- 不只在干净图像状态验证，报告正式纠错将使用的时间区间。

验收后保存adapter、LoRA、输入层/位置/mask契约和原生模型身份。正式纠错必须从这个checkpoint加载，而非重新初始化reader。

### 5.3 正式纠错训练

冻结选定GEN底座，逐样本构造z_t与原生条件/无条件预测；H_GEN、文本上下文、v_c/v_g都detach。梯度只进入reader适配器、reader LoRA和纠错读出。

使用第4节的单一MSE目标。默认不加semantic CE、cosine、raw-correction、shuffled或人为correct<off约束。caption能力由阶段初始化继承，期间用固定诊断集观察是否漂移。

下列数值是首轮工程起点，不是论文复现或质量保证，也不代表已获新训练预算授权：

| 项目 | 建议起点 |
|---|---|
| 分辨率 | 512×512，先固定方图 |
| reader LoRA | rank64，Q/K/V/O |
| corrector优化器 | AdamW，betas=(0.9,0.95)，eps=1e-8，weight_decay=0 |
| reader适配器/LoRA学习率 | 1e-5 |
| 新纠错head学习率 | 1e-4 |
| warmup / clip | 100步 / global norm 1.0 |
| global batch | 256；显存不足时用梯度累积，记录每rank microbatch |
| 采样 | 原生50更新，CFG4、shift3、cfg_norm=none作为明确固定配置候选 |
| 观察点 | 10/100/500/1000步；1000步为第一次阶段决策 |
| 后续长跑 | 通过语义与画质检查后再确定；不默认继续到5000 |

原生采样配置先检查该release的示例与config是否匹配；发现差异时在P1固定采用的值，再统一用于native/global/corrected，不按结果挑配置。

零输出head在首步会阻断到reader的梯度，首步主要更新head；后续reader开始收到梯度。这是初始化的预期行为，不用额外零门控修补。

## 6. 采样、global基线与公平对照

首版每步最多一个GEN条件预测、一个无条件预测、一个UND reader。reader只在前25更新运行。GEN观察从同一次条件前向获取，不额外再跑一遍GEN。

三条对照必须保持相同底座、分辨率、seed、noise_scale、时间网格、CFG及原生数值路径：

- `native`：只运行U1.5原生采样。
- `global`：不读取图像状态，使用训练集估计的逐时间、逐RGB通道常量增量，经过相同H_g后应用。
- `corrected`：reader和空间head输出增量，经过相同H_g后应用。

global只在训练集估计。新底座或新时间网格必须重估，不加载BAGEL校准。最终同时报告corrected−native和corrected−global。

额外做一次同结构、同预算的UND初始化与GEN初始化比较，用于判断UND特有贡献；不在主训练里强制制造UND优势。随机冻结大模型不是主要公平基线。

## 7. 验证内容与阶段判断

### 7.1 实现正确性

- 零head时，corrected输出与同环境原生采样在预先约定的BF16数值容差内一致；CPU投影与Euler公式测试单独通过。
- native条件/无条件分支与观察分支身份正确；prompt cache不被reader污染。
- RGB尺度、σ_R、t方向、终点除法、THW位置和矩形图网格正确。
- 每图独立投影，`applied_negative_fraction`在数值容差内为0。
- 只允许规定参数出现梯度；checkpoint续跑恢复optimizer、scheduler、sampler和各rank RNG。
- 1卡、单节点和四节点的损失归一化一致；不重复除以world size。

### 7.2 训练日志

至少记录：

`loss`、`zero_correction_mse`、`conditional_base_mse`、`conditional_corrected_mse`、`applied_base_mse`、`raw_delta_rms`、`applied_delta_rms`、`guidance_projection_raw/applied`、`applied_negative_fraction`、`gradient_reader/head`、`t/noise_scale`、吞吐与峰值显存。

loss只与同目标的zero_correction_mse直接对比。caption诊断、画质、语义和MSE分别报告。

### 7.3 语义与画质验收

第一轮使用固定32条困难prompt、固定seed，包含计数、颜色绑定、左右/上下关系、文字与姿态；同时查看native/global/corrected。记录逐题成功/失败、风格漂移与结构破坏。

阶段判断：

- reader caption不能读取生成内容：回到接口继承，先不扩大corrector训练。
- caption可读、loss下降，但语义不优于global：定位输出映射/监督与轨迹问题，不以增加训练步数替代判断。
- 语义改善但画质明显下降：检查增量尺度、时间区间和轨迹偏移，单独设计下一次实验；不在同一run中偷偷改强度。
- 语义与画质均有可复查收益：再进入六榜与重复seed验证。

完整六榜的判定使用相同prompt上的配对结果；给出配对差值和置信区间。GenEval等宏平均按原类别聚合，统计重采样也保持类别结构。没有统一“提升几个点”门槛冒充已知噪声水平。

## 8. 六榜与HTML迁移

沿用已确认的本地Qwen3.8-27B内部评分：

| benchmark ID | 单arm图片数 |
|---|---:|
| geneval | 553 |
| geneval_pp | 280 |
| t2i_compbench_pp | 2400 |
| dpg | 1065 |
| t2i_corebench | 1080 |
| wise_verified | 1000 |
| 合计 | 6378 |

能力阶段先用32条，不立刻全量生成三个arm。完整native/global/corrected比较为19134张，需单独列出生成与评分成本后执行；不能把单arm6378张预算默认为已包含三arm。

复用原manifest的prompt、seed、题目、rubric、parser与聚合，只替换生成器绑定。新arm建议`u15_native`、`u15_global`、`u15_corrected_stepXXXX`。同整数seed可用于跨底座描述性比较，但BAGEL与U1.5的噪声空间不同，不称相同初始噪声的受控实验。

每张PNG保存原记录、seed、底座版本、reader/corrector checkpoint、实际sampler config、trace、耗时和图像完整性信息。所有榜单校验完整ID集合；失败留pending，不计0。

继续用独立27B代理与新namespace，保持裁判版本和运行栈一致。原BAGEL评分未结束时不抢占其服务或覆盖缓存。

HTML增加“底座/训练阶段/arm/评分协议/最后同步时间”字段，展示：

- 六榜有效/期望条数、完成状态和原summary链接。
- native/global/corrected总分与分类差值；未完整评分不报最终总分。
- 同prompt同seed的三路图像与原始文字。
- 代表性成功和失败记录，固定预览顺序，不仅挑选改善样本。
- 生成耗时、额外UND调用次数、峰值显存，支持质量—成本比较。

首次迁移不给不同类别的六个总分简单求平均，除非另行定义并固定该综合指标。

## 9. Ascend适配、64卡扩展与环境边界

### 9.1 环境

本地发布训练按Torch2.5.1/CUDA12.4与FlashAttention验证，不是可直接调用的64卡NPU脚本。新建U1.5专用环境和部署目录，保留现有BAGEL与27B环境。

不把当前参考目录整体改成NPU版；固定上游快照，必要的device/attention适配放到独立副本或明确patch中。

### 9.2 适配顺序

1. 严格加载原生权重、tokenizer、配置和PixelShuffle head，核对缺失/额外参数；不静默忽略key。
2. 运行原生SDPA推理路径；核对GQA、causal/双向mask、THW RoPE、KV cache与dtype。
3. 将硬编码CUDA设备、随机数、同步、attention/并行算子替换为明确NPU实现；保留官方可用路径，不全局monkey patch CUDA。
4. 先验证原生文生图，再验证caption reader前向/反向和corrector数步更新。
5. 先用PyTorch DDP/FSDP与HCCL完成所需训练路径，不为迁移首版重建整个上游ISP训练平台。
6. 显存不够优先使用分片与activation checkpointing。用真实峰值决定microbatch，不按BAGEL显存推断U1.5。

四节点×16逻辑NPU为目标拓扑，不是实施前提。推理生成可按全局record index分片；训练用64rank同步和梯度累积。启动前核对本任务可用设备，不停止无关进程。

共享冻结UND权重、FSDP包装和LoRA调用路径由同一实现owner负责，避免参数重复注册或各分支各建一套状态。

### 9.3 证据与恢复

在扩到64卡前具备：原生PNG、采样trace、caption样例、一次真实反向、checkpoint加载与续跑、单节点多rank损失归一化结果。只验证实际改变的路径，不扩成无关算子矩阵。

每个run独立目录、进程组、node/rank日志、PID与退出码。停止/恢复只指向本run；失败保留日志和最后完整checkpoint，不用删除输出重跑掩盖失败。

## 10. 实施阶段、交付物与预算决策

以下为执行顺序；步数是建议的阶段上限，需要实施时固定，不是本文件自动启动训练的授权。

| 阶段 | 核心工作 | 交付和进入下一阶段的条件 |
|---|---|---|
| P0 契约恢复 | 固定U1.5 release；恢复成功caption实验；读数据schema | MODEL_MANIFEST、CAPTION_INTERFACE、数据划分；明确能继承哪些对象 |
| P1 原生适配 | 独立环境、NPU原生采样、固定配置 | 原生32条图册；新接入不改变原生含义；测吞吐/显存 |
| P2 T2I SFT能力 | 构建GEN-only RGB训练与续跑，先小规模验证 | 可运行SFT入口及独立checkpoint；是否执行长SFT另作底座决策 |
| P3 语义继承 | reader caption，建议先1000步内判断 | 分时间区间的caption诊断、同prompt状态敏感性、完整reader checkpoint |
| P4 纠错pilot | guidance-safe训练10→100→500→1000步 | 有效梯度、稳定画质、相对native/global的逐题语义记录 |
| P5 扩展评测 | 按P4结果决定更长训练或六榜 | 训练预算固定后执行，完整分母和三路成本；不自动追跑 |
| P6 交付 | 固定代码/config/模型身份，整理结果页 | 可用源码、命令入口、checkpoints、原始题级结果、HTML和结论 |

首个里程碑不是“64卡已经跑起来”，而是“原生生成正确＋GEN-hidden语义可读＋纠错不破坏基线”。通过后再用64卡增加吞吐。

P0–P4的实现、接口、采样语义和FSDP参数归属应有单一owner。数据转换与评分接入可独立推进，但必须消费同一冻结的record ID和模型身份。文档不安排新agent或启动新任务。

## 11. 建议代码组织与交付命令接口

在现有项目中新建独立命名空间，避免把BAGEL Runtime改成多层条件分支：

```text
qwen_latent_cot/u15/
  runtime.py         # 原生模型、文本cache、RGB状态、GEN观察
  reader.py          # caption继承接口与UND LoRA
  readout.py         # 单一prompt-aware空间纠错头
  objective.py       # RGB目标与逐图guidance-safe
  sampling.py        # 原生采样及native/global/corrected
  training.py        # caption / correction训练与日志
  t2i.py             # GEN-only RGB SFT
  checkpoint.py      # 底座、reader、corrector身份与精确恢复
  cli.py
configs/u15/
scripts/launch_u15_64.py
scripts/benchmark_u15_sixbench.py
```

数据审计/划分和六榜评分原逻辑优先复用，不在新目录复制一整套scorer。下面是待实现的CLI契约，不是当前已有可执行命令：

```text
python -m qwen_latent_cot.u15.cli native --model MODEL --config CONFIG --prompts PROMPTS --output RUN
python -m qwen_latent_cot.u15.cli t2i --model MODEL --corpus CORPUS --config CONFIG --output RUN
python -m qwen_latent_cot.u15.cli caption --model MODEL --corpus CORPUS --config CONFIG --output RUN
python -m qwen_latent_cot.u15.cli correct --model MODEL --reader READER --corpus CORPUS --config CONFIG --output RUN
python -m qwen_latent_cot.u15.cli sample --model MODEL --checkpoint CHECKPOINT --arm corrected --prompt PROMPT --seed SEED --output PNG
```

每个阶段使用完整config快照。checkpoint包括模型release/revision、代码版本、真实采样与观察接口、可训练参数清单、数据划分身份、训练步、优化器/调度器/RNG；不允许跨底座静默resume。

## 12. 尚需实施时落实的输入

这些是具体工作项，不是要求用户重新描述目标：

1. 找回成功GEN-hidden caption实验的代码和checkpoint位置；现有对话只确定它成功，尚未恢复完整接口。
2. 固定实际下载的U1.5权重revision与模型config，核对原生head和噪声配置。
3. 读取用户图文对schema，确认训练/验证和评测来源排除结果。
4. 测量NPU原生与reader成本，确定microbatch和64卡训练方式。
5. P4之后根据实际语义与画质证据确定长跑预算。

在上述输入未落实前，可以实施原生runtime与数据读取，但不能填造已验证caption接口、吞吐、显存或预期提升。

## 13. 来源与配套记录

- [SenseNova-U1官方仓库](https://github.com/OpenSenseNova/SenseNova-U1)
- [U1技术报告](https://arxiv.org/abs/2605.12500)：视觉入口、MoT分工、x-pred/v-loss。
- [U1.5技术报告](https://arxiv.org/abs/2609.11929)：空间重建、统一训练与后训练。
- 本地原生实现：`src/sensenova_u1/models/neo_unify/modeling_neo_chat.py`、`modeling_neo_vit.py`、`modeling_qwen3.py`，均相对参考项目根目录。
- 本地训练说明：参考项目`training/README.md`、`training/pyproject.toml`。
- [迁移评估](sensenova_u1_transfer_assessment_20261022.md)。
- [UND能力继承审计及用户澄清](und_corrector_inheritance_audit_20261022.md)。

**最终选择：迁移底座与基础设施，继承已验证的GEN-hidden语义读取通路；首版保留单一guidance-safe速度增量接口。先把UND读懂内部生成状态这件事接实，再验证它能否产生超越native/global的纠错增益。**
