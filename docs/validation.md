# 验证

状态：C0/C1已完成；保留C；GEN读取与C2尚未评测。更新：2026-10-12。

[返回 README](../README.md)

## 1. 已有证据

|记录|实际范围|
|---|---|
|[首次构建](archive/build.md)|本地macOS arm64；23项测试、wheel构建与安装后CLI检查|
|[审查修正](archive/review.md)|修正后40项测试；0.6.1 wheel与源码一致；梯度故障注入和原生时间网格检查|
|[历史清理](archive/cleanup.json)|旧模块/脚本/配置清单与备份位置；不是当前运行说明|

测试使用随机初始化的小型原生Qwen2MoT/BAGEL模块。流程中的VAE/tokenizer是替身；另有真实Qwen2Tokenizer类配临时词表的检查，仍未使用正式tokenizer文件。

这些是已有工程记录，本轮文档整理没有重新运行模型测试。日志和wheel在外层工作区`artifacts/memory_topology_build_20261011/`与`artifacts/memory_topology_review_20261011/`，不保证随仓库分发。

## 2. 已记录的修正

- 检查每层Memory K/V梯度与C跨冻结UND的梯度。
- 记录两个swap的等价关系，避免重复算证据。
- C0使用独立复述指令及原生tokenizer/BOS/EOS协议。
- 后续采样绑定原生actual t及完整时间网格。
- t=1同时检查跨x0输入与自由输出；修正局部否定事实规则。
- 使用线性残差；区分零可见G-fixed与中性G-neutral。
- 独立抽t/扩散seed，VAE posterior由图像内容hash派生。

## 3. C0真实权重结果

2026-10-12，`modelarts-job`，`loop-layer-NPU`提交`152e13a`。模型为 `/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT`；输入为 `data/c0_text16.jsonl`；输出为 `/root/outputs/c0_text16_20261012_111241`。16卡各处理一条人工编写文本，任务是原生caption KV的自由复述，不涉及图像或压缩Memory。

|指标|结果|
|---|---|
|完成样本|16/16|
|平均教师强制NLL|0.2711153743|
|严格文本匹配|7/16（43.75%）|
|逐条预设字面事实分|15条为1；`c0_08`为0|
|16卡端到端耗时|54.0313秒，含启动/加载/汇总；不是训练吞吐|

人工检查：主要对象、颜色、数量与关系保留。`c0_08`输出“single orange”，与参考“one orange”等价；字面词表漏掉同义词。`c0_15`保留五个紫色积木，但新增尺寸、地板材质与光照等无依据描述；事实分1不能排除扩写。其余多为同义改写或引导语。

结论：原生文本语义读取可用，允许进入C1小预算；严格复述未通过，存在无依据扩写。此结论不证明Memory压缩、视觉接地或GEN消费成立。校准文件初始`passed: false`是待人工核验状态，不是自动失败；放行时需保留上述范围和限制。

## 4. C1结果与拓扑选择

2026-10-12，`modelarts-job`，源码提交 `c3eff5a`，输出 `/root/outputs/c1_pilot_20261012_113553`。L/C各完成100步，HCCL world size 16、全局batch16、BF16；冻结检查通过。每组训练64条模板文本，验证8条（4个场景对）；测试集尚未运行。

|指标|L：每层独立|C：连续更新|
|---|---:|---:|
|第1步训练NLL|2.958457|2.976080|
|第100步训练NLL|0.430248|0.000001472|
|选中checkpoint步数|50|100|
|选中checkpoint验证NLL|1.336711|0.458495|
|验证字面事实准确率|16.67%|45.83%|
|第100步验证颜色正确|4/8，全部输出红色|8/8，4对颜色均跟随输入|
|第100步验证完整文本匹配|0/8|0/8|
|rank0框架分配峰值|28.312 GiB|28.415 GiB|
|截至第100步日志计时|41.542秒|51.208秒|

计时从训练循环开始，包含中途验证，不包含模型加载和最后一次验证；峰值仅为rank0的框架分配量，不是16卡HBM峰值。

观察：L输出对象和承载面常回落到训练词汇，颜色对比丢失。C保留颜色，但仍有对象替换；8条中7条将desk替换为table。`bicycle → bike`被字面评分漏计，人工认为等价；不改写原评分或checkpoint选择。验证对象及desk均未在训练出现，因此结果含新对象与新承载面的分布变化。

**决定：保留C作为下一阶段唯一学生拓扑，暂停L新增训练。** 依据是共同预算下C验证NLL较低，且自由输出保留4对颜色差异；不是仅按训练loss选择。保留L结果与代码作为历史参照，不删除。

固定checkpoint：`/root/outputs/c1_pilot_20261012_113553/C/best.pt`，step 100，SHA256 `d12b40adbf3f7aced96e51afef117fa8efda9ef42037f3e1b6147cac159a1ac2`。沿用原验证选择规则，不用后续GEN结果重选checkpoint。

范围：这是单seed、模板文本、4个验证场景对上的推进选择，不证明C普遍优于L。C训练NLL接近零与自由输出仍有错误同时成立；文本桥仅部分成立，完整语义压缩、视觉接地与GEN使用尚未证明。

## 5. 联合训练入口检查

2026-10-12，本地41项测试通过（9.30秒）。新增双进程CPU测试验证：参数广播、梯度累积与全局batch缩放、同步梯度裁剪、连续三步AdamW更新，均与单进程全局batch参照一致。

`modelarts-job`的16卡HCCL检查通过：参数广播、累积梯度SUM、连续三步SGD更新、统计归约与记录汇总。使用一个单参数线性模型，不加载BAGEL；日志为 `/tmp/bagel_hccl_check.log`。该小模型检查只验证通信与优化数学；真实BAGEL C1运行结果另见第4节。

## 6. 尚未验证

C1未见测试集与自然caption泛化、C2视觉迁移、最终画质尚未验证。GEN局部预测与人工审查已完成，见第9–10节；当前未显示Memory内容的可辨作用。下一步优先实现C的GEN响应蒸馏，暂不进入C2；语义读出与响应蒸馏仍是待实现方案。

## 7. GEN启动层与数据检查

2026-10-12，43项本地测试通过（5.06秒）。新增检查覆盖完整swap对分片、相对图像路径转绝对路径、16个任务的224条结果汇总、缺失arm拒绝汇总，以及局部预测图使用独立审查文件，不混入最终采样统计。未运行真实GEN任务。

数据根目录为 `/data/zyd_workspace/bagel-LatentCoT/datasets`。已查看CoRT验证清单、结构化test候选和适配manifest的示例：存在多轮编辑目标、`verified: false`和旧绝对路径；这些示例不能直接作为核验颜色对。默认 `memory_gen_color16.jsonl`尚未准备。数据转换与图像语义核验未完成，不将缺失标签归因于模型表现。

## 8. 16张源图筛选

2026-10-12，源图与缩略总览位于 `modelarts-job:/root/outputs/source_sample16_20261012_121337`。清单保存原始文件路径、来源行和prompt。Codex目视核对总览，为16张图分别写简短caption，仅核验可见对象与目标颜色；未核验原prompt的全部细节。

本轮为异场景配对的定性Memory实验，不是同场景仅颜色变化的控制实验。GEN人工审查结论见第10节；最终生成画质尚未评测。

启动器现在将清单的配对类型写入plan与summary，区分控制颜色对与异场景错配。针对分片和汇总的2项已有测试通过（0.83秒，`/tmp/bagel-loop-check-venv`）；系统Python缺少PyTorch动态库，未用于本轮检查。

## 9. 16图GEN定性首跑

2026-10-12，`modelarts-job:/root/outputs/gen_sample16_C_20261012`完成16卡并行、16张源图、2个noise seed、t=0.5、7组条件，共224张局部预测图。代码版本 `72f2031`；C checkpoint沿用第4节，BAGEL冻结检查通过。未续采样，未运行视觉Memory或C2。

|组别|相对G-layout的条件速度差异均值|相对G-layout的CFG速度差异均值|
|---|---:|---:|
|G-base|0.01933|0.07735|
|G-layout|0|0|
|G-fixed|0.00563|0.02251|
|G-neutral|0.01607|0.06413|
|G-text|0.02502|0.09965|
|G-TC|0.01357|0.05424|
|G-swap-TC|0.01355|0.05416|

数值为归一化向量差异，不是图像质量分。参考是G-layout；G-base与其差异包含位置布局影响。G-TC与G-swap-TC的平均幅度接近，不能据此判断内容是否被消费。重复原生CFG误差为0；后续人工审查结论见第10节。

`review/index.html`汇总32页对照图，每页为原图加7组条件；同一源图按seed分开。`local_blind/`保留224张匿名预测图及独立映射。人工优先比较G-text、G-TC、G-swap-TC，并检查G-base与G-layout的布局影响。所有图为x0_hat，不是最终生成结果；异场景swap不构成颜色独立控制证据。

## 10. t=0.7/0.8/0.9复查

2026-10-12，`modelarts-job:/root/outputs/gen_recheck_C_t789_20261012`完成16卡复查。沿用sample_pair_01/04中的4张图、C step 100与7组条件，4个noise seed，actual t=0.7/0.8/0.9，共48状态、336张局部预测图。代码版本 `a69976f`；冻结检查通过，未续采样，未训练新参数。

|actual t|G-text vs G-base像素差|G-TC vs G-swap-TC像素差|G-TC vs G-neutral像素差|
|---|---:|---:|---:|
|0.7|4.4156|0.6599|1.8853|
|0.8|7.8226|0.8632|2.6719|
|0.9|21.8758|1.9538|4.8701|

各项为16状态的平均像素绝对差，范围0–255。教师caption的数值作用随噪声增大，学生Memory与错配Memory的差异仍较小。差异幅度不证明语义正确，也不证明画质改善；人工审查与后续选择如下。

`review/index.html`包含16页，每页按同一源图与seed纵向比较三个t；seed 0/1附上已有t=0.5结果，未重跑。图像仍为单步x0_hat。`pixel_comparison.json`保留逐图数值，`review/annotations.csv`用于记录教师/学生语义作用、错配差异与伪影。启动层相关3项测试通过（0.86秒）；真实结果汇总核对336条完整记录。

### 人工审查与后续选择

2026-10-12，用户审查GEN对照图后反馈：G-text有明显变化；其余对照组（G-base、G-layout、G-fixed、G-neutral、G-TC、G-swap-TC）结果几乎完全相同。记录为总体定性审查，未提供逐图评分或盲评统计，不改写原数值结果。

结论：原生caption条件显示可辨作用；当前C1-C Memory未显示超出布局、固定或中性槽位的可辨作用，正确与错配Memory也无明显区别。现有证据不支持GEN有效使用学生Memory的内容。此结论不否定C1中UND可读出的部分caption信息，也不证明Memory完全不含语义；G-text的明显变化尚不等于语义正确或最终画质提升。

后续：保留C step 100与全部结果，优先实现冻结GEN的响应蒸馏，使新增writer学习原生caption的GEN响应。BAGEL/DiT保持冻结，暂不进入C2、不增加同模板CE步数、不扩大局部噪声网格或续采样。蒸馏尚未实现或运行；具体损失按[设计](design.md#4-路线二冻结-gen-的注意力响应蒸馏优先)制定。

## 11. GEN响应蒸馏实现检查

2026-10-12，首版实现提交 `bdcf0b0`。本地47项测试通过（4.47秒）：新增检查覆盖共同KV/条件KV同一softmax、不同条件长度、教师停止梯度、原生GEN query捕获，以及仅响应损失对Memory K/V和全部C残差的梯度；优化后BAGEL参数保持不变。已有双进程全局batch梯度SUM检查继续通过。

`modelarts-job`真实BAGEL单卡NPU检查通过：固定C step 100，原生GEN在t=0.9捕获共享query，原生caption构造教师响应，响应损失反传到0/13/27层Memory K/V及全部28层残差，并完成一次AdamW更新；冻结检查通过。记录为 `/tmp/response_npu_smoke.log`和 `/tmp/response_npu_smoke.json`。该检查未运行16卡响应训练，不证明生成质量。

数据已准备在指定datasets根目录：`c1_text80.jsonl`保留64/8/8文本划分；`memory_response_queries16.jsonl`为12张训练query图与4张验证query图，验证保留原高噪声复查的4张图。正式100步训练尚未启动，命令见[运行](run.md#c的gen响应蒸馏16卡联合训练)。

## 12. 响应蒸馏100步结果与后续checkpoint

`modelarts-job:/root/outputs/c1_response_20261012_134848`完成100步、16卡HCCL、global batch 16；源码 `788de2e`，BAGEL冻结检查通过。初始/50/100步验证response loss分别为0.044724/0.033690/0.020588，caption NLL为0.458495/0.395288/0.752646，字面事实分45.8%/58.3%/66.7%。按response＋0.1 NLL最小选step 50 best，保留原选择记录。

用户指定下一轮真实GEN对照使用step 100 `last.pt`，SHA256 `725ab4a4cc33a3fa68cfaa1283905948dd0d885a397c4ceeb724009045e2c2ce`。该选择用于检查响应蒸馏较充分的候选，不改写best或宣称生成质量已改善。复用原4图/t/seed的运行与人工审查已完成，结果见第13节；命令见[运行](run.md#第100步响应checkpoint的真实gen对照与本地审查)。

## 13. response100 GEN人工审查与扩展决定

`modelarts-job:/root/outputs/gen_response100_t789_20261012`已完成原4图、actual t=0.7/0.8/0.9、noise seeds 0/1/2/3的16卡对照：48状态、336张局部预测图。使用response100 last，SHA256 `725ab4a4cc33a3fa68cfaa1283905948dd0d885a397c4ceeb724009045e2c2ce`。

2026-10-12用户人工审查：G-TC在一些样例与base、swap等对照明显不同。这是范围有限的积极信号；尚无逐图语义方向评分、成功率或最终采样画质结论，也不推翻蒸馏前对照的阴性结果。

决定：保留C，从response100 last加载writer，新增1000步并扩大文本/query数据；所有原生参数继续冻结，暂不进入C2。扩展预算与数据来源见[实验](experiments.md#响应蒸馏扩展训练)。扩展训练结果尚未产生。

扩展实现检查：49项小模型测试通过（包括128图按16卡各8图分配、自然文本无事实词表的验证汇总）；Python与启动器语法检查通过。远端数据加载确认2112/40/40文本及128/4图像划分，128张训练图全部纳入rank分配；response100 parent hash与冻结接口配置匹配。上述检查不证明生成质量。

扩展任务已于modelarts-job启动：`/root/outputs/c1_response_expand_20261012_141729`，训练源码 `1f2b614`，16卡联合训练。当前结果未完成；最终状态以training_summary及冻结检查为准。扩展验证集新增自然文本，损失只与本轮step 0基线比较，不能直接与首轮8模板验证均值比较。
