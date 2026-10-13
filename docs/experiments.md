# 实验记录

更新：2026-10-12。本页仅记录已执行的实验、日志指标与用户人工审查反馈。

[返回README](../README.md) · [设计](design.md)

## 1. Training-free探索

|实验|结果|
|---|---|
|loop-layer推理模式比较|legacy_layerwise的training-free效果最好|
|将Memory作为独立latent序列|生成侧对Memory内容不敏感；随后使用prompt hidden初始化Memory|
|每层独立更新与一份Memory经过连续UND层|人工审查图像结果差异不大|
|joint-micro残差更新|R1之后的图像全部退化为噪点|

## 2. C0原生文本校准

16条人工编写caption，任务为原生caption上下文的自由复述。没有压缩Memory或低秩残差训练。

|指标|结果|
|---|---:|
|完成样本|16/16|
|平均教师强制NLL|0.271115|
|严格文本匹配|7/16|
|预设字面事实分|15条为1，c0_08为0|
|含加载与汇总的端到端耗时|54.0313秒|

人工审查：主要对象、颜色、数量与关系保留。c0_08的“single orange”与“one orange”等价，但字面词表未计入；c0_15出现额外尺寸、材质和光照描述。C0经人工确认放行；严格复述存在改写与扩写。

输出：`/root/outputs/c0_text16_20261012_111241`。

## 3. C1 Caption训练

|指标|L历史参照|C连续更新|
|---|---:|---:|
|第1步训练NLL|2.958457|2.976080|
|第100步训练NLL|0.430248|0.000001472|
|best步数|50|100|
|best验证NLL|1.336711|0.458495|
|best验证字面事实分|16.67%|45.83%|
|第100步验证颜色正确|4/8，均输出红色|8/8|
|第100步完整文本匹配|0/8|0/8|
|rank0框架分配峰值|28.312 GiB|28.415 GiB|
|训练日志计时|41.542秒|51.208秒|

计时包含中途验证，不含模型加载与最后一次验证。C的自由输出仍有对象和承载面替换，8条中7条把desk替换为table；bicycle→bike未被字面评分计为正确。后续实验使用C的step 100 best。

输出：`/root/outputs/c1_pilot_20261012_113553`。C checkpoint为 `C/best.pt`，SHA256 `d12b40adbf3f7aced96e51afef117fa8efda9ef42037f3e1b6147cac159a1ac2`。

## 4. 源图筛选与GEN对照定义

清单 `memory_gen_sample16.jsonl`采用8个异场景swap对，配对标记为 `unrelated_scene_mismatch`，共同prompt为 `An image.`。所有同图/seed/时间的条件共享同一状态。

|组|实际条件|
|---|---|
|G-base|原prompt与原生位置布局|
|G-layout|预留Memory位置，彻底屏蔽新增KV|
|G-fixed|全零且可见的Memory KV|
|G-neutral|固定中性文本种子的KV|
|G-text|原prompt后原生追加caption|
|G-TC|caption经C writer压缩得到的16槽Memory|
|G-swap-TC|读取配对另一张图caption的C Memory|

图像状态使用 `x_t=(1-t)x0+tε`，评测图为 `decode(x_t-t*v)` 的局部x0_hat。下列GEN实验均运行7组条件，保留原生无文本CFG分支，没有续采样或视觉Memory。

源图输出：`/root/outputs/source_sample16_20261012_121337`。所有数据清单位于 `/data/zyd_workspace/bagel-LatentCoT/datasets`。

## 5. 蒸馏前GEN评测

16图×2个noise seed（0/1）×7组，共32状态、224张局部预测图；

|组|相对G-layout条件速度差异|相对G-layout的CFG速度差异|
|---|---:|---:|
|G-base|0.01933|0.07735|
|G-layout|0|0|
|G-fixed|0.00563|0.02251|
|G-neutral|0.01607|0.06413|
|G-text|0.02502|0.09965|
|G-TC|0.01357|0.05424|
|G-swap-TC|0.01355|0.05416|

### 原4图、t=0.7/0.8/0.9

step 100，noise seeds 0/1/2/3，16卡按2对×4seed×2组t分片。共48状态、336张局部预测图；冻结检查通过。

|actual t|G-text vs G-base像素差|G-TC vs G-swap-TC像素差|G-TC vs G-neutral像素差|
|---|---:|---:|---:|
|0.7|4.4156|0.6599|1.8853|
|0.8|7.8226|0.8632|2.6719|
|0.9|21.8758|1.9538|4.8701|

各项为该t下16状态的平均像素绝对差（0–255）。用户人工审查：只有G-text有明显变化，其余对照组几乎完全相同。

输出：`/root/outputs/gen_recheck_C_t789_20261012`；数据清单为 `memory_gen_recheck4.jsonl`。

## 6. GEN响应蒸馏100步

2026-10-12，modelarts-job，提交 `788de2e`。从C1-C step 100初始化，16卡HCCL、global batch16、lr=1e-5、100步、每50步验证。损失为response＋0.1×caption NLL；原生BAGEL冻结检查通过。

文本沿用64 train / 8 val / 8 test。Query图为原16图中的12 train / 4 val；各rank缓存一张训练图在t=0.7/0.8/0.9的3份探针。16卡中的4张训练图重复分配，但noise seed不同。验证query使用独立seed。

响应读出固定0/13/27层，每层64个query，预留64个条件位置，共同prompt/self KV进入同一softmax。

|响应训练步数|验证response|验证Caption NLL|验证字面事实分|选择损失|
|---|---:|---:|---:|---:|
|0|0.044724|0.458495|45.83%|0.090574|
|50|0.033690|0.395288|58.33%|0.073219|
|100|0.020588|0.752646|66.67%|0.095852|

best按验证response＋0.1 NLL选中step 50；用户指定step 100 last用于GEN审查。

输出：`/root/outputs/c1_response_20261012_134848`。Step 100 last SHA256：`725ab4a4cc33a3fa68cfaa1283905948dd0d885a397c4ceeb724009045e2c2ce`。

## 7. 响应100步的GEN对照

2026-10-12，使用响应step 100 last，复用原4图、noise seeds 0/1/2/3、t=0.7/0.8/0.9，16卡，共48状态、336张局部预测图。

用户人工审查：G-TC在一些样例上与base、swap等其他对照组明显拉开差距。未提供逐图评分。

输出：`/root/outputs/gen_response100_t789_20261012`。

## 8. 扩展响应训练1000步

2026-10-12，modelarts-job，训练提交 `1f2b614`。从响应step 100 last加载writer，重新创建AdamW，新增1000步，累计响应1100步。16卡HCCL、global batch16、lr=1e-5、每250步验证；损失与读出设置沿用首跑，冻结检查通过。

|数据|训练|验证|测试|
|---|---:|---:|---:|
|原模板文本|64|8|8|
|新增自然描述|2048|32|32|
|合计文本|2112|40|40|
|GEN query图|128|原4图|未另设|

自然描述取自cort36k各split的短生成prompt，最多40 tokenizer tokens，只用于文本oracle，没有作为图像caption标签。其source ID与入选训练query图分离。每卡8张训练图×3个t，共384份训练探针。测试文本未参与训练或checkpoint选择。

|本轮步数|累计响应步数|验证response|验证Caption NLL|模板字面事实分|
|---|---:|---:|---:|---:|
|0|100|0.032839|4.789251|66.67%|
|250|350|0.024535|0.482668|100%|
|500|600|0.021407|0.279060|100%|
|750|850|0.019490|0.221803|100%|
|1000|1100|0.018171|0.163620|100%|

验证均值覆盖40条文本；字面事实分只覆盖8条有预设词表的模板，自然描述没有该评分。末步模板response/NLL为0.013619/0.000258，自然描述为0.019309/0.204460。选择损失0.034533，best选中本轮step 1000。

输出：`/root/outputs/c1_response_expand_20261012_141729`。训练日志区分本轮step与 `response_cumulative_steps`；原C1的100步Caption训练不计入响应步数。

## 9. 累计1100步的GEN对照

2026-10-12，modelarts-job。使用扩展训练step 1000 best，累计响应1100步。原4图，actual t=0.9，noise seeds 0–7，16卡，共32状态、224张局部x0_hat；冻结检查通过，没有续采样。

Checkpoint：`/root/outputs/c1_response_expand_20261012_141729/best.pt`，SHA256 `255aa6d7c5ffed326a554891e2189dbbd2bdc309fe05c6433b10258a095e3f56`。

**人工审查结论：G-TC与G-text性能效果基本持平。** 本记录对应所测局部预测图，没有逐图量化评分或最终采样画质统计。

## 10. 视觉闭环实现与Ascend E0

2026-10-12，modelarts-job，NPU分支提交 `92a17f5`。实现C2-observe、C2-joint及C3-loop，默认冻结BAGEL/DiT/VAE和原文本writer；新增感知t的低秩UND query残差。正式训练入口默认16卡，E0按方案使用单NPU。

初始化及独立教师沿用第9节1100步best；远端SHA256已核对。E0复用 `memory_gen_recheck4.jsonl` 中 `sharegpt4o_1117`、`sharegpt4o_42006` 两图，检查t=0.25/0.75/0.9的三个固定状态；完整R2反向使用首图、t=0.75、noise seed43。

|检查|实测结果|
|---|---|
|无Memory可微conditional与原生conditional|三个t最大绝对差均为0|
|屏蔽Memory与原生平移16位置参照|三个t最大绝对差均为0|
|零视觉模块与C1直接GEN读取、关闭视觉模块与文本writer|逐项一致|
|完整R2速度损失到视觉模块|28层输出投影首步梯度有限且非零；第二步输入投影也非零|
|同一状态R2、固定M1重复调用、诊断开关|合同通过；prompt cache未变化|
|BAGEL与VAE参数哈希|前后均为 `c50b0a68abad7914aa9fbe5a4fff450e2bb4453c54651f6b09b7039f0f06d1f2`|
|峰值NPU内存|38,551,704,576 bytes，约35.9 GiB|
|检查计时|28.83秒；模型加载、manifest及首次参数哈希在计时外|

两个工程更新的flow MSE为0.237012和0.236920；测试结束恢复视觉模块初值，没有保存为训练模型。容差记录在 `e0.json`：rtol=0.02，atol按BF16精度、原生重复误差及速度RMS计算；原生重复误差均为0。

首次检查在屏蔽布局处失败：0.4%元素超过容差，最大差0.0625。保留16位置平移，并让屏蔽臂直接使用原生attention；可见Memory的GEN attention使用原生无mask SDPA，旧UND路径保留。修复后平移对照误差为0。第二次检查在恢复初值时复制冻结writer，触发参数版本检查；改为仅恢复被测试更新的视觉模块后，完整E0通过。两个失败输出保留，未记为训练结果。

通过输出：`/root/outputs/visual_loop_e0_pass_20261012/e0.json`。本地CPU测试使用真实小型MoT层及两进程梯度同步，66项通过。当次E0时，正式16卡视觉训练、独立图像标签评测和完整采样质量评测尚未执行；512/64/64视觉清单尚未准备，现有自然文本prompt没有升级为配对图像标签。

## 11. 全量视觉配对清单

2026-10-12，modelarts-job，用户改为使用全量配对数据，以训练步数控制预算。`scripts/prepare_visual_data.py`读取CoRT一致性数据的三个原始split，只取标注为匹配的终态图像及其目标文本，生成 `/data/zyd_workspace/bagel-LatentCoT/datasets/visual_memory_full.jsonl`。

|项目|数量|
|---|---:|
|来源记录|36,624|
|训练|29,300|
|验证|3,662|
|测试|3,646|
|入选合计|36,608|
|排除此前GEN审查图|16，均来自原test split|
|重复图片、缺失图片及不可读图片|0|
|caption不超过48 tokens|29,861|
|caption为49–128 tokens|5,801|
|caption超过128 tokens|946|

保留完整caption与原始split，没有重新划分或截断文本。数据集SHA256为 `157089d0d0ed157b9fdd179985f54d5b8ce4fa0440980eee36bd97601f4c33ee`；来源清单哈希及构建统计保存在同目录 `visual_memory_full.summary.json`。

远端使用训练数据读取器完成全量校验：36,608条、36,608个独立family，schema、图片存在性、锁定图片哈希、family及图片内容跨split隔离均通过。输出为同目录 `visual_memory_full.validation.json`。

匹配依据是数据集的终态一致性标注，不是新增人工caption审查。记录 `human_reviewed=false` 及来源范围；没有生成事实标签，`target_facts=[]`，事实准确率不可用。学生GEN prompt固定为 `An image.`，答案caption只用于监督。观察代理标签尚未构建。

训练入口默认使用全量train池，16卡、global batch16、25%文本回放，通过 `--steps`控制更新次数。训练期间的val固定选择最多64个场景，独立评测保留完整val/test。输入校验改为rank 0执行一次后分发，避免16个进程重复读取全部图片。相关18项本地测试通过；尚未启动视觉训练。

## 12. C2-observe及固定来源对照

modelarts-job，`/root/outputs/visual_observe_full_20261012`，16卡C2-observe完成1000步。训练期间只优化语义损失，read/gen权重为0；BAGEL、VAE、文本writer及教师冻结检查通过。固定64个val场景、4个种子，排除t=1后的1536个状态用于checkpoint选择。

|Step|验证R1语义NLL|
|---:|---:|
|250|3.579268|
|500|3.469990|
|750|3.399632|
|1000|3.336412|

`best.pt`与`last.pt`均为1000步。训练期验证仅包含R1，没有来源对照；未记录为视觉接地或GEN质量验收通过。

2026-10-13执行来源对照，输出 `/root/outputs/visual_source_review4_20261013`。使用上述1000步best，SHA256为 `3d9cc1f4f728cedbaccafedcb15f2c999ccf846e49fe365b7149f53e36ce303a`。固定val样本 `visual_sharegpt4o_5178`、`visual_sharegpt4o_32408`、`visual_sharegpt4o_11204`、`visual_mj_v6_122908`，前两者互换来源，后两者互换来源。t=0.25/0.75/0.9、noise seed0，共12状态、正确/关闭/交换三组、36条自由caption。

正确与关闭组使用本图的同一状态；交换组只将writer的视觉来源换为另一图在相同t、种子、中性prompt下的来源。三组接收相同固定描述指令，自由caption不读取答案前缀。没有执行Memory的GEN注入、完整生成或参数更新。审查页提供原图、原生条件x0预测代理、随机排列的caption及可导出的标注。

工程核对：各组本图状态相同；12/12交换来源状态哈希与交换图正确组相符；12/12交换caption与交换图正确组相同；每个t下四张图的关闭来源caption相同。新增来源对照测试通过。

2026-10-13，用户人工审查记录：正确来源组的caption在t较小时与图像的语义相关性较强；t>0.75后开始出现模板化文字和与图像无关的文字。
