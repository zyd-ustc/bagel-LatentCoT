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
