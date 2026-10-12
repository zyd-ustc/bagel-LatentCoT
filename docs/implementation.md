# 实现

状态：NPU分支C1联合训练已运行；GEN分片启动层与蒸馏待扩展。更新：2026-10-12。

[返回 README](../README.md)

## 1. 代码与方案映射

|模块|当前职责|
|---|---|
|[memory.py](../qwen_latent_cot/diagnostics/memory.py)|L/C writer、中性seed、线性低秩残差、逐层未旋转Memory bank|
|[native.py](../qwen_latent_cot/diagnostics/native.py)|原生text KV、GEN KV捕获、可微冻结UND解码、单次GEN条件读取|
|[caption.py](../qwen_latent_cot/diagnostics/caption.py)|C0复述校准；C1文本压缩与C2视觉迁移，均为caption CE|
|[runtime.py](../qwen_latent_cot/diagnostics/runtime.py)|VAE编码、状态构造、CFG速度、图像与原生网格续采样|
|[gen.py](../qwen_latent_cot/diagnostics/gen.py)|base/layout/fixed/neutral/text/文本Memory/视觉Memory/swap阶梯|
|[trajectory.py](../qwen_latent_cot/diagnostics/trajectory.py)|原生轨迹及待核验观察标签|
|[data.py](../qwen_latent_cot/diagnostics/data.py)|split、反事实、噪声、字面事实分与聚类区间|
|[record.py](../qwen_latent_cot/diagnostics/record.py)|manifest、checkpoint来源与冻结检查|
|[report.py](../qwen_latent_cot/diagnostics/report.py)|四张结果表与盲评汇总|
|[cli.py](../qwen_latent_cot/diagnostics/cli.py)|唯一执行入口、配置与C0门槛|

当前保留C checkpoint做文本GEN读取；L实现继续保留。GEN入口支持只提供 `--text-checkpoint`，无需视觉模型；目前没有GEN多卡分片/汇总启动器，`--distributed`仅用于训练。下一步所需数据与分片合同见[实验](experiments.md#当前首跑c的颜色消费)。

当前没有hidden/KV对齐、教师logit KL、语义读出或GEN响应蒸馏。C1的caption KV是writer输入来源；不等于以其hidden作为蒸馏目标。后续扩展按[设计](design.md)新增teacher响应与损失，保留现有CE基线。

## 2. 张量与冻结合同

- 来源为同层原生旋转KV：文本走原生prefill；GEN来源为完整query，含图像边界token，不拼原prompt。
- 固定中性16个内容token生成每层seed；L重新取本层seed，C连续传hidden。
- 更新后hidden通过冻结UND norm/K/V/K-norm投影。bank的K未旋转，消费时旋转一次。
- Writer位置从2048起；来源重叠时报错。Caption Memory位置从0起，指令/答案从16起。
- GEN Memory接原prompt之后，query平移16；G-layout同样平移但mask新增KV。
- 原模型与VAE全部冻结，writer只注册新增残差。冻结UND保留输入梯度；首批检查每层Memory K/V与新增模块梯度。
- 微批1；`--distributed`通过HCCL同步writer梯度。全局batch均分各卡，loss除以全局batch后按SUM归约，再裁剪与更新；BAGEL不参与梯度通信。TaylorSeer禁用，无resume入口。
- 所有rank推进同一全局样本/t/noise随机流，各自执行按rank分配的微批；仅rank0做验证、记录模型hash和保存checkpoint，其他rank等待。每个拓扑独立联合训练，不混合L/C梯度。
- 可微诊断attention为显式mask的PyTorch SDPA；CUDA原生参照为FlashAttention，NPU沿用既有SDPA。目标后端数值容差未验证。
- 图像固定RGB bicubic resize到512；原生VAE scale/shift/patchify与posterior采样保留。posterior seed由内容hash派生，并与扩散、t、样本顺序随机流分开。

参数冻结只保证权重不变。新增KV仍影响归一化、位置和速度场，因此另做G-layout、反事实与画质检查。

## 3. 数据

每行JSONL必需`sample_id/scene_id/split/caption/label_type/verified/facts`。视觉clean数据需要`image`，GEN需要`prompt`。路径相对数据文件；ID只用字母、数字、下划线和连字符。

示例仅说明格式，不是核验数据：

```json
{"sample_id":"red_cube","scene_id":"cube_a","family_id":"cube_pair","pair_id":"color_pair","swap_id":"blue_cube","split":"test","image":"images/red.png","prompt":"A cube on a table.","caption":"A red cube on a table.","label_type":"clean","verified":true,"facts":[{"category":"attribute","aliases":["red"],"alternatives":["blue"],"unknown":false}]}
```

`family_id`绑定同图/场景族，`pair_id/swap_id`固定反事实；场景族/pair不可跨split，swap须同split和prompt。轨迹行用`state`指向实际t/step/seed/prompt状态，标签用核验后的`observed`。

事实类别为object/attribute/count/relation。`aliases`表示接受的断言，`alternatives`表示互斥错误，unknown保留词表以检查过度确定。

自动评分使用`locked_literal_assertion_v2`：按局部断言处理词表和否定，不能解析任意自然语言。保存自由文本供人工复核。当前checkpoint选择为验证集字面事实准确率优先、NLL次之；不按测试选择，不等价于画质。

## 4. 输出

|文件|内容|
|---|---|
|`manifest.json`|模型/源码/数据资产/config hash、后端、冻结/训练名单、seed IDs、状态和异常|
|`init.pt/best.pt/last.pt/optimizer.pt`|checkpoint与优化器审计；optimizer记录不代表支持resume|
|`train.jsonl`、验证目录|loss、梯度、实际t/seed、显存、共同选择规则|
|`captions.jsonl/caption_summary.json`|自由文本、NLL、逐事实、t/来源/标签与对照；bank另存|
|`gen.jsonl`、状态文件与图像|x_t、bank、conditional/CFG速度、局部统计、预测/可选最终图|
|`blind_annotations.csv/blind_key.json`|匿名标注与解盲映射；key不交给盲评者|
|`tables.json/report.md`|模型/数据/容量/源码/配置匹配后汇总的四表|

盲评target_facts填[0,1]准确率，quality填1–5；unchanged_facts/artifacts分别记录。空标注保留未评。

Source-swap与Memory-swap在单样本确定性writer下共用donor解码，记录等价关系，不重复算证据。t=1按noise_seed跨x0比较输入hash与自由输出。

## 5. 兼容性

旧checkpoint/C0记录缺少修正配置或复述指令token时拒绝加载；旧结果不能与修正主表混合。协议标识`memory_topology_diagnostic_protocol_20261011`保留为现有配置/记录ID，与文档简短文件名无关。

历史实现保留在older分支；清理清单见[归档](archive/cleanup.json)。实际验证与修正记录见[验证](validation.md)。
