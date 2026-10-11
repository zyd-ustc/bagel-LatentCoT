# 独立 Memory 诊断：实现与证据边界

日期：2026-10-11。状态：代码已构建；真实权重实验未执行。

## 分支与清理

远端没有精确名为 `loop-layer` 的分支。本地 `loop-layer` 从协议审计的 `loop-layer-NPU@42e7114` 创建。
切换前的 NPU 工作区存在大量未提交删除；修改前已备份到工作区的 `older/before-memory-topology-20261011/working-tree.tar.gz` 和 `local.patch`。
旧脚本、测试、配置、循环模块及对应运行文档的清单见 `legacy_cleanup.json`。原生模型代码及加载器保留。用户原有图示资产保留。

## 协议与代码对应

|协议|实现|
|---|---|
|固定中性16个内容token、各层S_l|`memory.neutral_seeds`；去掉已有特殊token后取固定文本前16个token；原生因果UND block生成所有层种子|
|L/C、仅新增残差可训练|`memory.MemoryWriter`；L按层重新取种子，C连续传hidden；瓶颈32、无bias、up零初始化；不注册decoder为writer子模块|
|更新后同层bank、消费时旋转|writer完整block输出加残差后，用冻结UND input norm/K/V/K-norm投影；`MemoryBank`拒绝已经旋转的K；`consume_bank`只旋转一次|
|来源保留专家/层/位置|`native.text_source`调用原生文本cache入口；`native.capture_gen`在attention输入捕获GEN原生Q/K/V投影，来源是当前完整GEN query，包含图像边界token，不拼原prompt KV|
|冻结decoder仍传输入梯度|`native.caption_logits`直接调用完整冻结UND block，保留输入梯度；不使用带no_grad的原生文本生成接口训练|
|C0/C1/C2、init/scratch|`caption.evaluate/train`与checkpoint阶段校验；C2默认要求各自C1，scratch必须显式标记；初始及末步checkpoint独立保存|
|加噪、pure-noise、真实轨迹|`data.controlled_noise`严格使用actual t，t=1显式返回eps；`trajectory.collect`记录采样节点真实t；两种来源单独报告|
|correct/off/source-swap/memory-swap/masked|Caption对照统一16槽位消费位置；单样本确定性writer使Source-swap与Memory-swap可以等价，仍分开记录；packed原语和整层bank交换有样本隔离测试|
|保留原prompt、G-base/layout/fixed/text/T*/V*/swap|`gen.diagnose_gen`；G-text在原prompt缓存后追加原生caption prefill；Memory不替换prompt；G-layout排除对应softmax项|
|一次GEN调用后释放、CFG|`native.gen_memory_call`仅拦截conditional一次；finally恢复所有方法；text-removed CFG保持原生；保存conditional与原生CFG最终velocity|
|同状态局部比较与后续采样|保留x_t/t，按G-layout记录距离/余弦/范数比，重复原生调用记录误差，并对比masked布局与原生仅平移query位置路径以记录后端等价误差；`runtime.continue_once`只首节点注入，余下按同一原生Euler时间表推进|
|四表、cluster CI、盲评|`report.summarize`；文本桥、视觉caption、GEN局部、最终事实/画质分表；按family/pair/scene聚类；没有标注不填质量结果|
|运行可追溯、冻结检查|manifest保存文件hash、源码commit及Python源码hash、模型配置、tokenizer/VAE、checkpoint、原数据资产hash、设备/dtype/后端、冻结/训练名单及seed IDs；每步检查原参数身份、in-place版本、requires_grad及grad|

## 明确的工程选择

- writer槽位固定在位置2048起。来源位置保持原生。来源与该位置重叠时停止，要求L/C共同预先重设位置。
- Caption Memory位置从0起，指令/答案从16起；masked也保持相同query位置。
- GEN Memory位置接在原prompt之后，GEN query位置按增加16个槽位平移。G-layout同样平移位置，屏蔽新增KV。
- 诊断writer/decoder/Memory reader统一用带显式mask的可微PyTorch SDPA。CUDA原生参照仍用FlashAttention，NPU原生参照用既有SDPA；实际后端写入manifest。
- CPU工程检查对BF16原生模块开启autocast；attention内部关闭CPU autocast，维持float32计算。目标GPU/NPU重复误差容差仍须以真实权重校准。
- 主执行入口微批为1，有效batch32通过梯度累积实现。packed attention、writer与整bank交换的原语支持并测试多个样本。
- 固定512图片用RGB bicubic resize和[-1,1]输入；VAE用原生scale/shift、patchify，posterior RNG单独隔离。该resize政策写入代码和配置，不代表高分辨率验证。
- 禁用TaylorSeer近似缓存。没有闭环、末层回送、GEN训练、可学习reader/gate或额外GEN→UND桥。

## 数据 JSONL

每行需要 `sample_id/scene_id/split/caption/label_type/verified/facts`。视觉clean行还需要image，GEN需要原prompt。
用 `family_id` 绑定同原图/场景族的所有变体；`pair_id` 和 `swap_id` 固定反事实关系。路径相对数据文件。sample_id只能含字母、数字、下划线和连字符。

以下仅示意格式，不是可用的真实核验数据：

```json
{"sample_id":"red_cube_a","scene_id":"cube_scene_a","family_id":"cube_family_a","pair_id":"cube_color_pair_a","swap_id":"blue_cube_a","split":"test","image":"images/red_cube_a.png","prompt":"A cube on a table.","caption":"A red cube on a table.","label_type":"clean","verified":true,"facts":[{"category":"object","aliases":["cube"],"alternatives":["sphere"]},{"category":"attribute","aliases":["red"],"alternatives":["blue"],"unknown":false}]}
```

事实类别固定为object/attribute/count/relation。`aliases`列出正确断言短语；`alternatives`列出互斥错误断言。unknown事实仍保留可被错误断言的词汇，用于统计unknown错误断言率。

自动事实分数采用预先固定的字面断言规则：匹配accepted短语，错误替代短语或否定不计为正确。它只支持受控、预定义事实词表，不能代替语义审查。自由文本与逐事实结果全部保存，允许人工复核。验证checkpoint规则明确为事实规则准确率优先、NLL次之；不是按测试集选择，也不宣称NLL代表画质。

同场景族和反事实pair不可跨split；swap对象必须在同split、同原prompt。观察轨迹行使用 `state` 指向含实际t、step、seed、prompt和state_source的文件，并使用核验后的observed标签。不会将clean与observed准确率合并。

## 输出与结果检查

- `manifest.json`：来源、参数、配置及执行状态；失败写入异常类型和信息。
- 训练：`init.pt/best.pt/last.pt/optimizer.pt`、`train.jsonl`、验证目录和共同选择记录。optimizer保存只用于审计，当前无resume入口。
- Caption：`captions.jsonl`、`caption_summary.json`、每状态未旋转bank、计时。t=1单独列出，并检查跨x0的相同eps输入及自由输出相同。
- GEN：逐组conditional/CFG速度、bank和x_t；预测图；`gen.jsonl`和计时；可选最终图和盲评表。
- 轨迹：actual t/step状态、conditional及CFG预测图、最终图、待核验观察标签。
- 汇总：`tables.json`和`report.md`。同模型、数据、容量、源码和配置才能合并。

`blind_annotations.csv`中的target_facts填[0,1]准确率，quality填1–5；unchanged_facts与artifacts独立记录。报告按同状态和seed对G-base/G-layout计算事实差、质量差及成对胜/平/负。空白标注保留为未评。

## 当前验证范围

新增测试直接使用小型随机初始化的原生Qwen2MoT/BAGEL模块。完整流程测试的tokenizer和VAE是测试替身。工程检查覆盖原生种子/GEN KV数值等价、L/C propagation、更新后读出、梯度、冻结检查、packed隔离、因果移位、mask、RoPE、pure-noise、checkpoint来源、C1→C2→GEN和trajectory。

本地未加载真实BAGEL权重，未验证目标NPU/CUDA后端的全模型数值容差、吞吐或峰值显存，未训练正式数据，未核验caption接地、GEN语义消费或画质。后续真实权重16样本工程检查必须先于质量解释。
