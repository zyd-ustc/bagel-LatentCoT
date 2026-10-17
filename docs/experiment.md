# Experiment

更新：2026-10-17。当前生成训练入口见第 7 节；前文保留此前实验记录。

[README](../README.md) · [原理](design.md)

## 1. 实验总览

| 项目 | 状态 | 结果位置或交付物 |
|---|---|---|
| Training-free memory 与层循环探索 | 已执行 | 旧层循环、连续 memory 与语义编辑的人工审查 |
| C0 原生文字校准 | 已执行 | 16 条 caption；原生文字读出 |
| C1 文本压缩与响应训练 | 已执行 | Caption 100 步；响应 100+1000 步 |
| 原生 GEN 梯度与布局 E0 | 已执行 | 单 NPU 梯度、布局、冻结参数检查 |
| 旧 C2-observe | 已执行 | 16 NPU、1000 步；固定来源读出对照 |
| 旧浅层生成与显式重生成 | 已执行 | 隐式 16 prompt 对照；显式 80 张终图 |
| 历史修正 C1 补训 | 已执行 | 16 NPU、100 步；修正读出及条件速度 |
| 索引训练主线重构 | 已实现并完成本地测试 | 118 项测试；原生小型 BAGEL 与 CPU 分布式 |
| 全量标注索引 | 显式 family 划分预读＋直接索引；全量运行单独记录 | 新旧来源、CoRT 修正/终图、任务视图 |
| 文本教师校准与终图评分 | 待执行 | 正确修正、保持与原生 Text 对照 |
| 新版 Observe / Feedback / C3 | 待执行 | NPU 运行、来源依赖与终图纠错 |
| 原生 GEN 适配与终图奖励 | 待执行 | memory 条件下的生成消费训练 |

历史运行主要位于 `modelarts-job`，远程仓库为 `/root/bagel-LatentCoT-loop-layer-NPU`。本地重构验证位于工作区 `artifacts/training_refactor_20261014/verification.json`。

### 轮次与输出术语

R1/R2/R3/R4 表示同一采样状态内 GEN 共运行一至四遍，分别对应零至三次反馈。训练入口的 `feedback_updates` 表示反馈次数。旧视觉训练记录中的 R1/R2 出口分别对应一次、两次反馈后的速度；下文保留该历史日志名称。

局部预测图使用当前速度形成的 x0 预测；窗口终图在指定外层时间窗口持续使用反馈。

## 2. 已执行：文本接口与生成响应

### 2.1 Training-free 探索

| 探索 | 人工审查或运行结果 |
|---|---|
| loop-layer 模式比较 | `legacy_layerwise` 在所测 training-free 方案中表现最好 |
| 独立 latent memory | 初期生成对 memory 内容的敏感性低，随后采用 prompt hidden 初始化 |
| 每层独立更新与连续 UND 更新 | 两类图像结果相近 |
| joint-micro 残差更新 | 首次反馈后的图像退化为噪点 |
| 中性种子下的浅层反馈 | 多数样本发生语义编辑，整体图像质量保持；正确编辑较少，方向控制是后续训练目标 |

### 2.2 C0 原生文字校准

原生 BAGEL 读取 16 条人工 caption 并自由复述。人工审查确认主要对象、颜色、数量和关系得到保留。

| 指标 | 结果 |
|---|---:|
| 完成样本 | 16/16 |
| 教师强制平均 NLL | 0.271115 |
| 严格文本匹配 | 7/16 |
| 字面事实分 | 15 条为 1；c0_08 为 0 |
| 含加载与汇总的端到端时间 | 54.0313 秒 |

c0_08 使用等价数量表述“single orange / one orange”，字面评分记为差异；c0_15 增加尺寸、材质和光照描述。

输出：`/root/outputs/c0_text16_20261012_111241`。

### 2.3 C1 Caption：独立层与连续更新

| 指标 | L 历史参照 | C 连续更新 |
|---|---:|---:|
| 第 1 步训练 NLL | 2.958457 | 2.976080 |
| 第 100 步训练 NLL | 0.430248 | 0.000001472 |
| best 步数 | 50 | 100 |
| best 验证 NLL | 1.336711 | 0.458495 |
| best 字面事实分 | 16.67% | 45.83% |
| 第 100 步颜色正确 | 4/8，均输出红色 | 8/8 |
| 第 100 步完整文本匹配 | 0/8 | 0/8 |
| rank 0 框架分配峰值 | 28.312 GiB | 28.415 GiB |
| 训练与中途验证计时 | 41.542 秒 | 51.208 秒 |

C 的自由文本仍有对象、承载面替换，8 条中 7 条将 desk 写为 table。后续文本实验使用 C 的第 100 步。

输出：`/root/outputs/c1_pilot_20261012_113553/C/best.pt`。SHA256：`d12b40adbf3f7aced96e51afef117fa8efda9ef42037f3e1b6147cac159a1ac2`。

### 2.4 文本 memory 的 GEN 对照

输入为 16 图组成的 8 个异场景交换对，共同 prompt 为 `An image.`。同图、seed、t 使用相同状态。输出为局部 x0 预测图。

| 对照 | 条件 |
|---|---|
| G-base | 原 prompt 与原生位置 |
| G-layout | 为 memory 预留位置，新增 KV 被屏蔽 |
| G-fixed | 全零、可见 memory KV |
| G-neutral | 固定中性种子 KV |
| G-text | 在原 prompt 后原生追加 caption |
| G-TC | C writer 将 caption 压成 16 槽 |
| G-swap-TC | 使用另一张图 caption 的 C memory |

蒸馏前，16 图×2 seed×7 臂得到 224 张图。下表是相对 G-layout 的平均速度差异。

| 对照 | 条件速度差异 | CFG 速度差异 |
|---|---:|---:|
| G-base | 0.01933 | 0.07735 |
| G-layout | 0 | 0 |
| G-fixed | 0.00563 | 0.02251 |
| G-neutral | 0.01607 | 0.06413 |
| G-text | 0.02502 | 0.09965 |
| G-TC | 0.01357 | 0.05424 |
| G-swap-TC | 0.01355 | 0.05416 |

4 图复核使用 t=0.7/0.8/0.9、4 个 seed，共 48 状态、336 张图。下表为 0–255 像素范围上的平均绝对差。

| t | G-text 与 G-base | G-TC 与 G-swap-TC | G-TC 与 G-neutral |
|---|---:|---:|---:|
| 0.7 | 4.4156 | 0.6599 | 1.8853 |
| 0.8 | 7.8226 | 0.8632 | 2.6719 |
| 0.9 | 21.8758 | 1.9538 | 4.8701 |

人工审查：G-text 有明显变化，其余臂视觉结果接近。输出为 `/root/outputs/gen_recheck_C_t789_20261012`；源图为 `/root/outputs/source_sample16_20261012_121337`。

### 2.5 响应蒸馏 100 步

2026-10-12，提交 `788de2e`。从 C1-C 第 100 步初始化；16 NPU、全局 batch 16、学习率 1e-5。损失为 response＋0.1×caption NLL。响应读取第 0/13/27 层、每层 64 个 query，预留 64 个条件位置。

文本划分为 64/8/8；query 图为 12 train、4 val。每 rank 缓存一张图在三个 t 上的探针，重复图使用不同 seed，验证 seed 独立。

| 步数 | 验证 response | 验证 NLL | 字面事实分 | 选择损失 |
|---|---:|---:|---:|---:|
| 0 | 0.044724 | 0.458495 | 45.83% | 0.090574 |
| 50 | 0.033690 | 0.395288 | 58.33% | 0.073219 |
| 100 | 0.020588 | 0.752646 | 66.67% | 0.095852 |

选择损失选中第 50 步；第 100 步用于图像审查。相同 4 图×4 seed×3 t 的 336 张图中，人工审查发现 G-TC 在部分样例上与 Base、Swap 拉开差距。

训练输出：`/root/outputs/c1_response_20261012_134848`。第 100 步 SHA256：`725ab4a4cc33a3fa68cfaa1283905948dd0d885a397c4ceeb724009045e2c2ce`。生成输出：`/root/outputs/gen_response100_t789_20261012`。

### 2.6 扩展响应训练 1000 步

2026-10-12，提交 `1f2b614`。从响应第 100 步加载 writer，重新创建 AdamW，新增 1000 步，累计响应 1100 步。16 NPU、全局 batch 16、学习率 1e-5、每 250 步验证。

文本为 2112 train、40 val、40 test，其中原模板为 64/8/8，新增自然文字为 2048/32/32。自然文字来自 CoRT36k 短 prompt，最多 40 tokenizer tokens。128 张训练 query 图提供每卡 8 图×3 t，共 384 份探针；验证沿用原 4 图。

| 新增步数 | 累计步数 | 验证 response | 验证 NLL | 模板事实分 |
|---|---:|---:|---:|---:|
| 0 | 100 | 0.032839 | 4.789251 | 66.67% |
| 250 | 350 | 0.024535 | 0.482668 | 100% |
| 500 | 600 | 0.021407 | 0.279060 | 100% |
| 750 | 850 | 0.019490 | 0.221803 | 100% |
| 1000 | 1100 | 0.018171 | 0.163620 | 100% |

末步模板 response/NLL 为 0.013619/0.000258，自然文字为 0.019309/0.204460。选择损失为 0.034533，best 为新增第 1000 步。32 个状态、224 张局部预测图的人工审查认为 G-TC 与 G-text 效果基本持平。

Checkpoint：`/root/outputs/c1_response_expand_20261012_141729/best.pt`。SHA256：`255aa6d7c5ffed326a554891e2189dbbd2bdc309fe05c6433b10258a095e3f56`。

## 3. 已执行：视觉读取与显式反馈

### 3.1 Ascend E0

2026-10-12，提交 `92a17f5`，单 NPU。初始化使用累计响应 1100 步的 checkpoint。测试图为 `sharegpt4o_1117`、`sharegpt4o_42006`。

| 检查 | 结果 |
|---|---|
| 原生与可微条件速度，t=0.25/0.75/0.9 | 最大绝对差均为 0 |
| 屏蔽 memory 与平移 16 位置的原生参照 | 最大绝对差均为 0 |
| 两次完整反馈反向，t=0.75、seed 43 | 输入梯度与冻结参数检查通过 |
| 两次工程更新的 flow MSE | 0.237012、0.236920 |
| BAGEL/VAE 前后参数 SHA256 | `c50b0a68abad7914aa9fbe5a4fff450e2bb4453c54651f6b09b7039f0f06d1f2` |
| 峰值 NPU 内存 | 38,551,704,576 bytes，约 35.9 GiB |
| 检查计时 | 28.83 秒；加载、manifest、首次参数哈希在计时外 |

布局检查首跑有 0.4% 元素超过容差，最大差 0.0625；屏蔽臂改用原生 attention 后通过。第二次运行在恢复冻结 writer 时触发版本检查，随后仅恢复被更新的视觉参数。通过输出为 `/root/outputs/visual_loop_e0_pass_20261012/e0.json`，两个失败输出保留在对应历史目录。

### 3.2 旧 C2-observe 与来源对照

16 NPU 完成 1000 步，仅优化语义损失；当时的文本 writer 与基座冻结。固定 64 个 val 场景、4 个 seed，选择使用 1536 个非纯噪声状态。

| 步数 | 验证语义 NLL |
|---|---:|
| 250 | 3.579268 |
| 500 | 3.469990 |
| 750 | 3.399632 |
| 1000 | 3.336412 |

Checkpoint：`/root/outputs/visual_observe_full_20261012/best.pt`。SHA256：`3d9cc1f4f728cedbaccafedcb15f2c999ccf846e49fe365b7149f53e36ce303a`。Best 与 last 均为第 1000 步。

2026-10-13 使用 4 个 val 场景、t=0.25/0.75/0.9、seed 0，生成 Correct、Source-off、Source-swap 共 36 条自由 caption。样本为 `visual_sharegpt4o_5178`、`visual_sharegpt4o_32408`、`visual_sharegpt4o_11204`、`visual_mj_v6_122908`，相邻两项互为交换来源。

12/12 交换状态哈希与供体匹配，12/12 交换 caption 与供体 Correct 相同；同 t 的关闭来源输出相同。人工审查发现低 t 的描述与图像语义相关，t>0.75 后出现模板化与无关文字。

输出：`/root/outputs/visual_source_review4_20261013`。

### 3.3 旧浅层生成

使用旧 Observe checkpoint，中性种子、GEN 0–7 层来源；窗口之后 UND 继续以 memory 自注意力更新。生成保留原 prompt。R2/R3/R4 与 early10/20 组成六个组合，加 Base 后每 prompt 共七张终图。

默认从 `data/prompts32.jsonl` 选择 16 条，seed 0、512 分辨率、50 个节点、shift 3、CFG 4，执行 49 次 Euler 更新，完整对照为 112 张图。单步 R2/R3/R4 加载检查输出在 `/root/outputs/shallow_memory_smoke_20261013`；后续完整结果为 `memory16_t2i_R234_early1020_20261013_142707`。

人工审查记录：语义编辑广泛出现，修改方向和正确编辑率是主要问题。当前重构已将正式 writer 来源统一为 P+GEN。

### 3.4 显式固定 memory 重生成

已核对产物：工作区 `artifacts/explicit_feedback_restart16_20261014_142513`。该运行先完成 Base，再用 BAGEL 反思，从同一精确初始噪声重生成，在 early10/20 固定消费同一份文字 memory。

| 项目 | 已执行内容 |
|---|---|
| 输入 | 16 prompt、seed 0，对应 148 个 VQA 问题 |
| 文字 writer | 累计响应 1100 步 checkpoint |
| 反馈来源 | 16 条 `bagel_auto` |
| 输出 | Base＋两个窗口各自的 Layout-memory/Text-memory，共 80 张 |
| Base 一致性 | 与旧浅层运行的 16 个 Base 文件哈希一致 |
| 人工审查 | Text-memory 能产生明显语义编辑 |
| 数值评分 | 待执行 |
| 原生 Text / Layout-text | 待执行 |

自动反思中已识别的目标冲突：

| 样本 | 原目标 | 自动修正 |
|---|---|---|
| `full_diag_020` | seven cars、four dogs | seven cars、seven dogs |
| `full_diag_024` | four white bicycles、three plastic cows | four white bicycles、four plastic cows |
| `full_diag_016` | turtle、three elephants、glass monkey 及跳跃关系 | 改为 two elephants，并改变关系和属性 |
| `full_diag_029` | bear behind seven metal trumpets | bear in front of guitars and trumpets |

当前将 Text-memory 登记为显式教师参照。后续分开测量自动反思、正确文字、16 槽压缩与 GEN 消费的贡献。

### 3.5 历史修正 C1 补训

2026-10-14，输出 `/root/outputs/c1_historical_feedback_20261014_163638`。从累计响应 1100 步初始化，P+C 读出修正并校准原生追加文字的完整条件速度；原生模型冻结。

16 NPU、全局 batch 16、学习率 1e-5、100 步；时间取 0.05/0.25/0.5/0.75/0.9。训练使用 pilot 的 640 train、40 val、40 test。首步 28 层 memory K/V 梯度与基座冻结检查通过，进程退出码 0。

| 指标 | 值 |
|---|---:|
| 第 50 步验证 NLL，40 条 | 0.2703438 |
| 第 100 步验证 NLL，40 条 | 0.1857517 |
| 终步训练总损失 | 0.3683133 |
| 终步训练 NLL | 0.3661245 |
| 终步条件速度 MSE | 0.0021888 |

Best 与 last 均为第 100 步，best 依据文本验证选择。事实词表为空，结果字段记为 `null`。Checkpoint 的显式生图对照列入下一节。

SHA256：`e7c3bce91f2d9ba2697fb06308d6eb6850aa463518a7d9a2e6d19c38ff2c1ed5`。运行目录保存 `trainer_caption.py`。

## 4. 当前实现与已执行的工程验证

### 4.1 训练架构

| 层次 | 文件 | 职责 |
|---|---|---|
| 数据基座 | `corpus.py`、`prepare_training_corpus.py` | 新旧标注注册、family 划分、归一化、任务分片及索引 |
| 任务与时间 | `curriculum.py` | Observe/Endpoint/Generation/Trajectory/Conditional 配额、噪声课程、覆盖游标 |
| 原生状态 | `runtime.py`、`flow_training.py` | 文本与 VAE 条件、实际噪声、合法 flow 目标、可微 GEN |
| 表示与来源 | `visual.py`、`memory.py`、`feedback_source.py` | 原生 UND 参数控制、全层 P+GEN 来源 |
| 反馈与损失 | `loop_training.py`、`visual_objectives.py` | 固定状态循环、逐轮新来源、语义/教师/flow |
| 训练与恢复 | `visual_training.py`、`visual_record.py` | 单一优化器、全局批次、分桶梯度同步、格式 4 原生 UND checkpoint |
| 验证 | `task_evaluation.py`、`visual_evaluation.py` | 有限 family 对照、CFG 后速度、完整后缀和窗口生成 |
| 入口 | `visual_cli.py`、`visual_training_npu.sh` | Observe、Feedback、C3 共用训练入口 |

主损失为 `sem / teacher / flow`。原生 UND 从 BAGEL 基础权重开始训练，后续阶段使用格式 4 checkpoint。C1 与 attention response 的结果保留为历史实验。

### 4.2 2026-10-14 本地验证

完整测试 118 项通过，耗时 10.87 秒。原生 BAGEL 的 28 个源码文件与重构前备份一致。

| 已验证项目 | 验证内容 |
|---|---|
| 原生计算 | 小型 BAGEL 布局、速度、GEN 输入梯度与冻结参数 |
| 原生条件图 | VAE prefill；文字与图像 CFG 分支；writer 的独立 P 来源 |
| 监督路由 | A 端修正、B 端保持/flow、高噪声端点 mask、纯噪声诊断 |
| 数据读取 | JSON 数组跨块解析、分片索引、全局 family、占位图排除、长目标 |
| 采样与恢复 | A/B 共 seed/t、任务配额、来源覆盖、恢复后与连续运行一致 |
| 分布式 | 两进程全局梯度缩放、同步与验证 family 分配 |
| 终图验证接口 | 监督状态与原生图像节点分别记录，完整 CFG 后缀 |

21 份新数据的 840 条既有审计样本均完成归一化。训练脚本语法与两个主入口的帮助命令通过检查。

### 4.3 2026-10-15 显式反馈验证

显式推理由 `explicit_feedback.py` 与 `feedback_cli.py` 完成：完整 Base、BAGEL 文字反思、固定 memory、相同初始噪声重生成。配置为 `explicit_feedback.json`，协议标识为 `explicit_feedback`。

完整测试 112 项通过，耗时 10.70 秒。固定 memory 的采样、初始噪声、Base 和时间窗口计算保持一致。现有 16 样本、32 个 early 窗口结果通过图像与逐步记录审计。

### 4.4 2026-10-15 标注直接索引

构建器改为只读标注，单遍生成记录分片与任务索引。取消图像读取、资产检查、图像哈希、样本去重、SQLite 和临时记录二次整理。BAGEL token 长度批量计算；每来源输出进度。来源采样按任务记录数平方根加权。

Family 划分修复后，构建器先读取配置了 `split` 或 `inherit_split` 的来源，收集显式 family 划分，再生成记录与索引。同 family 的其他来源沿用该划分；显式划分冲突时停止构建。未配置显式划分的来源仍只读取一次。索引格式更新为 `annotation_index_v2`；旧索引需要重新构建。

CoRT36k 保留原始划分；CoRT114 修正链、visual-state-only、stage12 合并清单、旧 manifest、nofix、entry 和配对记录全部注册。终图描述进入 Observe，修正进入 Feedback；未声明正确终点的图像保留其原有监督范围。重复记录分配独立 occurrence ID。

本地完整测试 115 项通过，耗时 10.87 秒；包括无图像访问、重复保留、CoRT 路由和训练恢复。远端 56 个实际来源的小样本验证生成 1,817 条记录、5,066 条任务索引，图像访问为 0；验证报告位于 `/root/outputs/annotation_index_smoke_g6awdem0/verification.json`。

本次审查修复后，本地完整测试 124 项通过，耗时 11.27 秒。新增 9 个用例覆盖 family 别名的来源顺序与划分冲突、旧索引重建要求、浅层入口的 writer/query 加载与新版元数据、Observe 的监督资格。完整模型与 NPU 验证待执行。

## 5. 待执行：下一轮实验

| 顺序 | 实验 | 起始设置 | 交付 |
|---|---|---|---|
| 1 | 全量数据整合 | 全部新旧标注，保留重复 | 来源、记录数与任务索引报告 |
| 2 | 新版 NPU E0 | 原生条件、Observe、Feedback 各一条完整反向 | 时间、峰值内存、冻结与梯度记录 |
| 3 | 文字教师校准 | 256 个同目标修正/保持 family；自动与核对反馈分别运行 | 反思、Text/Text-memory 终图与分项评价 |
| 4 | Observe | 训练原生 UND 注意力，从低噪声开始 | 分 t 与来源窗口的文字读出、关闭/交换来源对照 |
| 5 | 单次 Feedback | 端点、正确图 flow、真实状态与 Observe 辅助 | 修正、保持、视觉质量与来源依赖 |
| 6 | C3 | 新来源、新反馈；逐遍直接监督 | 新/旧/固定反馈对照与各遍终图 |
| 7 | 原生 GEN 适配 | 固定可靠 memory，训练原生消费参数 | 纠错成功率、保持、质量与跨 seed 稳定性 |
| 8 | 终图奖励 | 在已有效的闭环上定义轨迹与终图目标 | 奖励、采样和参数更新方案 |

### 5.1 起始训练设置

| 项目 | Observe | Feedback | C3 |
|---|---|---|---|
| 配置 | `visual_memory_pilot.json` | `visual_feedback.json` | `memory_loop_r2.json` |
| 计划更新数 | 20000 | 20000 | 后续生成阶段 |
| 全局 batch / 学习率 | 16 / 1e-5 | 16 / 1e-5 | 16 / 1e-5 |
| 训练参数 | 原生 UND 注意力 | 按 `und_train` 选择 | 按 `und_train` 选择 |
| `sem / teacher / flow` | 1 / 0 / 0 | 1 / 0 / 0 | 按生成实验确定 |
| GEN 来源层 | 全部 28 层 | 全部 28 层 | 全部 28 层 |
| 反馈次数 | 描述恢复 | 1 | 2 |
| 初始端点 t 上限 | 0.25 | 0.25 | 0.25 |
| 后续文字反馈 | 文字读出 | 单次修正/保持 | 当前完整预览重新反思 |

端点 Observe/Feedback 的实际 t 从 0.05 到校准上限抽取；C3 在线预览使用该范围内的原生节点。课程扩至 0.5 时，同步更新对应任务的校准上限。

C2 Feedback 的任务配额为 Observe 15%、Endpoint 85%。Generation、Trajectory、Conditional 的生成监督留在后续阶段。来源权重取可用任务记录数的平方根，A/B 两次抽样共同计入配额。

生成 flow 的原生节点配额为 0–19 占 60%、20–34 占 20%、35–48 占 20%。真实轨迹使用保存的实际节点。全部 UND 层读取同层 GEN KV。

### 5.2 原生时间与反馈窗口

50 节点、shift 3 产生 49 次 Euler 更新。early10/early20 指前十/二十次更新。

| 节点索引 | 实际 t，约值 |
|---|---:|
| 0 | 1.000000 |
| 1 | 0.993103 |
| 9 | 0.930233 |
| 10 | 0.921260 |
| 19 | 0.825688 |
| 20 | 0.813084 |
| 35 | 0.545455 |
| 40 | 0.402985 |
| 44 | 0.254237 |
| 48 | 0.058824 |

固定 memory 在 early10/20 使用至 t≈0.92126/0.81308。后续时间窗口比较按实际 t 记录，起始候选为 0.8–0.95、0.6–0.8、0.4–0.6。

### 5.3 对照与评价

| 问题 | 配对对照 | 评价 |
|---|---|---|
| 反馈内容是否有效 | Base、对应 Layout、原生 Text、Text-memory | 目标事实修正、正确事实保持、图像质量 |
| Memory 是否读取生成状态 | Correct、Source-off、同 P/t 的 Source-swap | 描述/修正正确率、保持建议、来源依赖 |
| 新一轮反馈是否有用 | New、Old、Fixed、Swap | 各遍终图、错误变化与额外计算 |
| 能力受哪个环节限制 | 自动/核对修正；Text/Text-memory/连续 memory | 反思、压缩、消费与时间窗口的分项差异 |

Feedback 的 Source-off 保留 P；来源交换保留接收方状态和 P，仅更换 GEN 来源。文本与 memory 使用各自长度对应的 Layout。已满足目标的显式样本沿用原生结果。

校准集起始为 256 个 family，覆盖数量、属性、对象增删、位置/关系和文字任务，局部编辑与全局改写分组。初始扩区标准为 Correct 比 Source-off 提高至少 10 个百分点，保持样本的错误修改建议率至多 10%。

Observe best 依据读取 NLL。C2 Feedback best 依据 repair/hold 两类 NLL 的平均值，文字正确性另做原图审阅。后续生成阶段使用固定事实修正、保持和质量评价选择终图。自动反思与核对反馈分别汇总，early10/20 分别汇总。

## 6. 运行入口

以下命令在远程仓库根目录执行。环境使用 `/root/venvs/bagel-NPU`，模型为 `/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT`。数据构建只扫描标注，直接生成记录分片和任务索引；保留重复记录，图像在训练取样时读取。

### 6.1 显式固定 memory 重生成

```bash
bash scripts/explicit_feedback_npu.sh /root/outputs/explicit_feedback_new --text-control
```

脚本选取 16 个 prompt，以 16 个 NPU worker 运行。默认 Base、early10/20 的 Layout-memory/Text-memory；`--text-control` 增加 Text 与 Layout-text。每个外层更新执行一次完整条件 GEN，窗口内复用同一 memory。

人工核对采用两次运行：先保存 Base 与模板，再填写反馈并运行对照。`PROMPTS` 指向准备好的推理输入；初始行使用 `sample_id`、`prompt`、`noise_seed`。设置模型、文字 checkpoint、输入清单及两次新输出路径后执行：

```bash
export BAGEL=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
export C1=/root/outputs/c1_response_expand_20261012_141729/best.pt
export PROMPTS=/absolute/path/to/feedback_inputs.jsonl
export PREVIEWS=/root/outputs/feedback_previews_new
export CHECKED_OUTPUT=/root/outputs/feedback_checked_new

python -m qwen_latent_cot.diagnostics.feedback_cli \
  --model "$BAGEL" --text-checkpoint "$C1" \
  --config configs/explicit_feedback.json --data "$PROMPTS" \
  --device npu:0 --output "$PREVIEWS" --prepare-only

python -m qwen_latent_cot.diagnostics.feedback_cli \
  --model "$BAGEL" --text-checkpoint "$C1" \
  --config configs/explicit_feedback.json \
  --data "$PREVIEWS/feedback_template.jsonl" \
  --device npu:0 --output "$CHECKED_OUTPUT" --text-control
```

模板保留精确 `initial_state`、prompt 与 seed，填写 observation、gap、correction 或 satisfied。每次输出使用新目录。

输出包括精确初始噪声、采样节点、checkpoint 哈希、反馈文本、memory、条件与 CFG 速度的哈希及范数、每步注入记录、分项时间、终图、`review.html` 和 `annotations.csv`。初始噪声保存在 `*_initial.pt`。

### 6.2 Observe → C2 Feedback

在 `modelarts-job` 的 `/root/bagel-LatentCoT-feedback_20261016` 运行：

```bash
# 两步 16 卡 smoke，独立输出，不启动正式训练。
bash scripts/train_feedback_nativeund_npu.sh smoke /root/outputs/feedback_nativeund_smoke_new

# 从 Observe best 开始正式 C2，重置优化器、步数和采样进度。
bash scripts/train_feedback_nativeund_npu.sh train /root/outputs/feedback_nativeund_attn_full_20k
```

入口默认 16 NPU，启用 W&B online。`OBSERVE_PARENT` 可指定另一个原生 UND checkpoint；默认使用 `/root/outputs/observe_nativeund_attn_20261015/best.pt`。`NPROC_PER_NODE=1` 选择单卡 smoke。数据入口使用 `TRAINING_CORPUS`，默认沿用当前完整索引池。

同阶段续训使用 `VISUAL_RESUME` 恢复优化器和采样进度；跨阶段使用 `VISUAL_PARENT` 初始化原生 UND 权重。C2 不使用旧 C1 初始化或文字教师。

Observe 继续训练可从其 last 恢复，并提高总步数。生成训练使用第 7 节的交互 SFT / RL 入口。

### 6.3 评分与审计

评分工具为 `scripts/score_explicit_feedback.py`；已有显式实验的评分状态为待执行。固定 memory 的窗口与终图审计使用：

```bash
python -m qwen_latent_cot.diagnostics.feedback_audit \
  --run /absolute/path/to/run --output /absolute/path/to/audit
```

评分记录事实修正、正确事实保持、图像质量和逐 seed 稳定性；运行记录保存模型/数据/反馈来源与分项计算量。

### 6.4 W&B 与训练日志

新版 Observe、Feedback、C3 共用日志入口。训练脚本默认开启 W&B online，项目为 `bagel-LatentCoT`。Tracking 依赖固定 W&B 0.22.3，使用原环境的 protobuf 4。在 `modelarts-job` 使用训练环境安装并登录：

```bash
/root/venvs/bagel-NPU/bin/python -m pip install -e '.[tracking]'
/root/venvs/bagel-NPU/bin/wandb login

export WANDB_PROJECT=bagel-LatentCoT
export WANDB_MODE=online
# 团队项目设置 WANDB_ENTITY；省略时采用登录账号的默认 workspace。
# export WANDB_ENTITY=your-team

bash scripts/visual_training_npu.sh observe /root/outputs/observe_wandb_new
```

W&B API key 在登录终端输入。训练启动命令沿用 6.2 节的模型、corpus、C0 与 checkpoint 设置。`WANDB_MODE=offline` 将运行保存在本地，`disabled` 关闭 W&B；直接调用 `visual_cli` 时默认 disabled，可用 `--wandb-mode` 明确启用。

只有全局 rank 0 创建 W&B run、上报指标和写训练 JSONL。损失为所有 rank 汇总结果，步耗时和内存取各 rank 最大值。每步保存总损失与 sem/teacher/flow 分项、梯度、学习率、实际噪声、任务覆盖、计算量和性能；验证按任务记录 NLL、来源关闭对照、事实读出与对应损失。Console 默认每 10 步输出，首步、末步、验证和 checkpoint 保存时也输出，`--log-every` 调整 console 频率。

每次启动创建新 run，名称默认采用输出目录名。checkpoint 恢复记录父路径与恢复步数，继续训练步数；W&B run 独立于此前运行。初始化清理旧的 run ID、resume、group 和 sweep 身份。运行元数据采用当前 sem/teacher/flow 目标，取消旧 attention-response 教师声明。

```text
run/
├── config.json / manifest.json / training_plan.json
├── training.jsonl          # 全局训练记录，每步一行
├── validation.jsonl        # 全局验证标量，每次验证一行
├── training_summary.json  # completed/failed、实际完成步数、耗时及错误
├── logs/rank_00.log       # 各 rank 的文本日志；rank 0 同时输出 console
├── wandb.json             # 本次 run 的 ID、项目、模式与在线链接
├── wandb/                 # SDK 本次运行文件
└── validation_*/          # 原始逐样本结果与终图
```

训练输出目录使用新路径。指标与终图质量分别记录；Observe 的 NLL 用于读取模型选择，C2 Feedback 使用两类反馈 NLL 的平均值选择 checkpoint；后续生成阶段采用终图评价。实现使用 [W&B 官方日志接口](https://docs.wandb.ai/guides/track/log/)。

2026-10-15 本地完整验证：130 项测试通过，耗时 20.82 秒。包含真实 W&B 离线训练、checkpoint 继续训练的新 run、两进程单 run、失败步数记录，以及浅层入口的 checkpoint writer 保持。


## 原生 UND 全层 Observe 重构（20000 步已完成）

### 架构

删除 block 后的低秩写入残差和带时间条件的 query 残差。Memory 从固定中性文字的 16 个槽位开始，连续经过全部 UND block。每层读取同层 GEN query KV，并通过原生 UND 的归一化、K/V 投影导出 memory。Observe 不向 writer 提供目标 prompt；Feedback 在所有层同时提供目标 prompt KV 与 GEN KV。噪声时间通过 GEN 来源体现，不新增时间编码模块。

UND 的参数由 memory 写入和文字读出共享。GEN 专家、VAE 和速度头冻结；GEN 来源作为停止梯度的输入。UND 参数更新也会影响原生生成过程中的文字条件编码，这属于共享 UND 的实际行为。

### 参数控制

使用 `und_train` 选择层范围，并分别控制 Q/K/V/O 投影、Q/K norm、两个 block norm、MLP gate/up/down 投影、最终 norm、embedding 和 LM head。每项独立开关。选中的原生参数以 float32 优化，前向使用已有混合精度。Checkpoint 保存原生 UND 参数、优化器和采样状态，架构版本更新，不迁移旧残差权重。

### 首次实验

从 BAGEL 原生权重开始；全部层参与计算并读取 GEN。只训练全部层注意力的 Q/K/V/O 投影与 Q/K norm，其余 UND 参数冻结。使用全量 Observe 数据池、16 卡、全局 batch 16、学习率 1e-5、低噪声 t∈[0.05,0.25]、20000 步。每 250 步验证同一批状态，比较原图、自由描述、Source-off 与 NLL。

### 实现与启动顺序

1. 更新 memory、原生参数选择、checkpoint 和训练入口。
2. 开启 UND block 的激活重计算，并按桶同步梯度。
3. 运行小规模参数与梯度 smoke test，检查 checkpoint 恢复。
4. 在 NPU 运行短程 smoke，确认有限 loss、注意力梯度和冻结参数状态。
5. 启动正式 16 卡训练，记录运行目录、日志和 W&B 链接。


首次实验的注意力开关包括 `q_proj/k_proj/v_proj/o_proj/q_norm/k_norm`，全部层打开。其余开关为 `input_layernorm/post_attention_layernorm/mlp_gate_proj/mlp_up_proj/mlp_down_proj/final_norm/embeddings/lm_head`，首次全部关闭。`und_train.layers` 可设为 `"all"` 或层号列表；训练层选择不改变全层读取与计算。

运行入口：`bash scripts/train_observe_nativeund_npu.sh`。入口先执行两步 16 卡 smoke，再恢复其优化器与采样状态继续到总计 20000 步。正式验证使用原有 64 family 预算。Checkpoint 保留 best、last 和最近两个 step 文件。


### 2026-10-15 实际运行

主机：`modelarts-job`。代码目录：`/root/bagel-LatentCoT-nativeund_20261015`。数据沿用当前完整索引池，Observe train 为 54,589,431 条。

本地原生 UND smoke 与两进程同步检查共 5 项通过。NPU 两步 smoke、6 条描述/Source-off 验证和 checkpoint 保存完成，输出为 `/root/outputs/observe_nativeund_attn_20261015_smoke`。正式任务恢复 smoke 的第 2 步优化器与采样进度，从第 3 步继续到 20000 步。

正式输出：`/root/outputs/observe_nativeund_attn_20261015`。启动日志：`/root/outputs/observe_nativeund_attn_20261015.launch.log`。W&B：[observe_nativeund_attn_full_20k](https://wandb.ai/zyd20040407-ustc/bagel-LatentCoT/runs/rx2nomt3)。

原生注意力可训练参数 822,219,776 个，共 252 个参数张量。第 46–48 步各注意力组梯度均非零且有限，单步约 1.24–1.38 秒，框架分配峰值约 41.7 GiB。图像描述质量由后续固定验证状态与原图对照评估。


### 2026-10-16 Observe 结果

正式训练完成 20000 步，耗时 43,734 秒（12 小时 9 分钟）。全程低噪声 t∈[0.05,0.25]，累计 320,000 次采样，来自 54,589,431 条 Observe train 视图。原生 UND 全层注意力训练，全层读取 GEN。

| 步数 | 验证 NLL | Source-off NLL | 来源收益 |
|---|---:|---:|---:|
| 1000 | 2.5865 | 2.8402 | 0.2537 |
| 5000 | 2.3280 | 2.7551 | 0.4271 |
| 10000 | 2.2103 | 2.7328 | 0.5225 |
| 19750，best | 2.0381 | 2.7183 | 0.6802 |
| 20000，last | 2.0481 | 2.6919 | 0.6438 |

固定验证切片为 113 条。验证 NLL 总体下降，第 20000 步相对 best 有小幅回升；后半程仍有改善：10000→20000 步 NLL 下降约 7.3%，15000→20000 步下降约 3.1%，保留继续 Observe 的实验价值。best 为 `/root/outputs/observe_nativeund_attn_20261015/best.pt`，对应 19750 步；last 为同目录 `last.pt`，对应 20000 步。继续 Observe 使用 last 恢复优化器与采样进度；新阶段 C2 Feedback 使用 best 初始化权重，并新建优化器与采样进度。

自由描述中，以人物开头的输出从旧残差模型的 106/113 条降为本轮 best 的 28/113 条。原图审阅观察到角马群、浴室、鹦鹉和自行车等主体读取改善；仍有加油机认成消防栓、狗的数量遗漏、颜色/动作补写和图中文字编造。记录为主体级读取进展，细类、属性、数量和关系继续评估。

审阅文件：工作区 `artifacts/observe_nativeund_results_20261016/review.html`，含 24 个跨来源样本、原图标注、旧残差、原生 UND 各阶段与 Source-off。

### C2 Feedback 准备

C2 学习 UND 的目标比较与文字反馈：每层读取目标 prompt KV 与同层 GEN KV，输出正向修正，或 `No change is needed.`。使用 Observe best 初始化，继续训练全部层 Q/K/V/O 与 Q/K norm，保持 MLP、embedding、LM head、GEN 专家和 VAE 冻结。

首轮配置：20000 步、16 卡、全局 batch 16、学习率 1e-5、t∈[0.05,0.25]。损失仅文字交叉熵；Observe 占 15%，Endpoint 占 85%。Endpoint A/B 在相同 t 与噪声 seed 下分别监督修正与保持。文生图数据继续进入 Observe 辅助任务。当前索引 train：Endpoint 37,844,468 条，Observe 54,589,431 条；真实 trajectory 视图当前为空。

C2 直接完成 GEN 来源提取、memory 写入和文字读出，不计算第二遍 GEN 速度，不运行终图后缀，不使用旧 C1 教师。生成侧适配保留在后续阶段。验证记录修正/保持标签、自由反馈、Source-off 与同目标来源交换；分别记录 repair/hold NLL，以两类 NLL 的平均值选择 best，同时保留 Observe NLL 观察读取能力。

当前 CoRT-114 stage11 来源引用的 `cort_sft_133k/intermediate/sharegpt4o/sdxl` 目录不存在，前次审阅也记录了缺图。本轮通过 `excluded_sources` 排除该来源的 57,858 条 train Endpoint，保留其余 CoRT-36k 修正链。C2 实际 Endpoint train 为 37,786,610 条。原索引与 Observe 的数据快照保持原样。

状态：本地 8 项 smoke/日志检查通过。首次 16 卡 smoke 因另一个采样进程占用物理卡 0/1 而显存不足；随后在空闲物理卡 2/3 上完成双卡 smoke，全局 batch 16。

NPU 输出为 `/root/outputs/feedback_nativeund_prepare_smoke_2npu_20261016`：从 Observe best 初始化，训练 2 步，六组注意力梯度均非零且有限，loss 为 2.1456、1.5799，峰值约 42.1 GiB。每步 16 条样本对应 16 次 GEN 来源提取，未运行第二遍 GEN 生成。31 条验证记录包含 17 条反馈与 14 条 Observe；checkpoint 保存完成，总训练/验证/保存计时约 130 秒。

当前 smoke 的自由反馈全部为保持：hold NLL 0.0428、repair NLL 4.0095、错误保持率 1.0。正式训练监控 repair/hold 分项、错误保持率与来源交换，使用平衡 NLL 选择 best。此 smoke 权重不作为正式 C2 的初始权重；正式训练重新从 Observe best 开始。

远端准备目录为 `/root/bagel-LatentCoT-feedback_20261016`。正式 16 卡入口已准备，当前未启动正式 C2；物理卡 0/1 仍由另一采样任务占用。后续 smoke 的验证行数限制为 8，控制检查开销。


## C2 第 2000 步生成 pipeline

Checkpoint 固定为 `/root/outputs/feedback_nativeund_attn_full_20k/step_002000.pt`。生成代码部署于 `/root/bagel-LatentCoT-gen_c2_2000_20261016`，入口为 `scripts/shallow_memory_npu.sh`。

| 参数 | 语义 | 示例 |
|---|---|---|
| `--early` | 前 N 次原生 Euler 更新消费 memory，或全部更新 | `10 20 30 full`；也接受 `early10` |
| `--loop-layers` | 每个非末遍重复的 GEN 前缀 block 数 | `8 16 28`，或 `full`；8 表示 block 0–7 |
| `--passes` | 当前节点内 GEN 总遍数，反馈次数为该值减一 | `2 3 4` |
| `--seed` | 所有对照共用的初始噪声 seed | `0` |
| `--selection-seed` | 固定本次抽取的 prompt | `0` |
| `--count` | 本次 prompt 数 | `16` |

三个参数轴构成笛卡尔积，额外生成一张 Base。同一 prompt 的各对照使用相同初始噪声。默认模型、分辨率、CFG、重归一化、timestep shift、原生节点、prompt KV、16 槽 memory 初始化及逐轮重置沿用 checkpoint 和原有推理入口。

UND 始终走全部 28 层。选择前缀 L 时，前 L 层读取 P+GEN，后续 UND 层读取 P；最后一遍 GEN 完成全部 28 层并消费最后写入的 memory。L=28 对应 C2 全层来源与计算。每遍 GEN 从相同入口 hidden 开始，memory 每次从固定中性种子开始；窗口外使用原生 GEN，CFG 的无条件分支保持原样。

原生 `sampler_steps=50` 表示 50 个时间节点、49 次 Euler 更新，`full` 覆盖全部 49 次；没有新增采样更新。

卡空闲后在 `modelarts-job` 执行：

```bash
cd /root/bagel-LatentCoT-gen_c2_2000_20261016
bash scripts/shallow_memory_npu.sh /root/outputs/c2_2000_generation \
  --early 10 20 30 full \
  --loop-layers 8 16 28 \
  --passes 2 3 4 \
  --count 16 --seed 0 --selection-seed 0
```

该命令按默认 16 卡分配 prompt，每个 prompt 共 37 张（Base+36 对照），合计 592 张。一次只运行一个设置时分别传单个参数，例如 `--early 20 --loop-layers 28 --passes 2`，每 prompt 输出 Base 与该设置两张图。

输出包括 PNG、每张图的逐节点注入记录、循环层/遍数、checkpoint 步数与来源配置，以及 `index.html` 对照页。`NPROC_PER_NODE` 控制推理进程数，`ASCEND_RT_VISIBLE_DEVICES` 指定物理卡。

必要检查：8 项小型原生 BAGEL smoke 通过，覆盖浅层 block 调用次数、GEN 入口重置、prompt cache 与 CFG 隔离、Base、时间窗口、结果页，以及全层 R3 与训练循环出口逐元素一致。当前 16 张 NPU 被另一采样任务占用，完整权重 NPU 出图未执行。


远端已通过 CPU checkpoint 核验：format 4、C2-feedback、step 2000、28 层、16 槽、252 个原生 UND 参数张量。`--prepare-only` 已生成一条 prompt 的 37 个对照计划，位于 `/root/outputs/gen_c2_2000_plan_20261016/selection.json`；参数轴和 checkpoint 核验记录位于同目录 `checkpoint.json`。

## 7. 交互生成训练

2026-10-17 起，生成训练独立为 `qwen_latent_cot.interaction`。`visual_cli` 与 `visual_training_npu.sh` 负责 Observe / Feedback 文字训练。交互 SFT / RL 使用 `interaction.cli` 与 `train_interaction_npu.sh`。

### 7.1 架构与参数

| 文件 | 职责 |
|---|---|
| `interaction/adapters.py` | 两组 rank16 / alpha16 LoRA；调用范围内开关；固定 SFT 参考 |
| `interaction/student.py` | 冻结 format-4 parent；第一遍 GEN → UND writer → 第二遍条件 GEN |
| `interaction/data.py` | caption-image JSONL；确定性采样；候选导出与图像筛选 |
| `interaction/training.py` | 配对 flow SFT、验证、日志、优化与保存 |
| `interaction/checkpoint.py` | `interaction_lora_v1`；parent、两组 LoRA、optimizer、sampler、各 rank RNG、固定参考 |
| `interaction/sde.py` | FP32 SDE 采样 / replay、按实际转移方差计算 KL、分组优势 |
| `interaction/rl.py` | 49 步 rollout、early20 交互、单随机转移、一次 replay 更新 |
| `interaction/rewards.py` | GenEval2 逐题概率、训练奖励缓存、两卡常驻 judge |
| `diagnostics/inference.py` | 原生 BAGEL、parent 和训练后交互的统一推理 |

SFT 只使用 `mean((v_loop-(epsilon-x0))²)`。writer LoRA 仅在 UND 写入和 bank 导出时启用，consumer LoRA 仅在第二遍条件 GEN 中启用。作用范围进入激活重计算闭包。原始参数保留 parent dtype 并冻结；memory 保留生成损失到 writer 的梯度。

配置：`configs/interaction_sft.json` 默认 16 卡、global batch16、500 步、20 步 warmup，writer LR=2e-6、consumer LR=1e-5，在 100/250/500 步保存。训练 t 均匀采样原生 50 节点、shift3 的前 20 个节点，包含 t=1。训练使用无 CFG 条件速度，验证生成保持原生 CFG4。

RL 冻结 SFT writer，只更新 consumer。每组 8 个候选，默认每批 7 组、100 批；14 张 policy 卡与 2 张 Qwen3-VL judge 卡。每条轨迹随机选择 i=1…19 的一个 SDE 转移，其余采用 Euler ODE。rollout 与 replay 调用同一转移函数；KL 使用包含时间差的实际方差。日志中的 logprob 为单图 latent 维平均代理分数。固定参考为 SFT 出口，当前批 rollout 的分数单独保存。

### 7.2 数据与命令

训练和验证分别提供 JSONL，每行只需完整目标 caption 与绝对图像路径：

```json
{"caption":"A red pickup truck beside a tree.","image":"/absolute/path/to/target.png"}
```

候选导出保留来源、原始 family 与 split 到 `.audit.jsonl`。导出支持远端在用的 v1 索引和本地 v2 索引，仅取 generation role=2。图像筛选检查解码、短边至少 512、长宽比至多 1.2、内容哈希重复及提供的 benchmark 排除哈希。语义审核、人工抽查和近重复审核单独记录；筛选报告统计完成数量。

在 `modelarts-job:/root/bagel-LatentCoT-interaction_20261017` 执行。先设置经过审核的配对文件和明确的 Observe parent：

```bash
export TRAIN_PAIRS=/absolute/path/to/reviewed/train.jsonl
export VAL_PAIRS=/absolute/path/to/reviewed/val.jsonl
export OBSERVE_PARENT=/root/outputs/observe_nativeund_attn_20261015/step_020000.pt

WANDB_MODE=online bash scripts/train_interaction_npu.sh sft /root/outputs/interaction_sft_500 \
  --parent "$OBSERVE_PARENT" --train "$TRAIN_PAIRS" --val "$VAL_PAIRS"
```

该 Observe 路径用于明确示例初始化。真实训练噪声范围从 parent 配置写入 `run.json`；不能据目录名认定它接受过多噪声训练。`--parent` 可传实验目录，启动时固定到最高编号、具有完整 ZIP 目录的 checkpoint，跳过空文件与截断文件。

`--resume /absolute/path/to/step_000250.pt` 替代 `--parent`，恢复相同优化设置、相同数据和卡数的训练；总步数、保存节点与验证图数可调整。`--steps`、`--global-batch`、`--val-images` 用于短跑。每张卡始终逐样本计算并累积全局梯度。

RL 奖励文件由 train caption 派生并审核，独立保存：

```json
{"caption":"A red pickup truck beside a tree.","questions":[["Is there a red pickup truck?","Yes"],["Is a tree beside the truck?","Yes"]],"teacher":"actual_model_and_version","approved":true}
```

启动前将 `REWARD_QUESTIONS` 指向实际缓存。原子概率的算术平均是训练奖励。正式 benchmark 的题目继续用于评测。

```bash
export REWARD_QUESTIONS=/absolute/path/to/reviewed/train_questions.jsonl
bash scripts/train_interaction_npu.sh rl /root/outputs/interaction_rl_smoke32 \
  --checkpoint /root/outputs/interaction_sft_500/step_000500.pt \
  --train "$TRAIN_PAIRS" --val "$VAL_PAIRS" \
  --reward-questions "$REWARD_QUESTIONS" --steps 1 --prompts-per-batch 4 --val-images 0

bash scripts/train_interaction_npu.sh rl /root/outputs/interaction_rl_100 \
  --checkpoint /root/outputs/interaction_sft_500/step_000500.pt \
  --train "$TRAIN_PAIRS" --val "$VAL_PAIRS" --reward-questions "$REWARD_QUESTIONS"
```

脚本使用 policy 卡 0–13、judge 卡 14–15。judge 环境为 `/root/venvs/qwen3vl-npu`，生成环境为 `/root/venvs/bagel-NPU`。每次启动建立独立评分队列，退出时结束本次 judge 进程。训练生成全程保留 GEN→UND→GEN 内部交互。

### 7.3 推理与 benchmark

训练后 checkpoint 可通过同一引擎选择 `bagel`、`parent_no_memory`、`parent_memory`、`interaction_memory`、`interaction_memory_masked`、`interaction_memory_swap`。屏蔽与交换保持已训练 consumer；parent 对照关闭两组 LoRA。交换来源使用相同 prompt / t、不同初始噪声的独立轨迹。

```bash
python -m qwen_latent_cot.interaction.cli sample \
  --model /data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT \
  --checkpoint /root/outputs/interaction_sft_500/step_000500.pt \
  --caption 'A red pickup truck beside a tree.' --output /root/outputs/interaction_sample.png

python scripts/benchmark_memory_controls.py prepare --run /root/outputs/interaction_bench \
  --checkpoint /root/outputs/interaction_sft_500/step_000500.pt \
  --arms bagel parent_no_memory parent_memory interaction_memory interaction_memory_masked interaction_memory_swap
```

`train.jsonl` 记录 flow / 旧预测 flow、速度差、两组梯度、吞吐与显存。阶段验证直接使用 val caption-image 对，输出 parent / correct / masked / swap 终图。RL 另存每条选定转移、当前批旧分数、条件身份、种子、分组奖励与零方差组比例。

### 7.4 已执行：重构验证

2026-10-17，远端代码目录 `/root/bagel-LatentCoT-interaction_20261017`。

| 检查 | 已执行结果 |
|---|---|
| 小型原生 BAGEL + CPU 分布式 | 25 项通过；作用范围、两组梯度、重计算、checkpoint / RNG / optimizer 恢复、R2 CFG、SDE replay 与固定参考 |
| 真实 BAGEL 单卡 SFT | 2 步通过，两组梯度有限且非零，保存 `step_000002.pt` |
| 真实 BAGEL 16 卡 SFT | 20 步通过；第 2–20 步平均 1.569 秒/步；峰值 38.406 GiB |
| 完整 Euler 验证 | val 中一条真实 caption；parent / correct / masked / swap 四张 512px 终图已生成 |
| 14 policy + 2 judge | 32 张 rollout 完成；真实 Qwen3-VL 概率评分后，完成一次 consumer 更新并保存 |

SFT 输出为 `/root/outputs/interaction_sft_single_20261017` 与 `/root/outputs/interaction_sft_16npu_20261017`。两组 LoRA 各 10,092,544 个 FP32 参数。smoke 使用 Observe `step_020000.pt`，其实际 Observe 噪声课程上限为 0.25；新 SFT 从原生前 20 个节点采样。

验证数据从现有索引 bounded export 后筛选，包含 train15 / val8 / test9 对，仅用于执行与资源检查。完整 8k–16k 质量池的语义审核、人工抽查与 benchmark / 近重复排除继续独立完成。正式 500 步 SFT 尚未启动。

RL 成功输出为 `/root/outputs/interaction_rl_smoke32_v2_20261017`。4 个实际 train captions × 8 candidates；QA 为本次 caption-only smoke 标注，缓存内记录 `scope=runtime_smoke`，未使用超大 teacher。该批耗时 66.383 秒，不含模型加载；replay ratio=1.0，consumer 梯度范数=6.2033e-5，零方差组比例=0，框架峰值显存=37.914 GiB。平均训练奖励 0.730856 只描述这批图和问题。

首次 RL smoke 在概率检查中断：官方 softmax 别名概率累加得到 `1.000000037`。恢复原评分器的 1e-6 容差后重跑成功。训练评分按标注 Yes / No 选取答案 token，并合并重复 token ID；benchmark 保留官方答案处理。

每批只有一次 optimizer 更新。更新前 ratio 用于核对 rollout / replay；更新后另算 ratio、clip fraction 和固定参考 KL，供实际步长校准。SFT 和 RL 的这些 smoke 验证执行正确性；终图质量增益由正式训练后的 val 与 benchmark 判断。

RL 保存后已核对：writer 未变，原始 UND parent 未变，consumer 的 224 个 LoRA 张量发生更新，固定参考与 SFT 出口逐元素相同。统一 InferenceEngine 从该文件重新加载后完成 49 次 Euler 更新，其中前 20 次使用 memory，输出 512×512 终图。对一条保存转移的更新后探针，ratio=0.999999225、KL=6.8157e-7。核验位于该 RL 目录的 `verification.json`，重载终图为 `reloaded_euler.png`。
