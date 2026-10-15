# Experiment

更新：2026-10-15。实验按已执行结果、当前实现和待执行工作组织。

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
| 全量语料构建 | 待执行 | 来源实测规模、资产、family、任务视图 |
| 文本教师校准与终图评分 | 待执行 | 正确修正、保持与原生 Text 对照 |
| 新版 Observe / Feedback / C3 | 待执行 | NPU 运行、来源依赖与终图纠错 |
| 原生 GEN 适配与终图奖励 | 待执行 | memory 条件下的生成消费训练 |

历史运行主要位于 `modelarts-job`，远程仓库为 `/root/bagel-LatentCoT-loop-layer-NPU`。本地重构验证位于工作区 `artifacts/training_refactor_20261014/verification.json`。

### 轮次与输出术语

R1/R2/R3/R4 表示同一采样状态内 GEN 共运行一至四遍，分别对应零至三次反馈。训练入口的 `feedback_updates` 表示反馈次数。旧视觉训练记录中的 R1/R2 出口分别对应一次、两次反馈后的速度；下文保留该历史日志名称。

局部预测图使用当前速度形成的 x0 预测；单节点终图在一个节点注入反馈后完成原生后缀；窗口终图在指定外层时间窗口持续使用反馈。

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
| 数据基座 | `corpus.py`、`prepare_training_corpus.py` | 来源注册、归一化、资产、family、分片及索引 |
| 任务与时间 | `curriculum.py` | Observe/Endpoint/Generation/Trajectory/Conditional 配额、噪声课程、覆盖游标 |
| 原生状态 | `runtime.py`、`flow_training.py` | 文本与 VAE 条件、实际噪声、合法 flow 目标、可微 GEN |
| 表示与来源 | `visual.py`、`memory.py`、`feedback_source.py` | 现有 writer/query 残差、独立文字教师、P+GEN 来源 |
| 反馈与损失 | `loop_training.py`、`visual_objectives.py` | 固定状态循环、逐轮新来源、语义/教师/flow |
| 训练与恢复 | `visual_training.py`、`visual_record.py` | 单一优化器、全局批次、梯度同步、格式 3 checkpoint |
| 验证 | `task_evaluation.py`、`visual_evaluation.py` | 有限 family 对照、CFG 后速度、完整后缀和窗口生成 |
| 入口 | `visual_cli.py`、`visual_training_npu.sh` | Observe、Feedback、C3 共用训练入口 |

主损失为 `sem / teacher / flow`。C1 权重作为可选初始化或独立文字教师；原文字回放与 attention response 保留在历史诊断入口。

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

## 5. 待执行：下一轮实验

| 顺序 | 实验 | 起始设置 | 交付 |
|---|---|---|---|
| 1 | 全量数据整合 | 全部新旧来源 | 来源、资产、family 与任务视图报告 |
| 2 | 新版 NPU E0 | 原生条件、Observe、Feedback 各一条完整反向 | 时间、峰值内存、冻结与梯度记录 |
| 3 | 文字教师校准 | 256 个同目标修正/保持 family；自动与核对反馈分别运行 | 反思、Text/Text-memory 终图与分项评价 |
| 4 | Observe | 联合更新 writer/query，从低噪声开始 | 分 t 与来源窗口的文字读出、关闭/交换来源对照 |
| 5 | 单次 Feedback | 端点、正确图 flow、真实状态与 Observe 辅助 | 修正、保持、视觉质量与来源依赖 |
| 6 | C3 | 新来源、新反馈；逐遍直接监督 | 新/旧/固定反馈对照与各遍终图 |
| 7 | 原生 GEN 适配 | 固定可靠 memory，训练原生消费参数 | 纠错成功率、保持、质量与跨 seed 稳定性 |
| 8 | 终图奖励 | 在已有效的闭环上定义轨迹与终图目标 | 奖励、采样和参数更新方案 |

### 5.1 起始训练设置

| 项目 | Observe | Feedback | C3 |
|---|---|---|---|
| 配置 | `visual_memory_pilot.json` | `visual_feedback.json` | `memory_loop_r2.json` |
| 计划更新数 | 1000 | 500 | 500 |
| 全局 batch / 学习率 | 16 / 1e-5 | 16 / 1e-5 | 16 / 1e-5 |
| 训练参数 | writer/query | writer/query | writer/query |
| `sem / teacher / flow` | 1 / 0 / 0 | 0.1 / 1 / 1 | 0.1 / 1 / 1 |
| GEN 来源层 | 0–7 | 0–7 | 0–7 |
| 反馈次数 | 描述恢复 | 1 | 2 |
| 初始端点 t 上限 | 0.25 | 0.25 | 0.25 |
| 后续文字反馈 | 文字读出 | 单次修正/保持 | 当前完整预览重新反思 |

端点 Observe/Feedback 的实际 t 从 0.05 到校准上限抽取；C3 在线预览使用该范围内的原生节点。课程扩至 0.5 时，同步更新对应任务的校准上限。

Feedback 起始任务配额为 Observe 15%、Endpoint 40%、Generation 30%、Trajectory 10%、Conditional 5%。缺失 Trajectory/Conditional 配额转给已有 Generation。来源权重取可用 family 数的平方根，A/B 两次抽样共同计入配额。

生成 flow 的原生节点配额为 0–19 占 60%、20–34 占 20%、35–48 占 20%。真实轨迹使用保存的实际节点。来源层比较计划覆盖 0–7、0–15、0–27。

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

Observe best 依据读取 NLL。Feedback/C3 保存 last 与阶段 checkpoint，终图选择使用固定事实修正、保持和质量评价。自动反思与核对反馈分别汇总，early10/20 分别汇总。

## 6. 运行入口

以下命令在远程仓库根目录执行。环境使用 `/root/venvs/bagel-NPU`，模型为 `/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT`。数据构建命令来源、资产、family 与任务视图报告。

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

单节点预览/返回实验使用 `configs/explicit_feedback_single_node.json`，模板保存 `state` 与实际 t；原生 Text、Text-memory、隐式 memory 和交换反馈从该状态继续。添加 `--visual-checkpoint` 选择学生，`swap_id` 指向同 P/t 的另一反馈样本。`preview_kind=local_cfg` 选择局部预测，`complete` 选择完整后缀。

输出包括精确状态、采样节点、checkpoint 哈希、反馈文本、memory、条件与 CFG 速度、每步注入记录、分项时间、终图、`review.html` 和 `annotations.csv`。固定重生成使用 `*_initial.pt`，单节点实验使用 `*_state.pt`。

### 6.2 Observe → Feedback → C3

```bash
export TRAINING_CORPUS=/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus
export C0_PASS=/root/outputs/c0_text16_20261012_111241/calibration.json
export WRITER_INIT=/root/outputs/c1_historical_feedback_20261014_163638/best.pt

bash scripts/visual_training_npu.sh observe /root/outputs/observe_v3

export TEXT_TEACHER=/path/to/e1_validated_text_checkpoint.pt
export VISUAL_PARENT=/root/outputs/observe_v3/best.pt
bash scripts/visual_training_npu.sh feedback /root/outputs/feedback_v3

export VISUAL_PARENT=/root/outputs/feedback_v3/last.pt
bash scripts/visual_training_npu.sh loop /root/outputs/loop_v3
```

Observe 使用可选的 C1 初始化。Feedback 默认教师速度项使用经终图校准的 `TEXT_TEACHER`；纯 flow 运行在相应配置中使用 `teacher=0`。脚本默认 16 NPU，`NPROC_PER_NODE=1` 选择单卡。

`--writer-init` 选择 C1 writer，`--checkpoint` 选择视觉初始化，两者同时给出时采用所选 C1 writer 和视觉 query。`--text-checkpoint` 选择独立冻结教师。训练数据入口为 `--corpus`，小型诊断使用 `--data`。

格式 3 checkpoint 保存优化器、随机状态、索引指纹、课程设置、来源覆盖游标与教师身份。`VISUAL_RESUME` 或 `--resume` 恢复同阶段训练，并可增加步数预算。历史 checkpoint 用于权重初始化。

可选文字教师补训使用 `scripts/c1_feedback_npu.sh`。状态与教师导出使用 `python -m qwen_latent_cot.diagnostics.feedback_dataset`。局部/窗口生成诊断使用 `visual_cli eval/sample/source-review` 与 `scripts/shallow_memory_npu.sh`；生成审查页由 `scripts/download_shallow_review.sh` 下载。

### 6.3 评分与审计

评分工具为 `scripts/score_explicit_feedback.py`；已有显式实验的评分状态为待执行。单节点速度与终图审计使用：

```bash
python -m qwen_latent_cot.diagnostics.feedback_audit \
  --run /absolute/path/to/run --output /absolute/path/to/audit
```

评分记录事实修正、正确事实保持、图像质量和逐 seed 稳定性；运行记录保存模型/数据/反馈来源与分项计算量。
