# 数据构建与验收（v2，2026-11-10）

本文档记录**当前实际运行**的数据链路：源图筛分 → 指令生产 → 教师出图 → 最终验收。
所有数字取自实测，不是计划值。代码位于
`/data/zyd_workspace/bagel-LatentCoT/code/isochronic-v2-20261110`。

> v1（`isochronic-terminal-20261108`）已归档为 `isochronic-v1-archived-20261110`。
> v1 中「教师跑到 t=0.36 后切成纯 T2I 尾段」的混合构造已废弃，原因见 §3。

---

## 0. 训练数据合同：L 与 R 的分工

这是本轮最重要的一条分工，先写在最前面。

| 数据 | 学什么 | 怎么来 |
|---|---|---|
| **L：现成编辑对** | 指令对应的编辑能力、内容保持 | 共享噪声构造线性状态对，**不需要教师重产** |
| **R：真实历史态编辑对** | 把已学会的编辑能力迁移到真实 T2I 中间态 | 本文的父轨迹 + 完整 IT2I 目标 + 终端监督 |
| **保持样本** | 无需改动时保住原轨迹结果 | 真实父轨迹 + 保持条件，0 额外生图成本 |

**R 不承担「教模型所有编辑操作」的任务。** 那会把数据规模和能力覆盖
重新绑死在教师成功率上。

基础编辑能力已有起点（`step_003607`）。因此下一轮定位是：

> **从该编辑能力起点，做真实状态适配；用 L 回放保住已有能力。**

### 关于 `R_keep`

`terminal_active.json` 目前仍分配 `R_keep=10%`。**本轮暂无对应数据生产。**
两条路，必须择一明确：

- 若要用：保持样本可直接从已筛分的 32,340 张 `body_ok` 完稿构造
  （`a` 由该图在线加噪，`b = a`），零生图成本；
- 若不用：把份额并入 `R_active`，不留无数据支撑的配比。

---

## 1. 三条职责分离的原则

数据链路被切成三段，每段只回答一个问题。把它们混在一起，是 v1 反复返工的根源。

| 阶段 | 回答什么 | 谁在做 |
|---|---|---|
| 源图筛分 | 这张图**值不值得**被编辑？ | GLM（CPU） |
| 指令生产 | 要做的编辑**是什么**？ | GLM（CPU） |
| 教师出图 | 编辑之后**应该长什么样**？ | 原生 IT2I（NPU） |
| 最终验收 | 这张目标图**能不能**拿去训练？ | GLM + 像素诊断（CPU） |

**训练不读生产阶段的任何 `accepted` 标记**，只认 `accept_final.py` 产出的 approved 清单。

---

## 2. 环节一：源图形体筛分

### 为什么先筛

主动编辑的语义是「改这一处，其余保持」。如果源图本身已经融合、缺部件、糊成一片，
这条指令就等于要求教师**同时做两件事**：完成指定修改，并悄悄修好没人要求的损伤。
这样的标签天然歧义，靠损失权重无法补救。

所以先用 `body_ok` 把源图分成可用与不可用，而不是生产完再靠验收补救。

### 判据（`scripts/score_parents.py`）

| 字段 | 含义 | 用途 |
|---|---|---|
| `BODY_OK` | 物体是否实心完整、无融合／涂抹／撕裂／重复 | **硬门**：主动编辑只取 `yes` |
| `READABLE` | 主要物体能否辨认 | 辅助门 |
| `SUBJECTS` | 可见主要物体清单 | 供指令生成参考 |

### 实测

- 父轨迹完稿总计 42,760 张
- 实际送评 39,464 张（其中有状态文件、可进入生产的子集）
- **`body_ok` 32,340 张**
  - 占已评样本：**81.9%**
  - 占全部父轨迹：**75.6%**
- GLM 吞吐约 1,300 张/分钟（128 并发，纯 CPU）

> 两个分母都要看：**75.6% 才是最终可用比例**，81.9% 只说明送评的那批质量较好。

> 修读注意：`BODY_OK` 后面模型有时会追加一句话，解析时只取首个 token，
> 否则 `yes -- ...` 这类回答会被误判为未解析。

---

## 3. 环节二：指令生产（70% 看图 + 30% 纠错）

### 两种来源

| family | 份额 | 做什么 |
|---|---:|---|
| `invent` | 70% | 看图，提一个有价值、目标可判定的局部修改 |
| `repair` | 30% | 对照原始提示词，指出图**确实没做到**的那一条要求 |

`repair` 返回 `NO_FAILURE` 表示这张图没有它找得出的问题，直接丢弃（不计入通过率分母的错误，而是本就不该生产）。

### 操作配额（4 类，已收敛）

```
remove    40%   删除明确对象
attribute 25%   改指定对象属性
add       20%   添加对象           move  15%   明确方位移动
```

**v1 的 `relation`（关系/姿态）本轮暂不做。** 理由是**优先级**，不是它没有价值：
本轮目标是先把已有操作（增删改移）的训练闭环跑通，关系编辑的目标状态较难表述，
等闭环稳定后再加回。

配额用**步进交错**分配（`glm_edit_plan.py: interleave()`），保证任意长度的前缀都贴合配比——
否则小批试点会全落在同一操作上。

### 输出字段

`OPERATION / EDIT / EDIT_BOX / TARGET / EDIT_QUESTION / PRESERVE`，
解析后校验：框非空且面积 ≥ 64 px、至少一条保留题、目标描述 ≥ 8 词。

实测：

| 口径 | 数值 |
|---|---|
| 送入 GLM 的父轨迹 | 42,760 |
| 有效指令 | **34,802** |
| **接受率（以 42,760 为分母）** | **81.4%** |

被拒的主因：`repair_no_failure`（图没有它找得出的问题）、
`instruction_length`、`operation_mismatch`。

---

## 4. 环节三：教师出图（完整原生 IT2I）

### 关键修正

v1 的教师是这样造的：

```text
噪声 ──IT2I──→ t=0.36 ──切条件──→ 纯 T2I ──→ 终图
```

这不是「原生编辑器的输出」，而是**中途丢掉图像条件的混合体**。
当时这样做，是因为旧方案要求目标必须是一个“可接续的状态” `b`。
现在监督已经搬到终图上，`b` 不再需要，这个约束就失去了理由。

v2 改为：

```text
x_src + e ──完整 IT2I 从噪声跑到 1.0──→ y
```

**图像与指令条件全程生效**，与原生 `it2i_generate()` 语义一致
（`teachers.py: teacher_full()`）。

### 学生侧保持不变

```text
真实中间态 a_t ──等时编辑 E_θ──→ b̂_t ──冻结尾段 F──→ x̂
```

**教师负责给出好答案，学生学习用自己的部署路径达成它。两者不必走同一条生成路径。**
这是本轮最重要的一条分工。

### CFG 做了实测，结论是保持 4

同一父图、同一指令、同一种子，只改教师 CFG：

| cfg | 编辑Δ | 保留区Δ | 全局Δ | hf |
|---|---|---|---|---|
| **4.0** | 0.1481 | **0.1316** | **0.1246** | 1.87× |
| 2.0 | 0.1476 | 0.1795 | 0.1635 | 1.86× |
| 1.0 | 0.1498 | 0.2317 | 0.1954 | 1.88× |

hf 几乎不动，**降低 CFG 反而让漂移单调变大**。

**这个实验只支持一件事：在当前配置下，保持 CFG=4。**
它**不能**推出「官方降低 CFG 的建议对 IT2I 不适用」，也不能推出「问题就是图像条件本身」——
那只在单张图上测过一次，是随机种子与任务依赖的。机制归因尚未证明，不作结论。

> **部署与学生冻结尾段仍固定 CFG=4**；教师通过 `teacher_cfg_scale` 可独立配置，
> 但当前实测值就是 4。不要再拿 `config.py` 的断言当作「CFG 已排除」的证据——
> 那是我们自己写的约束。

### 三时刻与种子

- 一个父轨迹的三个时刻（0.25 / 0.36 / 0.50）**全部写入**，
  每个时刻带自己的 `a_t` 与自己的预览 `P_t`（预览是个体属性，不能共用）
- `state_b = None`：不再要求教师状态
- 种子读 `record["seed"]`（v1 读错键，导致教师退回默认种子）
- 记录 `teacher_cfg_scale`、`teacher_calls`、`teacher_seed` 以便溯源

---

## 5. 环节四：最终验收（唯一出口）

### 为什么必须只有一个出口

v1 有两套判据同时掌权：生产时 8B 规则**先斩后奏**，事后的 GLM 想纠正也无力回天；
而且 `glm_verify.py` **只写 yes 不写 no**，在 187 条上给出 187 个通过。

现在的分工：

| 判据 | 性质 | 作用 |
|---|---|---|
| 像素：编辑框变化量 | **诊断** | 记录 `edit_delta`，太小则标记疑似未编辑（`< 0.005`），**不否决** |
| 像素：保留框漂移 | **诊断** | 记录 + 标记可疑重绘，**不否决** |
| GLM 四判据 | **决定** | 见下 |
| 人工图册 | **补充** | 形体畸形，模型最不擅长，标出来给人看 |

`< 0.005` **不是硬门**。大框里删一个小物体，均值变化本来就小；
拿它当否决条款，等于重犯上一轮 `0.06` 的错误。它只产生一个
`needs_eye_review` 标记，最终由语义判据裁决。

`accept_final.py` 的 GLM 四问：
`EDIT_DONE / TARGET_MET / KEEP_OK / QUALITY_OK`。

**`partial` 一律不算通过。** 指令要么发生了要么没有，半成品会教出半成品。

**判定字段只取首个枚举值。** 模型经常写
`EDIT_DONE: yes -- the object was removed`，早期解析保留整行导致它不等于 `"yes"`，
把正确回答判成不通过。现已改为：判定字段截取首个 `yes|partial|no`，
完整解释留在 `NOTE`；读不出枚举时报 `unparsed`，不与「判否」混为一谈。

这个修复让试点合格率从 **52% 升到 58.3%**（active 68%→74%，repair 14%→21%）。

### 为什么收回像素硬门

v1 的 `> 0.06 / ≤ 0.04` 未经校准，会把数据推向「大幅像素变化」而不是「正确局部编辑」：
删除框内一个小物体，变化面积很小，编辑成功也可能不过线；
大面积纹理噪声轻易过线，却没完成编辑。

> 「能够拒绝样本」不等于「拒绝标准正确」。

### 实测：v1 的 1,312 对全部重评

| 结论 | 数量 |
|---|---:|
| 输入图像对 | 1,312 |
| 唯一 uid | **779**（533 个 uid 各出现两次） |
| 按 uid 重评后：通过 | **2** |
| 保留区被破坏 | 1,253 |
| 完全没变 | 51 |
| 其余（quality / target_partial / keep_broken） | 6 |

1,253 + 51 + 2 = 1,306，与 1,312 差 **6 条**，去向就是上表的「其余」。
重评按 uid 去重，所以评估对象是 779 个唯一样本、1,312 张图。

这暴露的不是阈值问题，而是**旧数据大面积整幅重画**。

### 试点实测（48 条请求）

解析修复后重跑（同一批 48 条）：

| | 生产 | 通过 | 通过率 |
|---|---:|---:|---:|
| active_edit | 34 | 25 | **74%** |
| repair | 14 | 3 | **21%** |
| 合计 | 48 | 28 | **58.3%** |

修复前为 25 / 52%，差额即格式误拒。

合格样本 hf 比中位数 **1.05**，接近历史参照（1.05–1.12）。
**这只说明通过样本的高频统计接近历史水平，不证明编辑是否完成、形体是否正常。**
那两件事分别由 `EDIT_DONE` 与 `QUALITY_OK` 负责，hf 不能替代它们。

**repair 通过率偏低，说明的是「这批修复请求的生产效率低」**，
不足以定义教师或学生的普遍能力边界（样本仅 14 条）。
失败样例：手电筒没放进手里、船没缩小到四分之一、大象雕像没换成獾。

→ 因此本轮**降低 repair 的生产预算**，把算力让给 active。这是预算决定，不是能力判决。

---

## 5.5 训练数据合同（训推一致的三条硬条件）

### 条件一：`a_t / P_t / x_src / y` 必须绑定同一条父轨迹

**同一个提示词不够，必须是同一条实际父轨迹。** 这是数据是否成立的核心，
比「教师是否共享种子」重要得多。

```text
父轨迹（同一 uid，同一 seed）
├─ a_t           这条轨迹在 t 时刻的真实状态
├─ P_t           从该状态产生的预览（个体属性，三个时刻各一份，绝不共用）
├─ x_src         这条轨迹的原生完稿
└─ y             对这张 x_src 执行本条指令得到的完整 IT2I 结果
```

生产中由 `coupled.py: load_source()` 从同一个 progress 记录读取全部时刻，
`build_pair()` 用该记录的完稿作为教师输入——三者在结构上不可能错配。

### 条件二：尾段条件按任务类型分开，含义不能混

| 任务 | `c_tail` | 理由 |
|---|---|---|
| `active_edit` | 编辑后的场景描述 | 学生要生成的是改动后的画面 |
| `repair` | **原始生成要求**（`parent_condition`） | 自纠偏若改写生成目标，就不叫纠偏了 |

代码中由 `coupled.py: tail_for()` 强制：`repair` 一律取原始提示词，
缺失则直接拒绝该样本（`repair_without_original_prompt`）。

**教师看到完整源图，学生只看到中途预览**——这个信息差正是监督的来源。
但学生需要的**指令与尾段条件，部署时也必须能提供**，否则训推不一致。

### 条件三：按父轨迹家族划分 train / val

一条父轨迹会产出三个时刻，可能还有多条指令，**它们共享同一张目标图**。
若其中一部分落入 val，验证分数衡量的就是「对这张图的记忆」。

因此 split 按 **父轨迹 uid** 分配，不按记录：

- `build_requests.py` 用 `sha1("split:" + uid)` 做稳定分桶（默认 5% val）；
- 同时把 `source_family` 与 `prompt_family` 都设为父 uid；
- 训练侧 `data.py: _remember()` 对跨 split 的家族直接抛错，作为第二道保险。

稳定哈希的好处：重建清单不会重排 split，断点续跑也不会泄漏。

实测 500 条样本：0 个父轨迹跨 split，家族字段全部绑定到父 uid。

---

## 6. 资产现状（截至 2026-11-10 20:12）

| 项 | 数量 |
|---|---:|
| 父轨迹完稿 | 42,760 |
| 已送评 | 39,464 |
| 形体可读（`body_ok`） | 32,340（占已评 81.9%，占全体 **75.6%**） |
| **有状态文件、可生产** | 40,570 |
| 有效指令 | 34,802（接受率 81.4%） |
| **可生产请求（交集）** | **21,773**（active 17,210 / repair 4,563） |
| 试点合格 R | 28 / 48（58.3%） |

> **注意**：`requests_v2.jsonl` 当前内容是**试点用的 48 条**（被 `pilot` 模式覆盖）。
> 正式放量前必须重跑 `run_v2_build.sh requests` 生成完整 21,773 条。
> 上面那个 21,773 是完整构建时的实测输出，不是当前文件的行数。


> **22% 的指令找不到状态文件**：渲染被中断时只写了 `targets/*.png`，没写 `progress/*.json`。
> `build_requests.py --parent-cache` 会在构建请求时直接过滤，不让它到生产阶段才暴露。

---

## 7. 已修复的三个真实缺陷

1. **`load_source` 猜文件名**：实际是 `<uid>_<hash>.json`，代码找 `<uid>.json`，
   试点 13/48 静默丢失 → 改为按 uid 建索引
2. **父轨迹分散在两个缓存**：`--source-cache` 只收单值 → 改为多缓存依次查找
3. **请求指向无状态父轨迹**：→ 构建阶段过滤

---

## 8. 运行命令

```bash
# 0) 环境
export METTO_BASE_URL="https://metto-api.com"
export METTO_GLM_KEY="<key>"
CODE=/data/zyd_workspace/bagel-LatentCoT/code/isochronic-v2-20261110
ROOT=/data/zyd_workspace/isochronic_data_20261107

# 1) 源图形体筛分（CPU，约 30 分钟 / 42k）
python3 $CODE/scripts/score_parents.py \
  --parents $ROOT/active/mt $ROOT/active/mt_b \
  --out $ROOT/parent_scores.jsonl --workers 128

# 2) 合并成生产请求（会过滤无状态父轨迹）
python3 $CODE/scripts/build_requests.py \
  --scores $ROOT/parent_scores.jsonl \
  --plan $ROOT/active/plan/all42k.jsonl \
  --parent-cache $ROOT/active/mt $ROOT/active/mt_b \
  --out $ROOT/requests_v2.jsonl --limit 3000

# 3) 教师出图（NPU，单卡约 6.5 条/分钟）
cd $CODE && PYTHONPATH=$CODE ASCEND_RT_VISIBLE_DEVICES=0 \
  PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
  /data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python -m scripts.prepare_coupled \
    --model /data/zyd_workspace/models/SenseNova-U1.5-8B-MoT \
    --source /data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b \
    --config configs/u15_isochronic_state/terminal_active.json \
    --requests $ROOT/requests_v2.jsonl \
    --source-cache $ROOT/active/mt $ROOT/active/mt_b \
    --output $ROOT/v2_full

# 4) 最终验收（唯一出口）
python3 $CODE/scripts/accept_final.py --cache $ROOT/v2_full \
  --out-approved $ROOT/v2_full_approved.jsonl \
  --out-rejected $ROOT/v2_full_rejected.jsonl \
  --report $ROOT/v2_full_report.json --workers 96
```

一键脚本：`bash /tmp/run_v2_build.sh {requests|pilot|full}`

---

## 9. 时间与产能

**口径必须说清：6.5 条/分钟是单卡速率，不是 16 卡。**
试点跑在 **1 张 NPU** 上（`ASCEND_RT_VISIBLE_DEVICES=0`，48 条约 7.4 分钟）。

| 口径 | 速率 | 说明 |
|---|---|---|
| 单卡 | **6.5 条/分钟** | 试点实测 |
| 16 卡（同机，线性外推） | 约 104 条/分钟 | **未实测**，受共享存储与数据加载限制，实际会低于线性 |
| 64 卡（4 节点） | 约 416 条/分钟 | **未实测**，且需 4 个节点全部空闲 |

按单卡 6.5 条/分钟、通过率 58.3%：

| 目标合格数 | 需生产 | 单卡耗时 |
|---:|---:|---:|
| 500 | 858 | 2.2 h |
| 2,000 | 3,431 | 8.8 h |
| 全量 | 21,773 | 55.8 h → 约 12,700 |

多卡时按 shard 切分（`--shard-index / --shard-count`），每 shard 独占一张卡；
**多卡速率尚未实测，排期前应先跑一次 4 卡试水。**

---

## 9.5 闭环诊断与指令生成（另见专文）

推理期的「预览 → UND 诊断 → 编辑指令 → 中间态编辑」链路不在本文范围，
单独记录在 **[闭环诊断与指令生成](closed_loop_diagnosis.md)**，原因是它当前
**存在若干已知缺陷，需要人工审查**：

| 缺陷 | 表现 | 实测占比 |
|---|---|---:|
| 指令模板单一 | `Change the number of X to exactly N.` | **48%** |
| 复合指令无约束 | 一句里 2–3 个动作 | **68%** |
| 判定一刀切 | `pass_threshold=0.65` 不区分题型 | — |
| 材质词表封闭 | `checkered`/`sparkling` 被当成对象缺失 | — |

该文逐条列出了全部正则模板、材质词表、不可执行词表，以及修复前后的行为对比。

**重要**：当前闭环的「92% 失败率」包含了上述缺陷造成的**过判**，
不能作为模型能力结论。定性需要接入 Geneval2 的 `vqa_list` 独立判分。

---

## 10. 训练接线（已完成，2026-11-10）

> 本节记录**实际接线结果**。之前列为「未实现」的两项，现均已落地并验证。

### 10.1 唯一验收出口已接入训练

做法不是改训练器六处读取逻辑，而是让验收**把裁决写回记录**：

```
accept_final.py          → approved/rejected jsonl
apply_verdicts.py        → 改写 progress/*.json 与 manifest*.json 的 states[t].accepted
trainer (data.py)        → 照原样读 accepted，无需改动
```

**关键的坑**：训练器读的是 **64 个分片 manifest**，不是 `progress/`。
manifest 在生产结束时写出，那时还没有任何裁决，所以全部是 `accepted=None`，
训练器直接报 `The state-pair cache has no accepted training times`。

**只改 progress 不改 manifest，等于没改。** `apply_verdicts.py` 现在两者都写。

未评判的记录写入 `accepted=False, reject_reason="not_judged"`，不偏向任何一边。

### 10.2 反演依赖已取消

| 位置 | 原状 | 现状 |
|---|---|---|
| `launch_train64.py` | `Stage 2 requires --inverted` | `--native` 或 `--inverted` 二选一 |
| `config.py` STAGE_WEIGHTS | Stage 2 必需 `inverted_pair` | 已移除 |

### 10.3 `R_keep` 已从本轮配比移除

`terminal_active.json` 原分配 `R_keep=10%`，但本轮没有生产保持样本，
训练器会拒绝启动（`Role weights ask for records this run does not have`）。

| 角色 | 旧 | 新 |
|---|---:|---:|
| R_active | 50% | **60%** |
| L_edit | 30% | 30% |
| R_repair | 10% | 10% |
| R_keep | 10% | —（移除） |

若下一轮需要保持样本，可直接从已筛分的 32,340 张 `body_ok` 完稿构造
（`a` 在线加噪，`b = a`），零生图成本。

### 10.4 首个 64 卡训练运行（进行中）

```bash
python scripts/launch_train64.py --stage 2 \
  --config configs/u15_isochronic_state/terminal_active.json \
  --linear <L_30k.json> --native <v2_full> \
  --init-from <step_003607> --output <train_v2> --steps 5000
```

实测：**35.6 秒/步**（64 卡，全局 batch 64，峰值显存 41.3 GiB / 64 GiB）。
5,000 步约 **49 小时**。

### 10.5 数据闭环验证（已核）

| 检查 | 结果 |
|---|---|
| 验收合格 | 11,065 |
| 训练器加载 | **11,065**（完全一致） |
| approved 未加载 / 误纳 | 0 / 0 |
| R 按 split | train 10,855 / val 210 |
| L train split | 28,849 |
| 运行中 uid 越界 | **0** |

这条验证比「进程还活着」有意义得多：它证明训练消费的正是验收通过的那批样本。
