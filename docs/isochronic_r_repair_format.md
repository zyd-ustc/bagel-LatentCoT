# R-repair 数据格式与训练用法

日期：2026-11-09。数据构建背景见 [数据底座构建方案](isochronic_data_build_plan.md) §8，
训练配方见 [纠错训练方案](isochronic_repair_training_plan.md)。样例见
[isochronic_r_repair_samples.jsonl](isochronic_r_repair_samples.jsonl)（16 条）。

## 1. 一条 R-repair 记录是什么

它绑定的是一次**真实的生成失败**，不是一次自由选择的编辑：

| 字段 | 含义 | 训练时用在哪 |
|---|---|---|
| `uid` | 父轨迹 id | 检索 |
| `source_image` | 父图**原生完稿** `x_src` | 保持损失、画质参照 |
| `parent_condition` | 原提示词 `c_raw` | 尾段条件、mask 参照 |
| `failed` | UND 实际看到的错误（一条） | 不参与损失，给人看 |
| `edit_instruction` | 修复指令 `e` | **学生输入** |
| `edit_questions` | 恰好一条"修好了吗" | 验收 |
| `preserve_questions` | 1–3 条"不许碰的要求" | 验收 |
| `target_image` | `x_eff = Φ^{c_tail}(b)`，教师产出的目标图 | **结构损失** |
| `tail_condition` | 自纠错时等于 `c_raw` | 尾段条件 |

关键一条：**`x_eff` 不是回给学生的答案，而是训练时定义"改对"的参照。**
学生看不见它，也看不见 `x_src`；它只看得到飞行中的状态和预览。

## 2. 样例（节选，完整 16 条见 jsonl）

```json
{"uid": "tgt_under_02585", "family": "under", "target_count": 3,
 "failed": "The image contains 4 watermelons instead of exactly 3.",
 "edit": "Remove one watermelon to leave exactly three green watermelons.",
 "edit_q": ["Are there exactly three green watermelons?"],
 "preserve": ["Are the watermelons still green and uncut?"],
 "action": "remove", "delta": -1}
```

```json
{"uid": "tgt_space_00042", "family": "space",
 "failed": "The bicycle is on the right, the motorcycle on the left.",
 "edit": "Swap the bicycle and the motorcycle so the bicycle is on the left.",
 "edit_q": ["Is the red bicycle on the left and the motorcycle on the right?"],
 "preserve": ["Are both vehicles still parked on the same street?"],
 "action": "swap", "delta": 0}
```

同一句原提示词，四种画面给出四种动作——这正是闭环要的行为：

| 画面 | `failed` | `edit` |
|---|---|---|
| 只有 3 个粉色芍药 | "contains 3 pink peonies, not 5" | 补 2 个 |
| 有 4 个西瓜 | "contains 4, instead of exactly 3" | 删 1 个 |
| 手在下、脸在上 | "the relation is inverted" | 交换两者 |
| 两个主体都戴蓝围巾 | "both wear blue" | 兔子改红 |

## 3. 训练时怎么用

### 3.1 一个样本的完整路径

```python
# 1) 取真实历史态与预览（父轨迹在 t=0.36 的缓存，不重跑前 31 步）
a       = cache["state_a"]        # 带噪真实状态
P       = cache["preview"]        # 3 次求值的前瞻预览
e       = row["edit_instruction"] # "Remove one watermelon ..."
c_tail  = row["tail_condition"]   # 自纠错 = c_raw

# 2) 两路 UND 预填（各一次，8 步内复用）
image_prefix = UND("<image>" + P)
joint_prefix = UND("<image> + Edit request / Generation target / Preserve" + P)

# 3) 冻结锚点，不反传
b0 = H0(a, t=0.36; image_prefix)

# 4) 8 步等时编辑，t 冻结，沿 s 走
q = a
for m in range(8):
    u = t * (Hθ(q, t, m/8; joint_prefix) - b0)
    q = q + u/8
b_hat = q

# 5) 冻结原生尾段续生到 t=1（CFG=4，与部署完全一致）
x_hat = euler(b_hat, t: 0.36→1.0, NativeT2I(c_tail))

# 6) 终端损失
L = terminal_loss(x_hat, teacher=row["target_image"],
                  source=row["source_image"], valid)
```

梯度路径：`x_hat → 尾段 19 步 × 2 CFG 分支 → b_hat → 8 步编辑 → LoRA`。
底座参数全程冻结。

### 3.2 损失的三项与掩码

```
M  = 软掩码(|x_teacher − x_src| 的 q60–q95，3×3 模糊)
L  = mean_M |x̂ − x_teacher|                       # 结构：改对了
   + 1.0 · mean_{1−M} |x̂ − x_src|                 # 保持：没碰别的
   + 0.5 · Huber(hf(x̂)/hf(x_src) ∉ [0.75, 1.15])  # 纹理预算
```

三项的分工来自实测：教师图本身是 1.89 倍粗糙的，如果只贴结构项，学生学到的就是"变糙"；
保持项把无关区域锚在干净的原生完稿上，高频带阻止"用纹理买位移"。

**掩码的局限（必须记住）**：`M` 由像素差分生成，会同时圈进真正的编辑、教师的构图漂移和
光照变化。所以结构项里仍有一部分是在模仿教师纹理——**不能由 L 下降推断"语义已经改对"**，
语义要看 `edit_questions` / `preserve_questions` 在 `x̂` 上的判定。

### 3.3 角色与占比

`R_repair` 走终端监督，和 `R_edit` / `R_keep` 同组；线性对不走这条路：

| role | 损失 | 权重 |
|---|---|---:|
| R_repair | 终端（走尾段） | 0.50 |
| L_edit | 状态流匹配 | 0.20 |
| L_reverse | 状态流匹配 | 0.15 |
| R_keep | 终端 | 0.10 |
| R_edit（v6） | 终端 | 0.05 |

配置：`configs/u15_isochronic_state/terminal_repair.json`。

### 3.4 验收（不是损失，是判据）

每一步出图后检查：

| 判据 | 阈值 |
|---|---|
| `edit_questions` 在 `x_hat` 上全过 | **必须** |
| `preserve_questions` 在 `x_hat` 上不破 | **必须** |
| `hf(x_hat)/hf(x_src)` 中位 | ≤1.30，硬失败 >1.5 |
| 原本正确的要求被改坏的比例 | ≤5% |

前两条是语义，后两条是画质。**两个必须分开判**——这是这一轮反复吃亏的地方。

## 4. 当前数据状态（18:40）

| 项 | 数量 |
|---|---:|
| 父轨迹 | 2,968 |
| 修复请求（第二轮，含 backfill 标记） | 574 / 733（跑到 1,220 条处） |
| 其中需要离线补保留题的 | 约 23% |

以 `action` 分类（第一轮 2,207 条）：remove 28% · swap 16% · add 49% · move/change 6%。
对照旧池 `coupled_v6` 的 **98.3% add、0 remove、0 swap**。

`needs_preserve_backfill=true` 的记录，其保留题由 `decompose(parent_condition)` 离线补，
标记 `preserve_verified=false`。**这批在开训前要抽查 30 条**，确认补出的保留题没有被修复
动作本身破坏；若破坏率高，这批只用于预训练，不进入门禁判定。

## 5. 离线用完稿、部署用预览：两种信息必须分开

这一节记录 19:00 的一次修正，它决定了本批数据能回答什么问题。

### 5.1 发生了什么

第一版请求是让 UND 看**预览 `P`（t=0.36 的半成品）**写的，再用 `x_src`（t=1 完稿）早筛。
结果 2,967 条里几乎全部被 `edit_not_needed` 挡掉，抽查两条：

| uid | UND 从 `P` 写的问题 | 在 `x_src` 上的概率 |
|---|---|---:|
| `tgt_attr_05482` | "Is the green scarf on the short mug?" | **0.98** |
| `tgt_over_00496` | "Are there exactly 6 brown ceramic cups visible?" | **0.85** |

**完稿里本来就是对的。** 预览只有 3 次求值的前瞻，本身模糊；UND 对着它数杯子、判属性错位会误判，
而这类缺陷在完整生成中并不延续。早筛没有错，错的是请求的来源。

### 5.2 现在的规则

| 阶段 | 观察对象 | 目的 |
|---|---|---|
| **离线造数据** | **`x_src`（原生完稿）** | 判定"不改的话，最终会错在哪里"，据此写修复指令 |
| **训练编辑器** | 只给 `a`、`P`、`e` | 学生看不到 `x_src`，也不需要先画完再改 |
| **上线自动诊断** | 只能看 `P` | 还没解决：部署时拿不到完稿 |

请求写出后仍要**在完稿上复核**（`edit_true_on_completion` / `protected_on_completion`）：
看完稿写题，不等于写出的题必然为假。复核不通过的进
`*.review.jsonl`，作为**诊断器的误触发负例**——正确动作是直通，而不是把错误指令送进编辑器
再要求它忽略。

### 5.3 这批成果能回答和不能回答什么

**能回答**：在反馈正确的前提下，模型能否在真实中间态上原地修好，且不破坏其他内容。

**不能回答**：系统能否自己判断该改什么。离线能用完稿确认错误，部署时不能。
用这批数据训练出来的是"**给对反馈，能不能修**"，不是"**系统已经能自己给对反馈**"。

两步都要做，顺序不能反：先让编辑器学会修复真错误，再攻"只看预览能否判断得同样准"。
若先做后者，预览误判会决定编辑器学什么。
