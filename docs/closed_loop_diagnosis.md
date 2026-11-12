# 闭环诊断与指令生成：实现现状与已知缺陷

更新：2026-11-11 14:15。记录**当前实际运行**的诊断链路，供人工审查。
代码基线：`/data/zyd_workspace/bagel-LatentCoT/code/isochronic-v2-20261110`。

---

## 1. 链路概览

```
硬提示词 P
   ↓ 原生 T2I 前向到 t=0.36
预览 P_t
   ↓ decompose(P) → 原子问题集 Q
Q 的逐条判定（UND）
   ↓ 与 pass_threshold 比较
失败集 F
   ↓ feedback_from_questions()
编辑指令 e + 保留项
   ↓ 等时编辑器 8 步
b̂ → 冻结尾段 → 终图
```

**这条链路的每一步都在用启发式规则做判断**，而不是学习出来的模块。下面是全部规则。

---

## 2. 判定环节：问题从哪来

`decompose(rt, raw, cfg)` 把提示词拆成 yes/no 问题。**这不是自由生成的**，而是由提示词里的结构短语触发的模板：

| 触发 | 生成的问题形态 |
|---|---|
| 计数 `seven croissants` | `Are there exactly seven croissants visible in the image?` |
| 颜色 `green backpack` | `Is the backpack green in color?` |
| 空间 `X behind Y` | `Is the X behind the Y?` |
| 动作 `dog chasing cat` | `Is the dog chasing the cat?` |
| 存在 `a pig` | `Is there a pig?` |

**已知敏感点**：问题措辞直接决定后面的判定和指令生成。
`exactly ... visible` 这个措辞来自下面的计数正则，是后面「判定过严」的源头之一。

---

## 3. 判定环节：阈值

| 参数 | 值 | 位置 |
|---|---:|---|
| `pass_threshold` | **0.65** | `core.json` |

判定规则（`review_preview`）：

```python
failed = [q for q in questions if p_yes(q) < 0.65]
```

**这是一个全局固定阈值，对所有问题类型一视同仁。**

### 已观察到的缺陷

在 `execution_test.html` 的 7 例中：

- `two metal toys`：预览里**已经有两个金属小人**，却仍被判为失败，指令是「改成正好两个」
- `two stone raccoons`：同样，预览已满足，仍被判失败

**结论：计数类问题的判定过严。** 可能原因有二，尚未区分：

1. 生成式打分对「exactly N visible」这种措辞给分偏低；
2. `entailment_cap` 会用最弱的存在题去压低关系题，可能连带影响计数题。

**在区分这两者之前，不应再基于 56%/44% 这类比例做结论。**

---

## 4. 指令生成：正则模板（核心问题）

`feedback_from_questions()` 把失败的问句转成祈使句，**完全依赖正则模板**。
下面是全部模板，逐条列出以便审查。

### 4.1 主转换 `_imperative()`

| 问句形态 | 正则 | 产出模板 |
|---|---|---|
| 计数 | `exactly\s+(\w+)\s+(.+?)\s+visible` | **`Change the number of {noun} to exactly {n}.`** |
| 空间 | `(?:Is\|Are) the (.+?) (behind\|in front of\|to the left of\|...) the (.+)` | `Put the {subject} {rel} the {obj}.` |
| 动作 | `(?:Is\|Are) the (.+?) (playing with\|chasing\|jumping over\|...) the (.+)` | `Make the {s} {verb} the {o}.` |
| 存在 | `(?:Is\|Are) there (a\|an\|any\|\d+\|two..ten) (.+?)` | `Add {article} {noun}.` |
| 材质存在 | 存在 + `_MATERIAL_IN` 命中 | `Make the {subject} {material}.` |
| 融合 | `\b(distinct\|fused)\b` | `Separate the subjects so they are not fused.` |
| 颜色声明 | `_color_claim()` | `Make the {subject} {color}.` |

### 4.2 材质词表 `_MATERIAL_IN`

```
glass|wooden|wood|metal|metallic|stone|ceramic|porcelain|plastic|rubber|
fabric|leather|paper|marble|bronze|brass|copper|iron|steel|gold|silver|
crystal|ice|clay|concrete|velvet|silk|wool
```

**这是一个封闭词表。** 词表外的材质（如 `sparkling`、`checkered`、`striped`）不会被识别为材质，
会落到 `Add` 分支。

### 4.3 不可执行判定 `_UNACTIONABLE`

```
\b(other|besides|only subjects?|without any occlusion|partially obscured)\b
```

命中即丢弃该问题（记入 `rejected_questions`），不生成指令。
**这份表只有 5 个词**，覆盖面很窄。

### 4.4 空间与动作词表 `_SPATIAL_Q` / `_ACTION` / `_SETTING_Q`

仅用于**分类**（决定 `classify_question` 的返回），不直接生成文本。

---

## 5. 已知缺陷清单（按严重度）

### 缺陷 A：指令措辞冗长且不符合自然语言

```
问句  Are there exactly seven green croissants visible in the image?
指令  Change the number of green croissants to exactly seven.
```

**问题**：`to exactly seven` 是模板硬拼出来的，不是自然表达；而且它是**固定句式**，
下游模型（编辑器）只见过这一种写法，泛化风险高。

### 实测：模板占比（96 例闭环，65 条非空指令）

| 模板 | 条数 | 占比 |
|---|---:|---:|
| **计数 `Change the number of X to exactly N.`** | **31** | **48%** |
| 添加 `Add ...` | 19 | 29% |
| 属性 `Make the X Y.` | 9 | 14% |
| 空间 `Put the X ...` | 5 | 8% |
| 其他 | 1 | 2% |

**近一半的指令是同一种句式。** 这不只是措辞问题：编辑器在训练和推理中反复见到
`Change the number of ... to exactly N.`，它对这一句式的响应可能远好于其他表达。
一旦换成自然语言表述，行为是否保持是未知的。

用户已指出这一点。**建议改为更自然的表述**，例如：

```
Make there be exactly seven green croissants.
```
或直接给目标状态描述：
```
Seven green croissants.
```

### 缺陷 B：判定阈值一刀切

`pass_threshold = 0.65` 对计数、属性、空间一视同仁。
计数题天然更容易被判失败（模型对数量不敏感），应当单独标定。

### 缺陷 C：材质词表封闭

`sparkling`、`checkered`、`striped` 等常见属性词不在表内，会被误判为「对象缺失」，
从而生成 `Add ...` 而不是 `Make ... {attribute}`。

### 缺陷 D：组合指令无约束

一条指令可能包含 2–3 个子句（`Change ... . Change ... . Put ... .`）。
闭环没有限制子句数量，但执行测试表明**单句指令的成功率明显更高**。

**实测：96 例中 65 条非空指令，其中 44 条（68%）含多个子句。**

| 例 | 实际产出 |
|---|---|
| `a white cookie to the left of three bears, and five cars` | `Change the number of cars to exactly six. Add a kangaroo.` |
| `seven checkered bicycles behind five striped chairs ...` | `Add six checkered cats. Make the guitars metal. Put the six checkered cats and seven checkered ...` |

建议：**首轮只生成单一动作的指令**，复合任务留到后续。
这也是 §7 里「执行测试」只挑单句案例的原因——复合指令无法区分「整体没做」和「只做对一半」。

### 缺陷 E：转换失败曾经静默放行

**已修复**（2026-11-11）：新增 `diagnosis_outcome` 字段，
区分 `edit` / `failed_unconverted` / `no_failure`。
修复前 27 例直通中 6 例属于转换失败。

---

## 6. 已完成的修复（2026-11-11）

| # | 修复 | 验证 |
|---|---|---|
| 1 | `review_preview` 传入 `cfg`，走生成式打分而非 prefill margin | 48 例重跑，0 崩溃 |
| 2 | 存在题区分「对象缺失」与「属性错误」 | 6 条单测全过 |
| 3 | `diagnosis_outcome` 显式记录转换失败 | 4 例被正确标记 |
| 4 | `phase_grid` 与 `state_times` 解耦 | 训练/推理均 52 节点 43 区间 |
| 5 | `terminal_loss` 可用 `edit_box` 锚定结构区 | `box_mask` 覆盖 13.2%，单测通过 |

**注意**：修复 4、5 需要**重新训练**才生效；修复 1–3 在推理期即时生效。

---

## 7. 修复前后的行为对比

同一批 48 条硬提示词、同一检查点 `step_001200`：

| | 修复前 | 修复后 |
|---|---:|---:|
| 触发编辑 | 21（44%） | **44（92%）** |
| 原生直通 | 27（56%） | **4（8%）** |
| 失败题占比 | 48/127 = 38% | **117/127 = 92%** |

**但请勿直接解读为"修复后更好"。** 92% 的失败率里包含了缺陷 A/B 造成的过判。
需要用 Geneval2 的 `vqa_list` 做独立判分才能定性。

---

## 8. 未接入的判分

Geneval2 提示词自带 `vqa_list`（每个原子要求一条标准答案），
**目前完全没有用于判分**。这是判断「预览是否真的不满足要求」最客观的途径。

**建议下一步**：接入 `vqa_list`，对预览和终图各判一次，
用差值衡量闭环净收益，替换当前依赖 UND 自评的做法。

---

## 9. 审查建议顺序

1. **先看 §4 的正则模板**：这是指令质量的根源，决定了下游能学到什么。
2. **再看 §3 的阈值**：0.65 是否合理，是否需要按问题类型分别标定。
3. **最后看 §5 的缺陷清单**：决定先修哪个。
4. 修复后用 `vqa_list` 独立验证，而不是看闭环自身的诊断比例。
