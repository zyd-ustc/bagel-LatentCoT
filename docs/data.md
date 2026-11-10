# 最新数据方案与构建逻辑 (Data Pipeline v2)

本文档系统阐述等时状态编辑（Isochronic State Flow）的最新数据体系、任务分类逻辑、生产铁律与格式规范。

---

## 1. 核心战略转变：先学会“执行编辑”，再学习“自主诊断”

在早期方案中，团队曾尝试**“在原生生成自然画错的轨迹上挖掘数据”**。但实测表明：
1. **自然失败率极低且严重偏科**：原生模型在常规提示词下的自然错误率仅 5%~10%，且其中 **98.3% 都是少画物体的“添加类”**，几乎没有删除、空间关系对调等关键错误。
2. **预览误触发率高达 70%**：实测表明，在 $t=0.36$ 的 2 步轻量预览 $P$ 上看似错误的缺陷，有 **70% 在 50 步完稿后被底座自行修好了**。对着预览造任务会导致大量“无中生有”的伪编辑。

### 最新定位（执行器优先）：
> **核心目标：训练一个接收明确指令、能在真实生成中间态上完成局部语义改变的编辑器。**
> **底座当前是否画错，不再决定样本能否进入训练！**
> 即使原生轨迹生成的画面完全正确，只要在中间态给它一个明确的修改指令（如“把左边的杯子删掉”），模型就必须学会精准原地变轨。把“编辑执行”与“缺陷诊断”彻底解耦。

---

## 2. 最新四角色数据体系（50 / 30 / 10 / 10 架构）

训练集由四个职责清晰的角色组成，兼顾**真实轨迹分布、语义广度、错误撤销与画质保持**：

| 数据角色 | 份额 | 为什么编辑？源图是否正确？ | 目标条件 $c_{\text{tail}}$ | 训练监督形式 | 现有资产与来源 |
|---|---:|---|---|---|---|
| **`R_active`**<br>(主动编辑) | **50%** | **主力攻坚**：在真实轨迹中间态主动发起一次局部改变；源图完全正确 | 编辑后的完整新场景描述 | 真实展开后的**终端图像监督** | 13,492 条真实父轨迹候选；首批开训包 1,500 条，扩展包 6,000 条 |
| **`L_edit`**<br>(现成编辑对) | **30%** | **广度底座**：学习通用增删改与丰富属性词汇，防止过拟合 | 目标图对应的描述（或拼接衍生） | 共享噪声线性状态上的**条件流匹配 (CFM)** | 现有 49 万条编辑池（`dataset40w`），精选 10 万条分层池，**纯在线加噪合成** |
| **`R_repair`**<br>(定向纠错) | **10%** | **真实失败修复**：针对计数/方位/错绑等真正画错的轨迹实施纠错 | 原始文生图请求 $c_{\text{raw}}$ | 真实展开后的**终端图像监督** | 现有 2,086 条合格纠错目标，精准覆盖复杂错误撤销 |
| **`R_keep`**<br>(原图保持) | **10%** | **防画蛇添足**：明确要求保持当前画面不变 | 父轨迹原条件 $c_{\text{raw}}$ | **终端保持监督**<br>($y = x_{\text{src}}$) | 现有 6,576 ~ 13,492 条真实父轨迹，直接复用原生完稿，**0 额外生产成本** |

---

## 3. 主动编辑（R-active）五大互斥操作配额

为了彻底解决模型“只会加东西、不会删和改”的问题，R-active 显式规定了 5 类**互斥操作**配额：

```
                      ┌── 1. 添加 (Add)            : 20% (增加新主体/物件，如补齐数量)
                      ├── 2. 删除 (Remove)         : 25% (抹除指定主体，背景自然修补)
R-active 主操作配额 ──┼── 3. 移动 (Move)           : 20% (主体位置迁移，原位显露背景)
                      ├── 4. 关系/姿态 (Relation)  : 15% (拓扑对调、形态开闭、动作改变)
                      └── 5. 属性变化 (Attribute)  : 20% (换颜色、换材质、属性归属对调)
```

- **说明**：计数（count）是增删的附加标签，空间（spatial）是移动的附加标签，不作为重叠主类。
- 每条指令只指定一个主操作，确保题意聚焦、易于归因。

---

## 4. 数据生产与质检的三大铁律（避坑实测总结）

### 铁律一：离线造数一律看完稿 $x_{\text{src}}$，绝不看粗预览 $P$！
- **实测教训**：预览图只有 2~3 步草图，VLM 在上面数物体、判属性极其容易“看走眼”（70% 误判）。如果看预览拟定“加一个西瓜”，而完稿其实本来就有 3 个西瓜，这就会把正确的完稿改坏成 4 个！
- **铁律**：**造数专家看着完稿 $x_{\text{src}}$ 拟定任务与验收题**，确认该编辑在源图上确实有必要执行；学生推理时看 $P$，教师造数看 $x_{\text{src}}$。

### 铁律二：保留题必须从原文本分解，绝不靠看图自由描述！
- **实测教训**：如果让 VLM “看着图自己写保留题”，它写出的描述常常通不过自己后续的打分检验（50% 假阴性误报）。
- **铁律**：保留题统一从原始提示词 $c_{\text{raw}}$ 中**做文本原子化分解（Decompose）**，并在源图上打分验证。只有源图上**确实成立、且本次编辑不修改**的内容，才进入保留题集。

### 铁律三：双路目标生产（原生教师主路 + 外部强编辑补充路）
- **主路（原生 IT2I 教师读取完稿）**：
  $$\text{真实状态 } a \longrightarrow \text{完稿 } x_{\text{src}} + \text{指令 } e \overset{\text{IT2I 教师}}{\longrightarrow} b_{\text{teacher}} \overset{\text{冻结尾段}}{\longrightarrow} \text{终图 } y$$
  教师读取清晰完稿，保证目标图像形体明确；中间态 $b_{\text{teacher}}$ 仅作为缓存兼容项，不强加精确状态回归。
- **补充路（外部终图直接入终端监督）**：
  已有可用外部强编辑工具时，直接生成高质量目标图 $y$ 参与终端监督，**不再强行做逆向反演求 $b$**（反演会带来 2.55 倍粗糙高频，已彻底弃用）。

---

## 5. 规模化瓶颈破解：如何撑起几十万~百万级训练

针对“生产太慢、卡在数据阶段”的困境，大规模训练采用**轻量在线 + 精锐离线**架构：

1. **L 数据纯在线化（0 离线存储开销，激活 50 万底座）**：
   - 现有的 49 万编辑对（`dataset40w`）不需要离线跑任何扩散前向；
   - 训练时 DataLoader 实时读图，按随机 $t$ 在线完成线性加噪：$a = t x_{\text{src}} + (1-t)\sigma\varepsilon$。
   - 启用 `allow_derived_tail=True`，直接利用现成图对开启数十万级回放。
2. **R-keep 镜像对冲（0 生产算力成本）**：
   - 任何一张干净的完稿图像 $x_{\text{src}}$，直接在线加噪成状态 $a$，目标设为 $b = a$，指令固定为 `"Keep the current scene unchanged."`。
   - 瞬间获得数十万条原图保持样本，不消耗一秒钟的离线生图算力。
3. **R-active / R-repair 生产大瘦身（提速 6 倍）**：
   - 在数据生产阶段**砍掉 32 步尾段生成和 VLM 打分**，仅计算至 $t=0.36$ 得到 $b$。
   - 单条生产耗时从 30 秒暴跌至 4~5 秒；64 卡集群可在半小时内产出万条精锐变轨数据。

---

## 6. 数据格式规范与实操样例

### 6.1 R-active / R-repair 统一数据格式
每条合格的主动编辑或纠错记录遵循以下 JSONL 规范：

```json
{
  "uid": "tgt_under_02585",
  "role": "R_repair",
  "task_type": "repair",
  "action": "remove",
  "source_image": "images/tgt_under_02585_source.png",
  "target_image": "continuations/tgt_under_02585_effective.png",
  "parent_condition": "A close-up photo of exactly three green watermelons on a wooden table.",
  "failed": "The image contains 4 watermelons instead of exactly 3.",
  "edit_instruction": "Remove one watermelon to leave exactly three green watermelons.",
  "tail_condition": "A close-up photo of exactly three green watermelons on a wooden table.",
  "edit_questions": [
    "Are there exactly three green watermelons on the table?"
  ],
  "preserve_questions": [
    "Are the watermelons still green and uncut?",
    "Is the surface still a wooden table?"
  ],
  "delta": -1,
  "split": "train"
}
```

### 6.2 典型五大操作样例速查

| 操作类型 | 原要求 / 画面事实 | 编辑指令 (`edit_instruction`) | 唯一编辑题 (`edit_questions`) |
|---|---|---|---|
| **删除 (Remove)** | 画面画了 4 个西瓜（要求 3 个） | `"Remove one watermelon to leave exactly three watermelons."` | `["Are there exactly three watermelons?"]` |
| **添加 (Add)** | 画面仅有 3 朵粉色芍药（要求 5 朵） | `"Add 2 more pink peonies to the vase."` | `["Are there exactly five pink peonies?"]` |
| **移动/位置 (Move)** | 自行车在右侧，摩托车在左侧 | `"Swap the bicycle and motorcycle so bicycle is on the left."` | `["Is the bicycle on the left of motorcycle?"]` |
| **属性解绑 (Attribute)** | 兔子和狐狸都戴了蓝色围巾（错绑） | `"Make the rabbit's scarf red and the fox's scarf blue."` | `["Is the rabbit's scarf red and fox's blue?"]` |
| **姿态改变 (Relation)** | 画面中的雨伞处于闭合状态 | `"Open the umbrella while keeping its color and position."` | `["Is the umbrella opened?"]` |

---

## 7. 数据生产与开训命令流水线

### 7.1 生成完稿侧修复/主动请求
```bash
python scripts/repair_requests.py \
  --parent "$DATA_DIR/targeted/parent" \
  --out "$DATA_DIR/requests_v2.jsonl" \
  --limit 6000 --shard-index 0 --shard-count 16
```

### 7.2 生产耦合配对数据（原生教师主路）
```bash
python scripts/prepare_coupled.py \
  --model "$MODEL_DIR" --source "$SOURCE_DIR" \
  --config configs/u15_isochronic_state/terminal_active.json \
  --requests "$DATA_DIR/requests_v2.jsonl" \
  --source-cache "$DATA_DIR/targeted/parent" \
  --teacher it2i --teacher-image source --teacher-prompt instruction \
  --output "$DATA_DIR/coupled_active_v2" \
  --shard-index 0 --shard-count 8
```

### 7.3 合并训练缓存并执行硬闸门检验
```bash
# 合并多角色缓存
python scripts/build_training_cache.py \
  --caches "$DATA_DIR/coupled_active_v2" "$DATA_DIR/r_v3" \
  --allow "$DATA_DIR/coupled_active_v2=R_active" \
  --allow "$DATA_DIR/r_v3=R_keep" \
  --output "$DATA_DIR/train_active_pool"

# 数据硬闸门严格抽样校验（必须通过才可启动训练）
python scripts/verify_training_data.py \
  --model "$MODEL_DIR" --source "$SOURCE_DIR" \
  --config configs/u15_isochronic_state/terminal_active.json \
  --native "$DATA_DIR/train_active_pool" \
  --draws 6
```
