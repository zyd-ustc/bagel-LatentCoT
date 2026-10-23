# 显式闭环上界实验（不训练）

目标：在不动任何权重的前提下，测两个天花板。

1. **检出**：在 `t_read` 解码的预览图，原生 ViT-UND 能否发现最终图会犯的原子错误（数量、位置、颜色、属性）。
2. **纠错**：若检出，GEN 能否从 `t_early` 带着文字纠正重走并真正改对。

并给出**同算力对照**：同样在 `t_early` 回退，但只换 seed 重采、不改 prompt。GenEval++ 计数约 50% 成功率，随机重采本身能修好一半失败；显式闭环必须赢过它，否则隐式版本没有可追的上界。

## 协议

每次运行（单卡）对每条 prompt：

```text
base      50 步原生采样（CFG 4、shift 3、512px），保存 t_early 与各 t_read 的 latent
preview   在 t_read 解码 VAE 预览，用 ViT-UND 回答该 prompt 的 vqa_list
detect    任一原子答案不符 → 判定为失败
loop      回退到 t_early，把失败原子拼成约束句追加到 prompt，重走 t_early → 0
resample  回退到同一 t_early，用新 seed 对该状态重新加噪，原 prompt 重走 t_early → 0
```

- `loop` 与 `resample` 的 Euler 步数完全相同（`50 − rollback_index`），共享同一次 base 轨迹与同一次检出，因此是算力匹配的配对比较。
- `--full-resample` 额外跑一条全新 50 步、换 seed 的完整重采，作为非等算力参考。
- 修正文本由 `atoms.constraint` 确定性生成，例如：
  - 计数：`exactly seven croissants`（清单里的计数原子是**验证式**问题：`Are there exactly seven ...?`，避免开放式计数的系统偏差；实测开放式 “How many instances of bus” 会把单辆公交答成 2，而验证式能正确回答 No）
  - 属性：`the backpack is green`
  - 位置：`the croissant is to the left of the dog`
  - 存在：`there is a dog`
  不调用任何改写模型。

默认时间点（50 节点、shift 3）：

| 参数 | 节点 | t |
|---|---:|---:|
| `rollback_index` | 20 | 0.813 |
| `read_indexes` | 25 / 30 / 35 | 0.742 / 0.655 / 0.545 |

`read_indexes` 一次跑完、逐点解码，因此三个 `t_read` 共享同一条 base 轨迹。

## 运行

干跑（不需要 torch/NPU，用于核对协议与产物格式）：

```bash
python -m qwen_latent_cot.upper_bound.cli --dry-run \
  --manifest data/geneval2_800.jsonl --limit 8 --output /tmp/ub_dry
```

真实 NPU（建议先 32 条）：

```bash
python -m qwen_latent_cot.upper_bound.cli \
  --model /data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT \
  --manifest data/geneval2_800.jsonl \
  --config configs/upper_bound.json \
  --output /data/zyd_workspace/bagel-LatentCoT/outputs/upper_bound_32 \
  --device npu:0 --limit 32
```

官方榜单清单可直接使用（`manifests.py` 会把 include/exclude 元数据转成原子问题）：

```bash
--manifest /data/250010019/unimoe_r3/outputs/0_qwen38max_fullbench_20261002/manifests/geneval.jsonl     # 553
--manifest /data/250010019/unimoe_r3/outputs/0_qwen38max_fullbench_20261002/manifests/geneval_pp.jsonl  # 280
```

`data/geneval2_800.jsonl` 自带 `vqa_list`，两种格式都由同一个 loader 处理。GenEval++ 的 `region` 字段转成左右侧问题，官方空间指标仍以检测器为准，这里是 VQA 代理。

常用开关：

- `--read-index 25 30 35`、`--rollback-index 20`
- `--ids geneval2_0000 geneval2_0123`
- `--full-resample`
- `--seed 17`

## 产物

```text
<output>/images/                  base、preview_{idx}、loop_{idx}、resample_{idx}
<output>/records.json             每条 prompt 的原子答案、检出、两条重试结果
<output>/metrics.json             检出与重试的汇总
<output>/run.json                 配置、步数、臂定义
<output>/index.html               人工审图（预览 / base / loop / resample 并排）
```

`metrics.json` 的关键字段：

- `base.pass_rate`：不干预时的整图成功率；`base.by_skill` 按原子类型分。
- `read_points[].detection`：以“base 最终图是否失败”为标签，预览判失败的二分类混淆矩阵、precision/recall/F1、漏检率。
- `read_points[].retry`：在“base 失败且预览检出”的子集上，`loop` 与 `resample` 的整图成功率、四格计数、精确 McNemar `p`。
- `read_points[].atoms`：按技能统计失败原子在 loop/resample 后是否改对。

## 判读规则

- **检出**看 `recall`（漏掉多少必然失败的样本）与 `precision`（误触发多少额外重跑）。
- **纠错**只看 `retry` 子集：`loop_rate` 对 `resample_rate`。若两者持平，显式文字反馈没有超过“再抽一次”的收益；若低于，说明拼接约束反而破坏原生先验。
- **上界**取三个 `t_read` 中最好的检出—纠错组合，并注明对应 `t`；不能用最差 `t_read` 否证整个方向。

## 算力账

- 每条 prompt：base 50 步；每个检出的 `t_read` 追加 `loop` 与 `resample` 各 30 步（等算力）。
- 32 条、三个 `t_read` 全部检出时：`32 × (50 + 3×60) = 7,360` 次 Euler 更新。
- UND 问答：每条 1 次 base + 3 次 preview + 2×3 次重试判定 = 10 次，共 320 次；每次含一次 ViT 预填充。
- `--full-resample` 每检出再追加 50 步，不作等算力比较。

## 已知限制

- 预览与最终图都由同一个 native UND 判分，检出与纠错指标是自洽口径；等 27B 评分后端可用后应改用独立 judge。评分入口集中在 `UNDReader.score`，替换该函数即可。
- 高噪声端 VAE 预览偏糊，`t_read` 越早检出越难；本实验正是要量化这一点。
- 原子问题直接来自 GenEval 清单，答案匹配是规则式的（数字词/Yes-No/短语包含），不做语义改写判断。
- 单 seed、32 条是 pilot；置信区间与显著性只在配对子集上给 McNemar，不外推到全榜。

## 代码位置

```text
qwen_latent_cot/upper_bound/atoms.py      原子问题解析、答案匹配、约束句
qwen_latent_cot/upper_bound/manifests.py  官方 GenEval / GenEval++ 清单适配
qwen_latent_cot/upper_bound/native.py     原生采样、早停、回退、重加噪
qwen_latent_cot/upper_bound/reader.py     ViT-UND 问答
qwen_latent_cot/upper_bound/protocol.py   三臂协议与记录
qwen_latent_cot/upper_bound/metrics.py    检出与配对比较
qwen_latent_cot/upper_bound/gallery.py    人工审图页
scripts/upper_bound_closed_loop.py        CLI 入口（同 cli 模块）
tests/test_upper_bound.py                 原子/指标纯逻辑测试
```
