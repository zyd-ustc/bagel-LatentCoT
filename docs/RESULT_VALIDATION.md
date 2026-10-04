# Base / LegacyMem / GEN / GEN+M 的结果验证

在仓库根目录执行以下命令。需要原生 BAGEL 权重、分别训练的 GEN 与 GEN+M v3 checkpoint、官方 GenEval2 环境，以及本地视觉语言模型 judge 权重。
GEN-only 和 GEN+M 必须使用各自的 checkpoint。不能用含 M 的 checkpoint 临时关掉 M 来代表已经训练的 GEN-only。

训练最高深度为 3，默认运行上限为 4。矩阵包含 Base R0，以及 LegacyMem / GEN / GEN+M 的 R1–4。
manifest 明确标注 R4 是否 unseen。共享 α 不保证 unseen depth 有收益，收益必须由实际结果验证。

## 生成配对图片

以下示例覆盖仓库内 128 条 hard prompts。两次运行保持 native checkpoint、prompt 顺序、seed、CFG、schedule、长宽、batch size 与 body 区间相同。
GEN+M 从第二个 manifest 读取，避免混用控制组。

```bash
python scripts/evaluate/t2i_loop_matrix.py \
  --model-path /path/to/BAGEL-7B-MoT \
  --checkpoint /path/to/trained_gen_only/step_001000 \
  --prompts experiments/data/geneval2_hard_128.jsonl \
  --output-dir outputs/geneval2_hard_gen \
  --modes legacy_memory_only,gen_only \
  --depths 0,1,2,3,4 --memory-slots 8 --max-prompts 128 \
  --start-layer 16 --end-layer 24 --seed 0 --batch-size 1

python scripts/evaluate/t2i_loop_matrix.py \
  --model-path /path/to/BAGEL-7B-MoT \
  --checkpoint /path/to/trained_gen_memory/step_001000 \
  --prompts experiments/data/geneval2_hard_128.jsonl \
  --output-dir outputs/geneval2_hard_gen_memory \
  --modes gen_memory_anchored \
  --depths 0,1,2,3,4 --memory-slots 8 --max-prompts 128 \
  --start-layer 16 --end-layer 24 --seed 0 --batch-size 1
```

每个 arm 的图片记录包含 index、seed、原 prompt、原 benchmark record、初始噪声哈希、长宽和 depth status。
矩阵默认开启逐轮诊断，其计时包含中间 suffix/readout。正式性能比较应关闭诊断另测普通推理。

## TIIF-spatial 与通用质量

先取得 [TIIF-Bench 的原始数据](https://github.com/A113N-W3I/TIIF-Bench)。转换保留原题、原 yes/no 答案和来源位置；只筛选 spatial 记录。

```bash
python scripts/evaluate/prepare_tiif_spatial.py \
  --source-dir /path/to/TIIF-Bench/prompt_jsonl \
  --output /path/to/normalized_tiif_spatial.jsonl \
  --description short_description
```

对 normalized TIIF 和独立的通用 prompt 集分别执行上一节的两次矩阵生成。修改 prompt 文件、输出目录和 `--max-prompts`，覆盖所选数据集的全部记录。
通用质量集每行至少含 `prompt`，可用 `height` / `width` 指定不同长宽。长宽须能被原生 latent_downsample 整除。
它应独立覆盖 ordinary、easy/noop 和结构任务，不能只用 hard composition prompts 推断通用质量。

## 运行统一 evaluator

按 [GenEval2 官方说明](https://github.com/facebookresearch/GenEval2) 准备其独立运行环境。配置填写 checkout 路径与对应 Python；本仓库调用原 `evaluation.py` 的 `soft_tifa_gm`，不改写原题。
本地 TIIF judge 使用原 yes/no 问题与答案，执行确定性的问答。该协议与 TIIF 官方随机模板/API judge 不同，报告将其标为本地 variant。
官方 GenEval2 的 float32 token 概率求和可因舍入误差略超过 1。结果解析仅将距 [0,1] 边界不超过 `1e-6` 的值归到边界；明显越界、NaN 和 Inf 仍报错。官方 `score_lists.json` 保持原值，报告 provenance 记录此容差。
quality judge 独立判断画面连贯性和可见瑕疵，输出 1–5 分并归一化到 0–1。它是质量代理，人工偏好单独统计。

修改 `configs/evaluation/t2i_loop_results.yaml` 中的实际路径和 manifest 选择。TIIF 与质量集也可参照 GenEval2 配置，用 `manifests` 按 mode 从两个独立 checkpoint 选择图片。

```bash
python scripts/evaluate/evaluate_t2i_loops.py \
  --config configs/evaluation/t2i_loop_results.yaml
```

输出 `scores.jsonl`、各数据集的逐图分数、`summary.json`、`summary.md` 和带图片哈希的人工配对 CSV。
记录包含 scorer 源码哈希、benchmark / manifest / 配置哈希、本地 judge 的配置哈希与协议。
官方 GenEval2 subprocess 先完成，再加载本地 judge，避免两套 judge 同时占用显存。

| 指标 | 定义与解释 |
| --- | --- |
| semantic AM / GM | 每张图片的原题 atom 分数做 arithmetic / geometric mean，再对图片求平均 |
| semantic GM Δ | 与同 index、seed 的 Base 做差；对配对样本 bootstrap，输出 95% 区间 |
| Constraint Repair | 单个原题 atom 在 reference 中不通过、loop 中通过；阈值默认 0.5 |
| Constraint Damage | 单个原题 atom 在 reference 中通过、loop 中不通过 |
| Constraint Repair rate | 修复约束数 / reference 失败约束数；分母为 0 时 null |
| Constraint Damage rate | 破坏约束数 / reference 成功约束数；分母为 0 时 null |
| Repair / Damage fraction | 报告占全部配对约束的比例，并另报每张图片全 atom 通过的 prompt-level Repair/Damage |
| quality proxy / Δ | VLM 代理评分与配对 Base 差值；差值包含 bootstrap 区间 |
| decode invalid | 无法读取或损坏的图片比例 |
| invalid | decode invalid 或 judge 判定 unusable；未运行 judge 时 null |
| human pairwise | loop/base/tie 的计数、已评审数量、总数量与配对偏好区间 |

Constraint Repair/Damage 同时对比初始 Base 和同模式的前一个 R。初始 Base 表用于判断 initially-wrong / initially-correct constraints；相邻 R 表用于检查后续 refinement 是否修复更多、破坏更少。
Prompt-level 通过要求全部 atom ≥ 阈值，保留在 `repair_damage`；两种计数不会混合。

每个 arm 必须覆盖所选 benchmark 文件的全部记录。配对覆盖、prompt、长宽或初始噪声不一致会报错。缺少必需 arm / depth 会报错。
损坏图片在已启用的 semantic / quality 指标中计失败，并保留错误原因。模型调用失败会中止，不能静默转成零分。
没有 scorer 的指标保持 null 和 `unscored`；velocity delta、训练 loss 和有效秩不参与结果评分。

在导出的 `human_pairwise_<hash>.csv` 填写 `preference=base|loop|tie`，再在配置中设置 `human_pairwise_csv` 指向它。
重复运行保留已有评审。图片改变时生成新的 CSV；导入旧图片的评审会因哈希不符而报错。
部分人工评审会保留 reviewed / total 数量，不能当作完整 benchmark 的人工结论。

## 当前证据范围

本地测试验证数据接口、真实图片文件读取、报告计算和失败处理。官方 scorer 与本地 VLM 的测试使用接口 fixture。
本次未执行完整 7B 模型、真实 GenEval2/TIIF judge 或人工质量评审。当前没有 semantic gain、quality retention 或 Repair-Damage 的实测结论。
