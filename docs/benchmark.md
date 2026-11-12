# 文生图与编辑榜：单题生成和评分

该入口只处理一条指定记录，不会自动批量评测。它读取冻结清单的 `record_id`、原题和种子。生成与评分分开执行，评分只接受同一条记录生成的图片。

## 数据与支持范围

冻结清单和 CPI 评分模板已只读复制到 `modelarts-64` 的 `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/`。下表使用该目录中的实际路径。编辑榜的源图和评分元数据仍需按题复制，并按冻结记录校验。

### 文生图

| 榜单 | 冻结清单 |
|---|---|
| GenEval、GenEval++、T2I-CompBench++、DPG、T2I-CoReBench、WISE-Verified | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/0_eval62326_qwen27b_sixbench_20261019/foundation_v1/inputs/score_manifests/` 下对应的 `.jsonl` |
| OneIG-Bench、PRISM-Bench、SpatialGenEval、GenEval 2 | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/0_extra_bench_qwen21_20261027/inputs_four_bench_v1/manifest.jsonl` |
| TIIF-Bench、DSG、UniGenBench、GenAI-Bench | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/0_extra_bench_qwen21_20261027/inputs_recovered_offline_20261027/manifest.jsonl` |

五榜固定样本可用 `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/0_eval_official_t2i_fivebench10_20261108_v1/ROWS.json`，其中有 610 条记录。该文件保存了样本使用的原题和种子。

### 编辑

| 榜单 | 冻结清单 | 评分状态 |
|---|---|---|
| ImgEdit Basic / UGE | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/data/benchmark_editing7_20261030/imgedit/prepared784_v1/MANIFEST.jsonl` | GLM 原评分模板和解析规则 |
| MagicBrush | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/data/benchmark_editing7_20261030/magicbrush/prepared_single_v1/MANIFEST.jsonl` | 冻结 CLIP、DINO、L1 等工具 |
| Emu Edit | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/data/benchmark_editing7_20261030/emu_edit/prepared_single_v1/MANIFEST.jsonl` | 冻结 CLIP、DINO 等工具，沿用内部口径 |
| GEdit v2 | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/data/benchmark_editing7_20261030/gedit_v2/prepared_single_v1/MANIFEST.jsonl` | 已接成对评分；需要冻结评分模板和 `both` 生成结果，尚未进行真实裁判测试 |
| CPI 单图 | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/data/benchmark_editing7_20261030/cpi_bench/generation_single_raw_v1/MANIFEST.jsonl` | 生成入口可读取；评分尚未接入。需先确认清单中待审的画布协议 |
| ABACUS | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/data/benchmark_editing7_20261030/abacus/generation_single_raw_v1/MANIFEST.jsonl` | GLM 内部数量诊断；先确认清单中待审的提示词协议 |
| GEdit v1 | `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/0_qwen21_gedit_threeway_recovery_20261027/MANIFEST.jsonl` | GLM 复用 SC/PQ 模板和解析规则 |
| CPI 多图 | 多图输入 | 当前编辑器只接受单张源图，尚不支持 |

## 模型与裁判

以下在 `modelarts-64` 上执行。该节点可以访问 Metto，但没有挂载 `/mnt/sfs/250010019`：冻结清单已复制，编辑题的源图及所需元数据须另行复制。不要使用 `modelarts-64b` 的加速卡。

```bash
CODE=/data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU
MODEL=/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT
SOURCE=/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b
CKPT=/data/zyd_workspace/isochronic_data_20261107/train_full490k/step_014723
CONFIG=$CODE/configs/u15_isochronic_state/stage1_full490k.json
ENV_FILE=$CODE/.env
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
cd "$CODE"
```

在 `.env` 中填写 `METTO_GLM_KEY`。文件只由评分进程读取，不写入结果或版本库。默认服务为 `https://metto-api.com`，模型固定为 `glm-5.3-flash`。图片和题目会发送到该服务。一个单题可能包含多个裁判问题，每个问题各发一次请求；不自动重试。

## 文生图：生成一题

```bash
"$PY" -m qwen_latent_cot.benchmarks.cli generate --kind t2i --manifest /path/to/manifest.jsonl \
  --record-id 'geneval2:line-1' --output /path/to/new-t2i-output \
  --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --checkpoint-mode trained --checkpoint "$CKPT"
```

输出包含 `preview.png`、`baseline.png`、`result.png` 和 `case.json`。基线和闭环共享起始状态、原题和种子。反馈判定已满足时，`result.png` 与基线相同；反馈解析失败时，生成中止，不把失败当作已满足。

## 编辑：生成一题

```bash
"$PY" -m qwen_latent_cot.benchmarks.cli generate --kind edit --manifest /path/to/manifest.jsonl \
  --record-id 'imgedit_basic_1082' --output /path/to/new-edit-output \
  --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --checkpoint-mode trained --checkpoint "$CKPT"
```

评测入口将源图等比放进黑色 512×512 画布，生成后按同一矩形去除填充，再缩放到源图尺寸。`source_canvas.png` 和 `result_canvas.png` 保留模型实际输入输出，`result.png` 用于评分。此处理只用于评测，不改变现有训练和推理入口。它与历史 1024 像素结果的输入条件不同，比较时必须报告。

如果生成机器不能访问清单中的源图，可先复制**该题源图**，再加 `--source-image /path/to/copy.png`。程序校验副本与清单的 `source_sha256` 一致。文生图不使用清单里的历史图像路径。

两种生成均可用 `--checkpoint-mode native` 代替 `trained`，并省略 `--checkpoint`。这是未加载检查点但仍执行编辑的对照，**不是**不编辑的原生基线。使用 `--checkpoint-mode both --checkpoint "$CKPT"` 可在同一题中生成无检查点 `native_result.png` 和检查点 `result.png`。文生图的三组共享原始状态和同一份 UND 反馈；编辑榜两组共享源图和同一种子。

## 评分一题

评分命令重新读取同一冻结记录，检查 `case.json` 里的榜单、`record_id`、原题、种子及图片绑定，才调用裁判：

```bash
"$PY" -m qwen_latent_cot.benchmarks.cli score --kind t2i --manifest /path/to/manifest.jsonl \
  --record-id 'geneval2:line-1' --generation /path/to/new-t2i-output \
  --arm result --env-file "$ENV_FILE" --output /path/to/t2i-score.json

"$PY" -m qwen_latent_cot.benchmarks.cli score --kind edit --manifest /path/to/manifest.jsonl \
  --record-id 'imgedit_basic_1082' --generation /path/to/new-edit-output \
  --env-file "$ENV_FILE" --output /path/to/edit-score.json
```

文生图可把 `--arm result` 改为 `--arm baseline`，单独评分原生基线。`both` 模式下还可用 `--arm native_result` 评分未加载检查点的编辑结果。编辑评分默认读取原始源图；若评分机器只能访问已校验副本，也可传 `--source-image`。DPG 的问题依赖冻结评分数据，默认从 `/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/scoring_sources` 读取；该文件复制自冻结评分数据，也可用 `--source-root` 指向已复制的数据。

MagicBrush 和 Emu Edit 使用独立的本地 CLIP、DINO 与像素指标，不调用 GLM。运行前需将冻结指标资产复制到 `modelarts-64`，使用 `--metrics-assets` 指定目录。可用 `--metadata-file` 和 `--reference-image` 指定复制后的元数据与目标图；程序根据冻结清单检查它们。MagicBrush 还需要原始 `local_descriptions.json`，通过 `--captions` 指定；复制 `READY.json` 后可用 `--prepared-ready` 指定，以校验清单和说明文件。这两榜应在能访问冻结评测数据和 `tool_assets` 的机器及对应 CPU 环境中运行。MagicBrush 参考目标图，Emu Edit 的图像相似度参考源图。ABACUS 诊断也可用 `--metadata-file` 指定已复制的冻结元数据。MagicBrush、Emu Edit 和 ABACUS 的这些复制路径尚未做真实评分测试。

## 分数含义

评分保留各榜的冻结问题、模板和单图聚合规则，但裁判已经从 27B 改成 GLM。GenEval 2 补充了简短答案的格式提示，并允许解析器接受精确的裸答案（yes、no、unknown 或单个数字）；解释性文字仍记为无效。输出是 **GLM 内部口径**，不是官方榜单分，也不能直接当成原 27B 数值的同口径对比。模型、榜单、题目、图片、原始回答和有效状态均写入单题结果。不同榜单的量纲保持原样，不合并为总分。GEdit v2 的 `--rubric-root` 指向冻结模板仓库；`--reference-arm` 指定比较图，两种图片顺序分别判断，平票记半胜。

评分无效或无法判断时，结果为 `unknown` 或命令报错，不计作零分。正式比较时，分别报告各组覆盖率，并只在共同有效题上计算配对差值；类别宏平均的榜单保持类别宏平均。

## 单题联通记录

`modelarts-64` 上已用 `step_014723` 和不加载检查点两种模式生成 GenEval 2 与 ImgEdit Basic 各一题，输出位于 `/data/zyd_workspace/isochronic_data_20261107/benchmark_migration_smoke_20261112/`。ImgEdit 题 `imgedit_basic_1082` 的检查点结果已通过 GLM 实际评分：三个维度各 3 分，平均 3.0（1–5 分）。这只是管线联通结果，不代表模型的榜单表现。文生图题 `geneval2:line-1` 的检查点结果已通过 GLM 实际评分：5 个问题全部匹配，单题为 100 / 100。首次返回过不符合格式的回答，该次结果保留为 `unknown`；约束答案格式并允许精确裸答案后，完整重测结果保存为 `t2i-score-final.json`。该单题的 UND 反馈为已满足，闭环结果与基线相同，不能据此判断编辑收益。
