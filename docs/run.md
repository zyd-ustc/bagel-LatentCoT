# 复现手册

本文件覆盖：环境、原生权重与语料、单机/64 卡训练、三路采样、续训、评测与排障。所有命令在仓库根目录执行。

## 1. 环境

- Python ≥ 3.10
- PyTorch ≥ 2.5.1；Ascend 另装与驱动匹配的 `torch_npu`
- `transformers` 4.56–4.58（`pyproject.toml` 已约束）

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

测试使用真实的小型 BAGEL MoT，CPU 上即可跑完（当前 21 项）。它们验证的是训练/采样契约，不代表全模型数值行为。

## 2. 原生权重与语料

原生权重目录（例如 `Bagel-7B-MoT`）必须保持原样，训练与采样都会校验路径一致；不要在训练后替换目录内容。

语料是可复用的注释索引，构建时只读原始记录：

```bash
python scripts/prepare_photo_corpus.py \
  --source /absolute/path/observe_feedback_corpus \
  --output /absolute/path/correction_photos_v2_20261022
```

- 只接受 generation view（`role=2`，成对图文记录），完整 caption，不截断、不做文本替换。
- 保留原有 family 划分与 split 身份；移除同图跨 split 冲突，validation 再按图片去重。
- 输出目录含 `manifest.json`（含 `fingerprint`）、`views/`、`selection.json`。训练启动时会把 fingerprint 写进 checkpoint 的 provenance；续训要求它一致。
- 本轮语料统计：150,068 条独立路径检查，有效 train 743,062 / val 734 / test 720。

## 3. 训练

单机（1 个 NPU）：

```bash
python -m qwen_latent_cot.correction.cli train \
  --model /absolute/path/Bagel-7B-MoT \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --config configs/correction_guidance_safe.json \
  --output /absolute/path/new-run \
  --device npu:0
```

也可以走 `scripts/train_correction_npu.sh OUT --corpus CORPUS [--config ...]`，它按 `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` 自动选择单进程或 `torch.distributed.run`。

4 节点 × 16 NPU（项目当前的 64 卡入口）：

```bash
# 需在 ModelArts 节点上执行；VC_WORKER_HOSTS 必须是 4 个主机名
python scripts/launch_correction_64.py \
  --code /absolute/path/deployed-code \
  --output /absolute/path/new-run \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --config /absolute/path/deployed-code/configs/correction_guidance_safe.json
```

启动器会：审查四节点、只停止本项目旧的训练进程（不碰无关作业）、检查依赖与代码 revision 一致、把配置复制到 `<output>_launch/config.json`、以 detached 方式拉起四个 16-rank worker，并写 `node0..3.log` / `node0..3.pid` / `run.json` / `launch.json`。

注意：脚本里的 `BASE=/data/zyd_workspace/bagel-LatentCoT`、SSH 端口 2222、4 节点假设与本机路径都是这套集群的约定，迁移到其它集群需要改这些常量。

输出目录：

```text
<output>/run.json                 配置、provenance、可训练参数量、指标含义、验证样本
<output>/train.jsonl              每步一行：loss、base_mse、delta_rms、各组梯度、显存、耗时
<output>/step_00N.pt              只含可训练参数、optimizer、sampler、provenance、每 rank RNG、calibration
<output>/validation_00N/          固定 8 条验证：四路 t 的 MSE + 三路生图
<output>/semantic_00N/            固定 32 条 prompt 的三路生图与轨迹
```

全新训练要求 `<output>` 不存在（`FileExistsError`），避免误覆盖历史 run。

健康快照：启动器会在 `<output>_launch/` 生成 `status.py`，读取最新日志并输出当前 step、loss、检查点、四节点 rank 数与错误尾部：

```bash
python <output>_launch/status.py
```

关键配置（`configs/correction_guidance_safe.json`）：5000 步、global batch 256、512px、lr 1e-4、warmup 100、clip 1、seed 17、`cfg_text_scale` 4、global renorm、`correction_space: post_cfg`、`correction_objective: conditional_no_anti_guidance`、保存 10/100/500/1000/2000/3000/4000/5000、语义检查点 100/1000/3000/5000。

`activation_checkpointing` 默认关闭：单卡实测峰值约 47.9 GiB、约 0.49 s/样本；开启后约 41.2 GiB、0.55–0.57 s/样本。

## 4. 采样

```bash
python -m qwen_latent_cot.correction.cli sample \
  --model /absolute/path/Bagel-7B-MoT \
  --checkpoint /absolute/path/new-run/step_001000.pt \
  --prompt 'a photo of three books' --seed 17 \
  --output /absolute/path/three-books.png --device npu:0
```

三条对照臂：

| 参数 | 含义 |
|---|---|
| `--native` | 完全原生，无纠错 |
| `--global-only` | 只用训练期估计的全局常量场（post-CFG 模型才有） |
| 默认 | 完整 UND 纠错器 |

采样轨迹写到图片路径加 `.json`（每步 `delta_rms`、`gen_passes`、`und_passes`，guidance-safe 模式还有 `raw_delta_rms` 与 guidance 投影系数）。采样会校验 checkpoint 里的 `model_path` 与当前 `--model` 一致。

## 5. 续训

```bash
python -m qwen_latent_cot.correction.cli train \
  --model ... --corpus ... --config ... --output <same-run> \
  --resume <same-run>/step_001000.pt --steps 5000 --device npu:0
```

续训是精确恢复：可训练参数、optimizer、数据采样器、每 rank RNG、calibration 统计都会还原。硬性要求：

- 同一个原生模型、同一个 corpus fingerprint；
- 同一份方法/优化配置（仅允许改 `steps`、`save_steps`、`val_images`、`activation_checkpointing`）；
- 同一个 world size（64 rank 的 run 不能用其它卡数续）；
- 64 卡续训需重新拉起四个节点的 `torchrun`，保持 `NODE_RANK=0..3`、`MASTER_ADDR` 为原 node0、`NPROC_PER_NODE=16`、`NNODES=4`，并换新日志。fresh-run 启动器会拒绝已存在的输出目录，恢复时直接用 `torch.distributed.run`，不要改删旧目录。

## 6. 评测

- **验证**：`validation_00N/records.json` 记录固定 8 条样本在 4 个时间点上的 native / global / corrected 物理 MSE。只在保存点（且为 100 的倍数）或最后一步运行。
- **语义**：`semantic_00N/` 是固定 32 条带标注 prompt 的三路图，native 图在同 run 的 `semantic_native/` 复用。它**不是** benchmark 成功率，必须人工看图或另跑正式评测。
- 指标解释规则见 [method.md](method.md#7-度量定义)：`loss` 与 `base_mse` 不可混算比例。

## 7. 测试与 pilot

```bash
python -m unittest discover -s tests -v     # 21 项契约测试

python scripts/pilot_correction.py \
  --model /absolute/path/Bagel-7B-MoT \
  --corpus /absolute/path/correction_photos_v2_20261022 \
  --output /absolute/path/pilot --device npu:0
```

pilot 跑多次更新、保存/恢复、三路生图，用来确认全模型在目标硬件上"真的在训练"，不作为质量结论。

## 8. 排障

| 现象 | 处理 |
|---|---|
| `FileExistsError: Fresh training requires a fresh output directory` | 换新路径；续训用 `--resume`，不要删除旧 run |
| `Resume native model/corpus provenance differs` | 模型目录或语料 fingerprint 与训练时不同 |
| `Resume optimization/method config differs` | 改了方法/优化字段；只允许改步数、保存点、验证数、activation checkpointing |
| `Exact resume requires the same world size` | 用相同卡数重新拉起 |
| `Frozen native/UND parameter was modified` | 有代码路径更新了冻结参数，检查改动 |
| OOM（约 48 GiB 峰值） | 在配置中打开 `activation_checkpointing` |
| `Nonfinite sampling velocity` | 检查 checkpoint 与精度路径，不要静默跳过 |
| `Requires a new spatial-correction checkpoint` | 旧架构 checkpoint 不能续训当前模型 |
| 启动器报 `Existing correction training is active` | 先确认并停止旧 run；`--audit-only` 可只做审查 |

## 9. 版本与 provenance

部署时在代码根目录写入 `DEPLOYED_COMMIT`（提交哈希），训练会把它的内容记入 checkpoint 的 `provenance.source_revision`；`run.json` 同时记录 config、world size、训练对数和可训练参数量。归档日志中的每轮实验都记录了对应 revision（见 [experiments.md](experiments.md) 与 [archive](archive/README.md)）。
