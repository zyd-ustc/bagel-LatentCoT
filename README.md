# bagel-LatentCoT

BAGEL-7B-MoT 的 latent memory 研究实现。当前路线：
**Phase 1A.0 Reader Warm-up → Phase 1A.1a Self-CoT OPD**。
主分支为 `main`；历史实验与当前训练依赖分开记录。

## 当前实现

1. Reader Warm-up：冻结 BAGEL 和 memory writer，用原生 prompt-bank readout
   监督独立 memory reader，仅训练低秩 translation A/B。此阶段不向生图主路注入。
2. Self-CoT OPD：加载并冻结 reader，在 student states 上训练零初始化 injection
   gates，拟合 frozen `[prompt; reasoning]` teacher velocity。

OPD 是 on-policy distillation（在 student 自己生成的状态上进行蒸馏）。
两阶段固定 K=8、body `[12,20)`。Warm-up 支持 CUDA 多卡同步和断点续训；
OPD 当前仍为单卡入口。运行前必须通过配置、数据和 artifact 校验。

## 安装与检查

在仓库根目录执行。完整训练需要兼容 CUDA/PyTorch 环境，以及官方
[BAGEL-7B-MoT 权重](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)。

```bash
pip install -e '.[dev]'
python -m pytest -q
```

## 常用入口

| 用途 | 入口 |
|---|---|
| 导出 prompt-only 训练和 heldout 数据 | `scripts/data/prepare_reader_warmup_prompts.py` |
| Reader Warm-up、续训 | `scripts/train/bagel_memory_reader_warmup.py` |
| 8卡新训练：数据导出、smoke、正式训练 | `scripts/train/launch_reader_warmup_8gpu.sh` |
| 独立检查 reader checkpoint | `scripts/evaluate/bagel_memory_reader_warmup_eval.py` |
| Self-CoT teacher 数据与 OPD | `scripts/data/build_bagel_cot_teacher.py`、`scripts/train/bagel_memory_opd.py` |
| checkpoint 生图、评分清单与离线诊断 | `scripts/evaluate/bagel_memory_opd_eval.py`、`scripts/evaluate/prepare_phase1a_score_inputs.py`、`scripts/evaluate/analyze_reader_warmup.py` |

完整参数、运行顺序和进入 OPD 的两项门槛见
[Phase 1A 运行说明](docs/experiments/phase1a_v4/README.md)。
续训时 `--max-steps` 是最终总步数，必须使用全新输出目录；不得覆盖旧 run。

## 已有结果与证据范围

20474 的8卡 Reader Warm-up 已完成到 step-5000。最终8条 heldout prompt 的
readout MSE：correct=2.5826、shuffled=3.0956、zero=2.7046；native parity=0。
这是重建误差，不是生图质量评分。OPD 未自动启动，语义生图评测仍待执行。

- [完整训练和检查点记录](docs/experiments/phase1a_v4/resume_20474_20261002/RUN.md)
- [64条独立生图评测清单、评分校验和数据审计](docs/experiments/phase1a_v4/offline/README.md)
- [历史实验索引](docs/history/README.md)

## 项目结构

```text
qwen_latent_cot/bagel/       BAGEL、memory reader、Warm-up 与 OPD
qwen_latent_cot/evaluation/  GenEval2、评分校验、离线诊断
scripts/                    数据、训练与评测入口
configs/                    训练与评测配置
experiments/data/           固定 prompt 和评测清单
tests/                      CPU 单元与契约测试
docs/                       当前协议、实验记录与历史索引
```
