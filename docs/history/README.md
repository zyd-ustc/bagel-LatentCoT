# 历史实验索引

当前训练协议见 [Phase 1A v4](../experiments/phase1a_v4/README.md)。以下路线
保留用于复现和对照，不是当前 Warm-up → OPD 的阶段依赖。

| 路线 | 协议 | 入口 |
|---|---|---|
| Phase 0 / 0.5 zero-shot loop | [T2I Phase 0.5](../BAGEL_LatentCoT_T2I_Phase05.md) | `scripts/evaluate/bagel_loop_t2i_zeroshot.py` |
| Pair-grounded memory / flow | [Phase 1 paired](../experiments/phase1_pair_grounded/PLAN.md) | `scripts/train/bagel_loop_pair_memory.py`、`scripts/train/bagel_loop_pair_flow_sft.py` |
| Memory Grounding v2 | [v2 协议](../experiments/memory_grounding_v2/README.md) | `scripts/train/bagel_gen_memory_grounding.py` |
| normal-only R=2/4/6/8 | [R 消融](../experiments/normal_r_ablation/README.md) | `scripts/evaluate/run_bagel_memory_mechanism.sh` |
| FlowEdit 与显式反思 | [FlowEdit](bagel_flowedit_zeroshot.md)、[reflection loop](MOT_REFLECTION_LOOP_DESIGN.md) | `scripts/evaluate/run_bagel_flowedit_zeroshot.sh`、`scripts/evaluate/run_draft_prefix_loop.sh` |

## 清理边界

2026-10-03 删除两个旧 poll-and-launch 脚本：它们引用的
`bagel_loop_sft_train.py`、`bagel_loop_generate.py` 和 `loop_sft.yaml` 已不存在。
同时删除无人引用的 `scripts/common.sh`。旧版本可通过 Git 历史恢复。
不删除仍有入口、测试或复现用途的历史 Python 模块、配置和实验记录。
