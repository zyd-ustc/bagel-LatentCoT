# bagel-LatentCoT

BAGEL-7B-MoT 上的 latent-condition research code。Phase 0.5 当前主线是
**Anchored Dynamic Prompt Memory**：原生 prompt KV 始终作为 anchor，active
prompt-copy 在固定 \(x_t\) 上做 Read，随后只把同层 \(\Delta V_P\) 残差写回
GEN 的原生 prompt attention route。不再把 K=8 side memory 当成 semantic carrier。

设计文档：[docs/BAGEL_MoT_Latent_Loop_MDP_Research_Design.docx](docs/BAGEL_MoT_Latent_Loop_MDP_Research_Design.docx)

## 现在做到哪

| 阶段 | 状态 |
|---|---|
| **Phase 0** | 已实现。默认 **same-depth body loop**：prefix 一次、只在 \([s,e)\) 上把 memory slots recycle \(R\) 次、suffix 一次。Round-0 禁止所有 non-memory query 读取 memory，关闭跨层 UND relay。CFG 三条分支各自维护 memory；`K=0` 走原 `_forward_flow`。 |
| **Phase 0.5** | 已重构。保留 full-length native prompt anchor；`[12,20)` Read 产生同层 `ΔV`，Write 仅注入 `V₀ + αΔV`；默认只开 early 35% denoising steps。 |
| **Phase 1.1** | 已实现 Pair-Grounded Memory Read：同一 target-noised state 上用 source/target frozen visual reference 构造 memory delta，只训练 `[12,20)` UND-Q LoRA。 |
| **Phase 1.2A** | 已实现 Target Flow SFT：加载并冻结 Phase 1.1 UND-Q，只训练 GEN-Q，直接拟合 `epsilon - x1`；structured reflection 降级为 ablation。 |
| Phase 1.3+ | 等 1.1/1.2 go gate 后再做 joint relaxation、memory swap、persist 与 RL。 |
| **B2 FlowEdit** | 官方速度场差速积分，对照「纯 flow 编辑」 |
| **B3 显式反思链** | `draft_prefix_loop`：decode → UND 文本 → 官方 Editing |

## 代码

```
qwen_latent_cot/bagel/           # 官方 BAGEL + Phase-0 loop
  modeling/bagel/bagel.py        # prepare_vae_latent / _forward_flow / _forward_flow_loop
  inferencer.py                  # InterleaveInferencer + gen_image_flowedit
scripts/evaluate/                # B2 / B3 / GenEval2
scripts/train/                   # Phase 1 paired memory/flow + 后续 GRPO
experiments/data/                # GenEval2-hard 16
docs/                            # 主设计 + FlowEdit / 外循环说明
tests/test_mot_loop_phase0.py
```

Phase 1 新主线入口：

```bash
# 1.1: prefix -> strict Read -> STOP; UND-Q only
python scripts/train/bagel_loop_pair_memory.py \
  --config configs/training/loop_pair_memory_early.yaml

# 1.2A: Read -> Write -> suffix; frozen UND-Q + trainable GEN-Q
python scripts/train/bagel_loop_pair_flow_sft.py \
  --config configs/training/loop_pair_flow_early_fresh.yaml \
  --read-adapter /path/to/pair_memory_adapter.safetensors
```

完整协议见
[`docs/BAGEL_LatentCoT_Phase1_Pair_Grounded_Memory_Plan.md`](docs/BAGEL_LatentCoT_Phase1_Pair_Grounded_Memory_Plan.md)。
旧 `bagel_loop_delta_v_distill.py` 仅保留作 structured-reflection ablation。

启用 Phase-0 memory（默认 `K=0`，不改官方路径）：

```python
BagelConfig(
    ...,
    num_loop_tokens=8,
    loop_depth=2,
    loop_recycle_mode="same_depth",  # or "full_depth"
    loop_memory_persist=False,
    memory_loop_start_layer=16,
    memory_loop_end_layer=24,
    round0_memory_write_enabled=False,  # strict read round
)
# BagelBackbone.load() 会用原生 SOI/EOI embedding 确定性初始化 m0。
```

`return_loop_diagnostics=False` 是正式推理/训练默认值，此时严格只跑
`prefix ×1 + body ×R + suffix ×1`；机制探针显式设为 `True` 才计算逐轮
`ΔM / ΔG / Δv`。

旧配置 `round0_gen_reads_memory` 仍可读取，但已由语义准确的
`round0_memory_write_enabled` 取代。实验 metadata 同时记录
`num_read_rounds` / `num_write_rounds`，例如 `loop_depth=2` 表示 `1R + 1W`。

## 安装

```bash
pip install -e '.[dev]'
pytest -q tests/test_mot_loop_phase0.py tests/test_bagel_flowedit.py
```

权重用官方 [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)（需要完整 `ema.safetensors`）。

## 对照实验

Phase 0.5 先做固定 \(x_t\) 数值验证，再跑端到端 trajectory。两种模式都共享
prompt、initial noise、geometry、CFG 与 schedule；`alpha=0` 直接走原生
`_forward_flow`，用于 exact native parity。

```bash
python scripts/evaluate/bagel_dynamic_prompt_phase05.py \
  --mode probe --model-path /path/to/BAGEL-7B-MoT \
  --output-dir /data/outputs/dynamic_prompt_probe \
  --alphas 0,0.1,-0.1,0.2 --max-prompts 16
```

端到端 hard16：

```bash
python scripts/evaluate/bagel_dynamic_prompt_phase05.py \
  --mode generate --model-path /path/to/BAGEL-7B-MoT \
  --output-dir /data/outputs/dynamic_prompt_hard16 \
  --alphas 0,0.1,-0.1,0.2 --body-start 12 --body-end 20 \
  --step-fraction 0.35 --max-prompts 16
```

probe 的 `manifest.json` 记录 `relative_l2`、
`cos(Δv(+a), -Δv(-a))`、`2a/a` scaling 与逐层 `ΔV` norm；generate 模式保存
每个 alpha 的图和逐 step 诊断。当前不做 prompt compression、persistent memory、
K residual 或 learnable gate。

原 paired-edit specification 与 runner 保留在
`experiments/data/semantic_edit_phase05.jsonl` 和
`scripts/evaluate/bagel_loop_edit_zeroshot.py`，留待 Phase 2 editing 使用。
完整协议、arm 定义和评分命令见
[`docs/BAGEL_LatentCoT_T2I_Phase05.md`](docs/BAGEL_LatentCoT_T2I_Phase05.md)。

FlowEdit 4.1（文档 B2），16 卡：

```bash
bash scripts/evaluate/run_bagel_flowedit_zeroshot.sh /path/to/out
```

Draft-prefix 外循环（文档 B3）：

```bash
bash scripts/evaluate/run_draft_prefix_loop.sh /path/to/out
```

官方 T2I / Editing / Understanding 接口见 [inference.ipynb](inference.ipynb)。
