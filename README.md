# bagel-LatentCoT

BAGEL-7B-MoT 上的 **步内 understanding–generation hidden loop**。

主方案不是外循环编辑 agent，也不是 FlowEdit。每个去噪步里固定跑 \(R\) 轮：\(K\) 个 memory token 走 understanding expert，和当前 VAE/gen token 做原生 joint attention，只用最后一轮速度推进 \(x_t\)。

设计文档：[docs/BAGEL_MoT_Latent_Loop_MDP_Research_Design.docx](docs/BAGEL_MoT_Latent_Loop_MDP_Research_Design.docx)

## 现在做到哪

| 阶段 | 状态 |
|---|---|
| **Phase 0** | 已实现。默认 **same-depth body loop**：prefix 一次、只在 \([s,e)\) 上把 memory slots recycle \(R\) 次、suffix 一次。`full_depth` 仍可作为对照。CFG 三条分支各自维护 memory。`K=0` 走原 `_forward_flow`。 |
| Phase 1+ | 未做：双侧 attention LoRA、Loop-SFT / DS+LD、Flow-GRPO |
| **B2 FlowEdit** | 官方速度场差速积分，对照「纯 flow 编辑」 |
| **B3 显式反思链** | `draft_prefix_loop`：decode → UND 文本 → 官方 Editing |

## 代码

```
qwen_latent_cot/bagel/           # 官方 BAGEL + Phase-0 loop
  modeling/bagel/bagel.py        # prepare_vae_latent / _forward_flow / _forward_flow_loop
  inferencer.py                  # InterleaveInferencer + gen_image_flowedit
scripts/evaluate/                # B2 / B3 / GenEval2
scripts/train/                   # 后续 GRPO / semantic-state（非 Phase-0 主线）
experiments/data/                # GenEval2-hard 16
docs/                            # 主设计 + FlowEdit / 外循环说明
tests/test_mot_loop_phase0.py
```

启用 Phase-0 memory（默认 `K=0`，不改官方路径）：

```python
BagelConfig(
    ...,
    num_loop_tokens=8,
    loop_depth=2,
    loop_recycle_mode="same_depth",  # or "full_depth"
    loop_memory_persist=True,        # False = reset m every timestep
    memory_loop_start_layer=20,
    memory_loop_end_layer=28,
)
# load checkpoint 后
model.init_loop_memory_from_boundary_embeddings(
    [new_token_ids["start_of_image"], new_token_ids["end_of_image"]]
)
```

## 安装

```bash
pip install -e '.[dev]'
pytest -q tests/test_mot_loop_phase0.py tests/test_bagel_flowedit.py
```

权重用官方 [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)（需要完整 `ema.safetensors`）。

## 对照实验

FlowEdit 4.1（文档 B2），16 卡：

```bash
bash scripts/evaluate/run_bagel_flowedit_zeroshot.sh /path/to/out
```

Draft-prefix 外循环（文档 B3）：

```bash
bash scripts/evaluate/run_draft_prefix_loop.sh /path/to/out
```

官方 T2I / Editing / Understanding 接口见 [inference.ipynb](inference.ipynb)。
