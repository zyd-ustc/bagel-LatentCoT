# bagel-LatentCoT

BAGEL-7B-MoT 上的 **步内 understanding–generation hidden loop**。

主方案不是外循环编辑 agent，也不是 FlowEdit。每个去噪步里固定跑
`1 read + (loop_depth - 1) write`：\(K\) 个 memory token 走 understanding
expert，和当前 VAE/gen token 做原生 joint attention，只用最后一轮速度推进
\(x_t\)。当前方法称为 **Read–Write Loop with implicit context rerouting**。

设计文档：[docs/BAGEL_MoT_Latent_Loop_MDP_Research_Design.docx](docs/BAGEL_MoT_Latent_Loop_MDP_Research_Design.docx)

## 现在做到哪

| 阶段 | 状态 |
|---|---|
| **Phase 0** | 已实现。默认 **same-depth body loop**：prefix 一次、只在 \([s,e)\) 上把 memory slots recycle \(R\) 次、suffix 一次。Round-0 禁止所有 non-memory query 读取 memory，关闭跨层 UND relay。CFG 三条分支各自维护 memory；`K=0` 走原 `_forward_flow`。 |
| Phase 1+ | Q-only loop LoRA 与 Flow-GRPO replay 已接通；text-reflection → latent-loop 的 Δv distillation trainer 未实现。 |
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

Phase 0.5 是 frozen BAGEL 的纯 T2I compositional mechanism benchmark。所有 arm
共享 prompt、initial noise、1024×1024 geometry、官方 T2I CFG、50-step schedule，
只改变 `K/R/body/persist/read-write`。16 卡主矩阵：

```bash
bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /path/to/out
```

K 消融与主矩阵分开运行：

```bash
K_VALUES=1,4,8 bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /path/to/k_ablation
```

主矩阵固定为 Z0 vanilla、Z1 old loop、Z2 strict read→write、Z3 early、
Z4 late、Z5 persist、C0 full-depth、C1 read-only 与 C2 immediate-write。
生成完成后会写 `mechanism_summary.json` 及每个 arm 的 GenEval2 image map；启动
Soft-TIFA server 后用 `score_bagel_loop_t2i_geneval2.py` 汇总 AM/GM、skill 和
atomicity。`K_VALUES` 非空时不得同时设置 `ARMS`。

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
