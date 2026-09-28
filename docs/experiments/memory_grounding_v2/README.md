# Memory grounding v2

当前主线：先训练 GEN 消费 memory，再训练 Read writer 的 downstream effect。
不再用 source/target hidden delta 作为必须完成的前置目标。
本次仅完成本地代码、CPU 数值验证；没有训练结果或语义提升结论。

## 固定计算协议

K=8，body=[12,20)，fresh，same-depth。显式 `num_read_rounds=1`，
`num_write_rounds=W`；总 body 次数为 1+W。默认 A/B/C 为 W=1，D 为 W=3。
每个 Write 的 non-memory hidden 重置到 prefix 的 body-entry；只有 memory
从前一轮延续，不是完整 GEN hidden recurrence。

`write_memory_override` 只替换第一轮 Write 入口的 [B*K,D] memory，不能和
旧 prompt-memory、full-depth、persist、字符串替换模式混用。零 memory
注入后正常演化，绝非 static-null。`write_memory_update=False` 是底层
Write 内冻结接口，不属于默认评测组。

`mask_prompt_kv_during_write` 只影响 body 的 Write：non-memory 查询
（GEN 和 boundary，防止 relay）不能读取 prompt KV，memory 仍可读。
prefix、Read、suffix 都不受这个开关影响。默认评测始终关闭 mask。
A 按更新步把 mask 概率从 .50 线性降到 .10（3000 步）；同一样本的三臂
使用完全相同的 mask、prompt、x_t、t。不同 batch 成员共用抽样 timestep。

v2 UND-Q 只在 Read 的 memory 行启用，Write 中不启用 UND-Q adapter。
GEN-Q（可选后续 GEN-O）只在 Write 中启用。base/KV/FFN/Norm/m0/VAE/ViT
及投影全部冻结。prefix/suffix/native teacher 显式关闭所有 loop adapter。

## 阶段与代码

| 阶段 | 入口 scripts/train/ | 优化对象 | 目标 |
|---|---|---|---|
| A | bagel_gen_memory_grounding.py | GEN-Q | frozen/detached Read；native teacher + shuffle/zero ranking + effect direction |
| B | bagel_memory_effect_grounding.py | UND-Q memory rows | 冻结 GEN reader；native teacher 或 edit target-flow effect |
| C | bagel_memory_grpo.py | GEN-Q，后续可联合 UND-Q | 真实 GenEval semantic + correct−shuffle causal reward − FLUX quality penalty |
| D | bagel_loop_supervision.py | UND-Q + GEN-Q | 每个 Write 的完整 suffix→norm→llm2vae 监督 |

数学在 `memory_grounding.py` / `loop_supervision.py`；统一模型与状态回放在
`memory_training.py`。入口共享 `memory_stage_runner.py`；Flow-GRPO 复用
现有 SDE transition、paired advantages、PPO clip、Gaussian KL。

A 的 teacher reconstruction 和 ranking 使用逐样本 teacher energy 归一化，
floor=1e-6。shuffle 是 whole-sample derangement，绝不 shuffle memory slots。
batch_size 至少 2 且 T2I prompt 文本不得重复（Edit 则要求 source+instruction
条件不重复）；本实现逐样本执行前向以减少缓存占用。
每次 native no-grad rollout 保存 3 个抽样状态，供接下来的 3 次更新复用。
B-Edit 设置 objective/state_source 都为 `target_flow`；pair 仅提供 source
condition 和 target VAE latent，目标为 ε−x1，不构造 hidden-space target。

D 每轮输出经过完整 suffix 和 norm。`final_velocity` 就是最后一个
`write_round_velocities`，Read 输出不计入监督。DS 权重默认 [.3,.5,1.]；
monotonic 权重 .1；Loop Distill 默认关闭，启用时最终 teacher stop-gradient。
SMA/gated recurrence **没有实现**，配置会明确拒绝；是后续有证据再做的项目。

## Stage A 的 H200 单卡命令（未代跑）

在远端项目目录内，用已安装 PyTorch/Transformers 的 lcot 环境：

```bash
python scripts/train/bagel_gen_memory_grounding.py \
  --config configs/training/memory_reader_grounding.yaml \
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT \
  --output-dir /private/yida_workspace/outputs/memory_reader_v2_smoke \
  --max-steps 2 --max-prompts 2 --validate-only
```

预检通过后移除 `--validate-only` 才实际加载模型、跑 2 步训练。
正式训练使用新的 output-dir，去掉 max-steps/max-prompts 两个 smoke 限制。
这里 `--validate-only` 检查配置、数据和路径，不证明权重文件完整或显存充足。
A/B/D 支持 CUDA `torchrun` 同步数据并行：分片采样、LoRA 梯度平均、主 rank
统一保存。C（GRPO）及统一评测仍为单进程。Stage A 的完整 8 卡启动入口见
[多卡训练说明](multigpu/README.md)；不能把八个独立单卡任务写进同一个目录。

输出包括 resolved_config、实际 trainable tensor 名、每步 metrics.jsonl、
adapter safetensors、同名架构 JSON、optimizer.pt、完成时 status.json。
非空输出目录会拒绝覆盖。`--adapter-path` 是权重 warm-start，不是自动恢复
optimizer/数据游标；当前没有精确断点续训 CLI。

## 后续阶段门槛

B/C 必须填 adapter_path，检查点 schema 必须为 `bagel-memory-grounding-v2`，
同目录必须有同名 JSON，K/body/rank/alpha/GEN-O/writer-read-only 均严格校验。
旧 phase1 checkpoints 不会被静默当作 v2 初始化。

先用 held-out 评测确认 unmasked correct 的 error 稳定小于 shuffled、效果非零，
再在对应 YAML 中将 `reader_dependency_validated` 改为 true。该字段是操作者
对证据的确认，**不是程序已经证明 gate 通过**。
D 还需确认 `semantic_quality_validated`；只有 final Write 确实优于第一轮时，
才同时设置 `final_round_validated: true` 和 `loop_distill: true`。
GEN-O 扩展必须显式开启 reader_o_enabled 且先通过 reader gate。

C 必须配置真实 GenEval 服务、FLUX quality reward 代码/配置/权重和 per-atom
metadata。无 reward 服务不会生成替代的“语义分数”。同 prompt 的 group seeds
配对 correct/shuffled/native/zero；初始及随机转移噪声一致。只对 correct
轨迹选定 SDE steps 反传，KL reference 固定为本轮训练开始时的 adapter。

## 统一四组推理评测

```bash
python scripts/evaluate/bagel_memory_causality_eval.py \
  --config configs/evaluation/memory_causality.yaml \
  --model-path /private/yida_workspace/models/BAGEL-7B-MoT \
  --adapter-path /path/to/reader_step_0000100.safetensors \
  --output-dir /private/yida_workspace/outputs/memory_v2_eval_step100 \
  --max-prompts 16 --generate-images
```

native / zero / shuffled / correct 都从相同 seed/noise 开始，独立推进轨迹。
另在同一 native rollout 状态上做三臂回放，输出 teacher error、dependency
gap、direction cosine、relative Δv、body 12/15/19 的 attention mass，及
逐 Write 完整 velocity 的 flow error。不同 arm 不能修改 cache、state 或 Read memory。
评测默认从 checkpoint 读取 W/rank/alpha/GEN-O；可以显式 `--num-write-rounds`
改变 W 做后续消融。无 checkpoint 可运行训练前冻结 baseline。

**当前 v2 统一采用条件速度 CFG=1**，训练与四臂评测均相同，拒绝多分支 CFG。
这是新协议，不可直接与历史 CFG=4 的 normal-R 图作定量质量比较。
历史 normal-R CLI 保留原来的协议，不会替代这里的新训练后评测入口。
HTML 仅在 `--generate-images` 时生成，不下载图片即可通过旧 SSH 隧道方式查看。

默认数据为 hard heldout 的前 16 条；训练数据为 hard_rl_train（320 条），
heldout 文件共 80 条。metrics/summary 中记录的是 velocity 因果诊断，
**默认不调用语义/美学 scorer**；不能把 velocity gap>0 当作语义质量提升。
在评测 YAML 配置 geneval_url 可评分四臂 semantic；配置 diffusion_rm_repo、
flux_rm_config、flux_rm_checkpoint 可评分四臂 quality（都需 --generate-images）。
真实评分写入 image_scores.jsonl，并汇总 correct−shuffled semantic 和
correct−native quality；服务缺失会失败，不会生成伪分数。

## 本地验证与回退

```bash
python -m pytest -q
```

测试覆盖实际 tiny MoT decoder、Bagel velocity、Read/Write 梯度、mask 矩阵、
cache 不变、逐轮完整 suffix parity、teacher detach、配置 fail-closed。
备份路径见 PLAN.md；旧训练入口、评测结果未删除。
当前目录 `.git` 指向不存在的 Linux worktree，未修改它，也未创建 commit/push。
