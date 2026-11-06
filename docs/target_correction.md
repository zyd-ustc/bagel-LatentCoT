# 等时生成态编辑：实现合同

活动实现标识：`kind="u15_isochronic_state_flow_v1"`。方案见 [等时生成态编辑](isochronic_state_flow_training_plan.md)。旧的选择性历史编辑器、原生编辑 CFG 和长程展开训练已删除，旧检查点不能按这个 kind 加载。

## 1. 变量和参数化

- `t`：原生生成时间，0 是噪声，1 是图像。状态编辑期间固定。
- `s`：状态编辑进度，0 是输入状态，1 是转换后的状态。
- `a`：时间 `t` 的输入状态；`b`：同一时间、在 `c_tail` 下可接续到目标的配对状态。
- `P`：从 `a` 的副本、在 `c_parent` 下调用统一前瞻函数得到的预览。
- `b0 = sg[H0(a,t;P)]`：关闭 LoRA 的原生仅图分支 clean prediction，每次编辑只由锚点 `a` 计算一次。
- `uθ(q,t,s) = t [Hθ(q,t,s;P,e,c_tail) - b0]`：每单位 `s` 的状态位移，不是原生去噪速度。

训练路径：

```text
q_s = (1-s) a + s b
u*  = b - a
L   = mean_valid(((uθ(q_s,t,s) - (b-a)) / t)^2)
```

`b0` 不反传。`Hθ` 通过 GEN LoRA 和 `edit_s_embedder` 训练。不裁剪 `q_s`、`Hθ` 或 `uθ`，不加 CFG，不除以 `1-t`。

## 2. 模块

- `model.py`：`GENLoRA`、模块视图、冻结 `image_clean`、可训练 `state_clean/state_velocity`、`edit_s_embedder`。两个 UND 前缀分别是 image-only 与 joint；图像编码在两次 prefill 间共用。
- `field.py`：配置合同、`NativeT2I`、`preview_image`、固定 `t` 的 `edit_state`、逆向 Heun、基础网格。没有 `continue_edit` 和 `edit_cfg`。
- `data.py`：`LinearManifest`、`StatePairManifest`、`TrainingSampler`。L 在线构造，I/R 读缓存，保持源图家族 split，验证条件与协议。
- `loss.py`：`state_flow_matching` 按有效区计算 `((u-(b-a))/t)^2`。LPIPS 只用于缓存质量和评测。
- `train.py`：每次只抽来源、时间和编辑进度，执行单步 IFM；阶段预算由独立记录数决定。
- `prepare_state_pairs.py`：生产 I/R 缓存、源/目标重建记录、预览和来源元数据；不做多种子选优。
- `sampling.py`：读取固定时间状态，默认在该状态执行状态编辑，尾段调用 `NativeT2I`。
- `checkpoint.py`：严格同阶段 resume，以及只加载权重的 `load_stage_weights`。
- `conditions.py`：冻结的文本模板。
- `native.py`：显式冻结原生 T2I/IT2I 基线，不经过状态编辑 LoRA。
- `cli.py`：`train`、`sample`、`prepare-state-pairs`、`native`、`judge`。

## 3. 条件模板

```text
Edit request: {e}
Generation target after editing: {c_tail}
Preserve: {confirmed_preserve_attributes}
```

`Preserve` 为空时省略整行。`c_tail` 是完整目标场景描述。自纠偏时 `c_parent=c_tail=c_raw`；显式编辑时 `c_tail` 由父条件和用户编辑更新，UND joint prefix 与尾段 T2I 使用同一字符串。

目标图像不进入学生条件。L 数据若没有保存 `tail_condition`，使用确定性的 `derived_parent_plus_edit` 迁移文本，只拼接父场景和短指令，不读取目标像素。

## 4. 数据合同

L 数据是现成编辑对。清单为 `{"records":[...]}`，每条至少包含：

```text
uid, split, prompt_family, source_family,
source_image, target_image,
edit_instruction, parent_condition,
optional tail_condition, preserve,
semantic_approved, optional condition_protocol
```

I/R 缓存由 `prepare-state-pairs` 生成。目录下是 `manifest.json` 或 `manifest.shard_*.json`；训练入口可以直接指向目录。每条记录包含：

```text
uid, source_kind, split, source_family, prompt_family,
source_image, target_image, edit_instruction,
parent_condition, tail_condition, preserve, condition_protocol,
valid_region, states, source_seed_or_linear_noise_seed,
source_reconstruction_metrics, target_reconstruction_metrics,
semantic_quality_source, target_quality_source
```

`states` 的每个时间键是六位小数，指向一个 FP32 `.pt`：

```text
time, state_a, state_b, preview, optional base_prediction, valid,
source_reconstruction, target_reconstruction, accepted
```

缓存头记录 `kind`、`protocol`、`native_model_id`、`noise_scale`、`cfg`、`cfg_norm`、`t_eps`、`forward_grid`、`forward_solver`、`inversion_grid`、`inversion_solver`、`preview_protocol` 和 `data_version`。训练前逐条验证协议；同一 `prompt_family` 或 `source_family` 不能跨 split。

I 数据的目标侧接受条件：有效区 L1（[-1,1] 坐标）不高于 `0.08`，LPIPS-VGG 不高于 `0.20`。源侧重建误差只记录。R 的真实父状态必须来自前向缓存，不能用源图反演替代。

## 5. 配置

- `configs/u15_isochronic_state/core.json`：共同协议、时间、保持比例、反演阈值。
- `stage1.json`：L 数据，LR `5e-5 → 1e-5`，warmup 100，上限 6000 步。
- `stage2.json`：L:I:R = 20:40:40（R 缺失时程序显式改为 20:80），LR `1e-5 → 2e-6`，上限 4000 步。
- `sample.json`：只继承核心协议。

阶段预算：

```text
S1 = min(6000, ceil(8 * N_L / 64))
S2 = min(4000, ceil(8 * N_I∪R / 64))
```

`N_L` 是独立训练编辑对数；保持副本不另算；I/R 一条记录保存多个时间点，只算一个独立记录。

## 6. 检查点

检查点包含 `adapters.*`、`edit_s_embedder.*`、优化器、采样器 RNG 和清单。清单记录 `stage`、`step`、`world`、`config`、`identity`、`parameter_layout`、`data_version`、`input_domains`。

- `--resume`：同阶段、同 world、同 config、同 identity，恢复优化器和采样器。
- `--init-from`：只加载 Stage 1 权重到 Stage 2，重置优化器动量。
- `input_domains` 只记录实际抽到并训练过的来源；R 未实际训练时不会写成 `native_history`。
