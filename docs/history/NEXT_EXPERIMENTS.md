# 下一步实验规划

## 0. 已确立的事实（有实验支撑）

| # | 结论 | 证据 |
|---|---|---|
| F1 | frozen BAGEL 的 flow 起点锁定；中途换文本条件不改语义 | prompt_switch cos 0.978–0.9995；target_from_start cos 0.712 |
| F2 | prompt 固定时，注入 latent/hidden 对最终图几乎无影响 | K-sweep；within-step probe |
| F3 | `semantic_token` 载体 layout 脆弱，K≥2 即破坏数量/空间 | 金字塔 2→4 |
| F4 | 跨 timestep 的 GEN 自递归只动外观，不携带语义 | C2 probe（rel-L2 0.415，语义不变） |
| F5 | within-step naive loop 在 L=2 即崩（rel-L2 0.61 / MAE 34.9），阻尼改善但不能挽救；L=1 parity 成立 | `docs/WITHIN_STEP_LOOP_PROBE.md` |
| F6 | UND 理解路径正确（与官方 `understanding_output=True` 一致） | `reflection_sanity.py` |
| F7 | UND **不能**零样本读 GEN 投影的 VAE K/V（投影空间未对齐）；但两者共存不劣 | `und_reads_latent_probe.py` |
| F8 | 文本是唯一强条件通道（文本变→图变）；latent/hidden 弱 | F2 + instruction probe |

## 1. 必须先回答的两个"生死问题"

### A1. frozen GEN 是否服从"重写后的完整描述"？（决定纯文本 loop 是否存在）

- 问题：丢掉 P、只给重写描述 `T`，同 ε 下 GEN 是否按 `T` 改变语义？
- 做法：对 16 条 GenEval2-hard，用规则从 `vqa_list` 生成 `T`（把目标数量/属性写成完整描述）。
  - arms：`(i) GEN(ε|T)`；`(ii) GEN(ε|T, vae KV(x0_hat_0))`；`(iii) GEN(ε|P)` baseline。
  - 打分：GenEval2 `vqa_list` 过关率（用已有 scorer），**不是**像素差。
- 决策：`(i)` 相对 baseline 朝 `T` 显著移动 → 文本通道可控，继续 B。
  否则 → 放弃纯文本语义 loop，主线转 C。

### A2. 反思的时机窗口是否存在？

- 问题：draft 从哪个 t 起，UND 的反思"说对人话"（正确指出数量/位置）？
- 做法：复用 `multitraj_reflection_loop.py`，只跑 `draft + reflection`，
  `--truncations 0.95,0.9,0.85,0.8,0.75,0.7,0.6`，每 t 存 preview + reflection。
- 决策：窗口非空（存在 t 使反思正确且布局仍可改）→ loop 可行；
  为空 → loop 只能"固定 t 单次反思"，不能迭代。

### A3. edit CFG 是否真让图像条件起作用？（顺带，零成本）

- 问题：`cfg_img_scale=2.0` + `cfg_interval=[0,1]` + `cfg_renorm_type=text_channel`
  是否让 `vae KV` 条件真的影响输出？
- 做法：同 P 同 ε，只改这两个配置跑 regen，比 pixel MAE。
- 决策：若 `2.0` 与 `1.0` 无差别 → 图像条件对 frozen GEN 无效，"外观锚定"也无意义。

## 2. 轨道 B — 论文主张（A 通过后）

| 实验 | 内容 | 判据 |
|---|---|---|
| B1 单轮 loop | round1（带 P）→ 反思 → 丢 P → regen(`a_r` [+`vae KV`]) | VQA 朝目标移动 |
| B2 多轮 + 门禁 | R=3，gate = `a_r` 首 token（ACCEPT/REVISE） | VQA 随轮数提升 |
| B3 **compute-matched Best-of-N** | 同 NFE 下 N 条独立采样，用同一 UND 做 verifier 选优 | loop 必须**打败**它，否则只是多采样 |
| B4 消融 | 去 `a_r`（只 vae）／去 vae（只 `a_r`）／delta 措辞 vs 完整描述／有无门禁 | 定位收益来源 |

## 3. 轨道 C — 结构增量（与 A 独立，可并行）

| 实验 | 内容 | 判据 |
|---|---|---|
| C1 **SMA-lite（无参数）** | attention 输出去掉与自身 value 平行的分量，再跑 within-step probe 的 L=2/4/8 | `rel-L2 vs L=1` 是否下降 |
| C2 Deep Supervision + Loop Distillation（训练） | 把 `BagelCrossStepFlowModule` 从跨 timestep 改成**单步内重复 body block**；每轮 body-exit → suffix → `llm2vae` → flow loss；最终轮作 stop-grad teacher | 训练后 L>1 不再退化，且优于 L=1 |
| C3 学习版 SMA | 用 self-attention 信号生成输入依赖的调制系数 | 优于 C1 |

数据已就位（`~/work/datasets/cort_sft_133k/...`，245/245 分片），C2 可直接开跑。

## 4. 决策树

```
A1 fail  ->  放弃纯文本语义 loop；主线 = C（Looped MoT 结构增量）
A1 pass  ->  A2
  A2 fail ->  loop 降级为"固定 t 单次反思 + 重采样"，不做迭代
  A2 pass ->  B 为主线；C 作为正交增量（深度轴）叠加
```

## 5. 与 Looped MMDiT 的定位（写作时用）

- Looped MMDiT：**单步内**重复共享 block → 扩展**计算深度**（latent-space reasoning）。
- 本方案：**步间**由 UND 反思改条件并重采样 → 扩展**语义搜索**。
- BAGEL 的 MoT 比 MMDiT 多一个 UND 专家，所以能产出可读反思并转成文本条件；
  MMDiT 无此能力。两条轴正交，可叠。
