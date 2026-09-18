# MoT 编辑反思循环 — 早期生成 loop

零样本、不训练。BAGEL 已有的成品图 I2I Editing **不是**本实验。
脚本：`scripts/evaluate/draft_prefix_loop.py`。

## 1. 问题

frozen BAGEL 的语义（数量 / 位置）在去噪**最前面几步**就定了。中途换文本改不了布局；把成品图丢进官方 Editing 只能证明模型本来就会改图。

要测的是：UND 读 **draft**（不是成品），写出反思，然后 **从同一 ε 重跑生成侧的前若干步**，语义会不会动。

## 2. 早期生成 loop（t_A）

主实验锁 `t_A = 0.8`（v2：0.9 的 UND 只会刷 MATCHES；0.7 的 draft 已接近成品，图像条件会把布局钉死）。

```
I_r  := decode(x0_hat) at t_A          # draft，不是完整图
prefix := Euler(ε, cond) 从 t=1 停在 t_A
```

每一轮 **重跑这段 prefix**，不是接着当前 `x_t` 往下走。

## 3. Markov 协议

```
s_r = (I_r, a_r)                       无历史
r = 0 :  I_0 ← prefix(ε, P) @ t_A      无图像条件
r ≥ 1 :  a_r ← UND(I_{r-1}, P)         只有 UND 看见目标 P
         I_r ← prefix(ε, I_{r-1}, a_r) @ t_A
```

GEN 的条件是 **`(VAE-encode(decode(I_{r-1})), a_r)`**，看不见 P。draft 对 GEN 可见。
r=0 用 T2I 的 ε；**r>0 必须重新采样噪声**（官方 Editing 合同）。把 t_A 的 `x0_hat` 当干净 VAE 条件、或复用 T2I 的 ε，都会把下一轮变成加噪重建。
没有可用 `INSTRUCTION:`（含 no changes needed）则跳过该轮。

UND 走官方 `understanding_output=True`（自然语言问题，不要 INSTRUCTION 表格）。
GEN 用官方 Editing 超参（`cfg_img_scale=2`，`cfg_interval=[0,1]`，`text_channel`）+ VAE/ViT(draft) + **新采样的 ε**，Euler **同样停在 t_A**，decode \(\hat{x}_0\)。UND 每一轮看到的都是 draft，不是成品图。
v5 满 50 步 Editing 只证明通道存在（成品图上能换类/改 count）；那会让 r≥1 变成完整去噪图，和「早期生成 loop」不一致。`--edit-stop full` 可复现 v5。

## 4. 边界

| zero-shot 必须成立 | 训练再管 |
|---|---|
| prefix 重跑后数量/位置相对 `I_0` 能变 | 反思更准、指令更可执行 |
| `a_r` 必须跟当前 draft 相关，允许不准 | 文本压成 latent / hidden 当状态 |
| `(I_r, a_r)` 显式可续跑 | — |

不做：门禁、Best-of-N、免解码、跨专家 SFT、官方成品 Editing。
与 Looped MMDiT 正交：那是单步内加深计算；这是步间用 UND 改早期条件。

## 5. 已测 / 下一步

**v5 通道探针（满 50 步 Editing）：** UND 读 t_A=0.8 draft 能写指令，官方 Editing 成品图上能换类/改 count。记录：`artifacts/experiment/draft_prefix_loop_v5/RESULTS.md`。

**当前协议（v7）：** r≥1 也早停在 t_A。先 3 条看 draft 链会不会动语义；能动再上 16 条 VQA。

暂缓：t 再扫、门禁 BoN、训练。
