# BAGEL 原生边界状态理解循环（V5）

## 结论

V5 保持一次文生图轨迹和一次最终解码。prompt 是唯一推理输入；循环只携带
BAGEL 原生 `<vision_start>`、`<vision_end>` 的 UND hidden states。VAE hidden
每轮回到 body 入口，同一 `z_t`、timestep、position ids 和 prompt KV 不变。

fix/review 文本只在训练期经过冻结 BAGEL UND，形成 stop-gradient 目标表征。
它不进入生成 query，不经过 LM head，不生成 token，不新增 latent、projector
或 expert。

## 为什么从 V4 改为 V5

16-sample frozen probe 得到：

| 模式 | 范围 | loop/base | 样本胜率 |
|---|---:|---:|---:|
| boundary | 10:18 | 0.9999469 | 43.75% |
| prompt query | 4:12 | 0.9997867 | 75.00% |
| legacy full | 4:12 | 1.0117950 | 25.00% |

prompt query 的 depth-1 flow 已从 native `0.286637` 改成 `0.284716`，说明额外
query 本身改变了 BAGEL baseline；其 loop gain 因而不可作为原生循环证据。
V5 删除该模式。legacy full 明显破坏生成表征，也只保留为历史结果。

## 训练图

```text
generation: prompt KV + B0 + Gt -> depth1 -> carry B1/reset G -> depth2
                                      |                         |
                                   v1, B1                    v2, B2

target only: fix text -> frozen causal BAGEL UND -> h_fix (stop-gradient)

L_sem(depth) = 1 - cosine(mean(B_depth), h_fix)
```

默认 body 为 probe 选出的 `[10,18)`。仅第二遍 attention LoRA 可训练：GEN
`q/k/v/o_proj_moe_gen` 与文本 `k/v_proj`，rank 8。其余参数全部冻结。

## 验收

1. depth 1 与无 loop 的原生 BAGEL 完全一致。
2. target IDs 只出现在冻结 UND target forward。
3. semantic cosine loss 的梯度到达 loop LoRA，但 target encoder 无梯度。
4. fixed-timestep overfit 同时改善 flow 和 semantic depth-2/depth-1 差值。
5. shuffled target 对照显著差于正确配对后，再进入 GenEval2 和 GRPO。

