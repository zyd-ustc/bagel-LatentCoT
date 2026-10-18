# Anchored Dual-State Loop

This is the active experiment line. The old Observation → Memory writer → generation SFT route stays in the tree as a diagnostic record and does not start training unless `BAGEL_ALLOW_LEGACY_MEMORY_SFT=1`.

At one flow state `(x_t, t)` the frozen BAGEL pre-blocks run once and produce the body entry `P_t`. The shared body then runs. The side-branch reader follows the verified C2 observation recurrence: the query is the neutral seed embedding stored before layer 0, and every UND layer attends to that layer's full-query GEN KV. Full-query includes image tokens and boundary tokens. GEN body hidden is not pooled into the query, and the body span is the GEN reentry site, not the reader's layer subset. The reader does not add tokens to GEN attention. Its Q/K/V/O and Q/K norms are an independent frozen copy of a format-4 observation checkpoint. Layernorms, MLP, and embeddings stay the native modules, which that run did not train. The generation decoder is never overwritten by that checkpoint. Reader output enters only the feedback updater. A small updater and adapter turn the previous GEN hidden and that feedback into a residual on image tokens. The next body call is `P_t` plus `Bound(gamma * delta)`. Text tokens, prompt KV, and the latent `x_t` stay on the native path. The sampler steps only after the rounds finish.

`observation_source_mode` is `gen_full_query` for this reader. The checkpoint is accepted only when its saved training config resolves to that single source. C2-observe forces the observe task. On any stage, an observe row calls `write(source, t)` and returns before prompt concatenation. Any other active task, including the C2-feedback endpoint task and every C3 non-observe task, trained with prompt KV and is `prompt_and_gen`. A mixed file, a missing config, a stage that disagrees with `config.stage`, `visual_source_policy` other than `full_query`, or `source_layers` other than every layer is refused before the copy. The parent path is stored with the checkpoint path and sha256 and does not choose the mode. `prompt_and_gen` can be written in a config. Load and `forward_branch` both refuse it, because this reader does not concatenate prompt KV.

The first read captures Pre, Body, and Post of the original GEN forward, so layers `0 .. depth-1` are present. A later read is Anchored Observation:

```text
O_{r+1} = UND(KV_Pre^{native}, KV_Body^{(r)}, KV_Post^{(r)})
```

Pre is the native computation and is not run again. Body and Post are recomputed from this round's anchored entry. With body `[20, 28)` on a 28-layer model, Post is empty, so the read is native Pre KV plus the recomputed Body KV. The query and the layer set stay the verified recurrence. The KV combination does not: C2 observation captured one native GEN forward at every layer. This stitch is an architecture variant that still needs its own check. A full GEN re-forward would recompute Pre as well. The default `correction_rounds = 1` uses only the first, full-native read. The anchored stitch is the second read, used when `correction_rounds = 2`.

`observation_checkpoint` is required for `variant=dual`. Without it the loop refuses to read. `allow_unloaded_observer` exists only so the structural contract can run on a tiny model whose copy still holds native projections. That copy is not the verified reader. Matching the reference decoder shows that the copied projections compute the same slots. The trace shows that the anchored stitch reuses the native Pre KV objects. These checks do not show that the feedback improves the generated image. `gen_only` against the native pass is the repeated GEN computation. `dual` against `gen_only` adds the memory state and the UND read together. `correct` against `shuffled` keeps that state update and changes the observation content.

Each image token reads the feedback slots with one-head cross-attention of width `rank`:

```text
delta_i = W_O Attn(W_Q H_i, W_K M, W_V M) + U(H_i, P_t, t)
```

`M` stays `[slots, hidden]`. A token can use one slot without giving that mixture to every other image token. Samples do not share slots. The anchor and the flow time remain on the low-rank path. Text rows stay zero. The RMS cap is applied to the gated residual, `Bound(gamma * delta)`, and that capped tensor is what is added to the native body entry. The product and the cap run in float32 and are cast once into the anchor dtype. A zero gate adds an exact zero. Under the default adapter initialization the cast can still land on a bf16 zero, so a velocity change in the contract is produced by widening those weights and is not evidence that the default residual moves bf16 velocity.

Training and inference both read the learned gate. On a scheduled training step the gate stays in the graph, so a zero value still trains the gate and blocks adapter gradients. Inference reads that same parameter. An exact-zero gate returns the native velocity without a second body call. `force_native_bypass` returns that native velocity on a scheduled step as well, including while training. `force_full_zero_path` still runs the observer and the second body with a zero residual, which is the equivalence check, and does not read the gate. The two flags cannot both be set. The gate starts at zero; adapter and updater gradients open after the gate moves.

BAGEL flow time decreases from noise (`t = 1`) to data (`t = 0`). The schedule uses progress `u = 1 - t`, or an explicit sampler step index. The default window is the early 35% of progress. `[20, 28)` is only the candidate body span for a 28-layer model.

`variant=dual` is the main method. `variant=gen_only` is the no-memory control:

```text
H_{r+1} = B(P + Bound(gamma * ΔH(H_r, P_t, t)))
```

`ΔH` reads the current GEN hidden and the anchor. It does not read `M_r`, and this variant does not run the UND reader or the feedback updater. Feedback controls `correct`, `zero`, `mean`, `shuffled`, and `frozen` apply to the dual variant's memory state, inside one branch, one time, and one round. `state_mode=direct_hidden` feeds the body output straight back into the body and is not the default.

Every forward returns `round_velocities`, `round_deltas`, and `round_observations`. `round_velocities[0]` is the native velocity `v^{(0)}`. Entry `k` is the velocity after correction `k`, so two corrections produce `v^{(0)}, v^{(1)}, v^{(2)}`. The sampler still steps on the last velocity. Deltas and observations have one entry per correction. On `gen_only` each observation entry is `None`. When intermediates are logged, every correction record stores its relative drift from the native velocity.

`round_loss_weights` is optional. `None` keeps the flow loss on the final velocity. A tuple with one non-negative weight per correction replaces that term with `sum_r w_r MSE(v^{(r)}, target)`. The weights may differ. A zero weight stays out of the graph. The anchor prior stays on the final velocity, and the entry term stays on the last written residual. Early training can put more weight on the first correction, for example `(1.0, 0.25)`, and a later run can supervise the last round alone. `round_velocity_report` lists each round's error against one flow target, whether that error fell or rose relative to the previous round, and the drift from the native velocity. That report measures the round-to-round change. It does not show that feedback improves generation.

Check the contract on a tiny MoT decoder:

```bash
python -m qwen_latent_cot.anchored.cli self-check
```

The default config is `configs/anchored_dual_state_loop.json`. Checkpoints from this line store the gate, adapter, and feedback updater, plus the observation path, sha256, stage, resolved source mode, trained sources, and parent. They do not store a second copy of the UND readout.
