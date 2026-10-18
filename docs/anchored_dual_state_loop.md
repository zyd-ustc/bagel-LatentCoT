# Anchored Dual-State Loop

This is the active experiment line. The old Observation → Memory writer → generation SFT route stays in the tree as a diagnostic record and does not start training unless `BAGEL_ALLOW_LEGACY_MEMORY_SFT=1`.

At one flow state `(x_t, t)` the frozen BAGEL pre-blocks run once and produce the body entry `P_t`. The shared body then runs. The side-branch reader is the observation writer: neutral-text embeddings, every layer, and full-query GEN KV. Full-query includes image tokens and boundary tokens. The reader does not add tokens to GEN attention. Its Q/K/V/O and Q/K norms are an independent frozen copy of a format-4 observation checkpoint. Layernorms, MLP, and embeddings stay the native modules, which that run did not train. The generation decoder is never overwritten by that checkpoint. Reader output enters only the feedback updater. A small updater and adapter turn the previous GEN hidden and that feedback into a residual on image tokens. The next body call is `P_t` plus a gated, RMS-bounded residual. Text tokens, prompt KV, and the latent `x_t` stay on the native path. The sampler steps only after the rounds finish.

`observation_source_mode` is `gen_full_query` for this reader. The checkpoint is accepted only when its saved training config resolves to that single source. C2-observe forces the observe task. On any stage, an observe row calls `write(source, t)` and returns before prompt concatenation. Any other active task, including the C2-feedback endpoint task and every C3 non-observe task, trained with prompt KV and is `prompt_and_gen`. A mixed file, a missing config, a stage that disagrees with `config.stage`, `visual_source_policy` other than `full_query`, or `source_layers` other than every layer is refused before the copy. The parent path is stored with the checkpoint path and sha256 and does not choose the mode. `prompt_and_gen` can be written in a config. Load and `forward_branch` both refuse it, because this reader does not concatenate prompt KV.

The first read captures Pre, Body, and Post of the original GEN forward, so layers `0 .. depth-1` are present. A later read is Anchored Observation:

```text
O_{r+1} = UND(KV_Pre^{native}, KV_Body^{(r)}, KV_Post^{(r)})
```

Pre is the native computation and is not run again. Body and Post are recomputed from this round's anchored entry. With body `[20, 28)` on a 28-layer model, Post is empty, so the read is native Pre KV plus the recomputed Body KV. A full GEN re-forward would recompute Pre as well. The default `correction_rounds = 1` uses only the first, full-native read. The anchored stitch is the second read, used when `correction_rounds = 2`.

`observation_checkpoint` is required for `variant=dual`. Without it the loop refuses to read. `allow_unloaded_observer` exists only so the structural contract can run on a tiny model whose copy still holds native projections. That copy is not the verified reader. Matching the reference decoder shows that the copied projections compute the same slots. The trace shows that the anchored stitch reuses the native Pre KV objects. Neither check shows that the feedback improves the generated image. That question is the GEN-only versus dual-correct comparison, and the correct versus shuffled comparison.

The adapter still mean-pools the memory slots onto the image tokens. Replacing that pool with image-to-slot cross-attention is a separate mechanism.

`entry_scale = 0` returns the native velocity without a second body call. `force_full_zero_path` still runs the observer and the second body with a zero residual, which is the equivalence check. Training uses the learned gate tensor on scheduled steps so a Python zero does not block its gradient. The gate starts at zero; adapter and updater gradients open after the gate moves.

BAGEL flow time decreases from noise (`t = 1`) to data (`t = 0`). The schedule uses progress `u = 1 - t`, or an explicit sampler step index. The default window is the early 35% of progress. `[20, 28)` is only the candidate body span for a 28-layer model.

`variant=dual` is the main method. `variant=gen_only` keeps the same modules and replaces the UND read with zeros. `state_mode=direct_hidden` feeds the body output straight back into the body and is not the default. Feedback controls are `correct`, `zero`, `mean`, `shuffled`, and `frozen`, applied inside one branch, one time, and one round.

Check the contract on a tiny MoT decoder:

```bash
python -m qwen_latent_cot.anchored.cli self-check
```

The default config is `configs/anchored_dual_state_loop.json`. Checkpoints from this line store the gate, adapter, and feedback updater, plus the observation path, sha256, stage, resolved source mode, trained sources, and parent. They do not store a second copy of the UND readout.
