# Anchored Dual-State Loop

This is the active experiment line. The old Observation → Memory writer → generation SFT route stays in the tree as a diagnostic record and does not start training unless `BAGEL_ALLOW_LEGACY_MEMORY_SFT=1`.

At one flow state `(x_t, t)` the frozen BAGEL pre-blocks run once and produce the body entry `P_t`. The shared body then runs. A side-branch UND read sees image-only GEN KV from that body. It does not add tokens to GEN attention. A small updater and adapter turn the previous GEN hidden and that feedback into a residual on image tokens. The next body call is `P_t` plus a gated, RMS-bounded residual. Text tokens, prompt KV, and the latent `x_t` stay on the native path. The sampler steps only after the rounds finish.

`entry_scale = 0` returns the native velocity without a second body call. `force_full_zero_path` still runs the observer and the second body with a zero residual, which is the equivalence check. Training uses the learned gate tensor on scheduled steps so a Python zero does not block its gradient. The gate starts at zero; adapter and updater gradients open after the gate moves.

BAGEL flow time decreases from noise (`t = 1`) to data (`t = 0`). The schedule uses progress `u = 1 - t`, or an explicit sampler step index. The default window is the early 35% of progress. `[20, 28)` is only the candidate body span for a 28-layer model.

`variant=dual` is the main method. `variant=gen_only` keeps the same modules and replaces the UND read with zeros. `state_mode=direct_hidden` feeds the body output straight back into the body and is not the default. Feedback controls are `correct`, `zero`, `mean`, `shuffled`, and `frozen`, applied inside one branch, one time, and one round.

Check the contract on a tiny MoT decoder:

```bash
python -m qwen_latent_cot.anchored.cli self-check
```

The default config is `configs/anchored_dual_state_loop.json`. Checkpoints from this line store only the gate, adapter, and feedback updater.
