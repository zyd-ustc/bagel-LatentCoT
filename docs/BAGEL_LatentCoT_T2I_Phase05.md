# Phase 0.5: Anchored Dynamic Prompt Memory

## Contract

Phase 0.5 no longer adds K=8/16 memory tokens. Prompt prefill records:

- native per-layer prompt `K₀/V₀`;
- every prompt token's layer-entry hidden state `H_P^{l,0}`;
- original prompt positions and packed sample ownership.

For each enabled denoising step, fixed `(P, x_t, t)` receives two body passes over
`[s,e)`:

1. **Read**: active prompt-copy rows restart from `H_P^{s,0}` and may attend to
   native prompt cache plus current GEN rows. All non-prompt-copy queries are
   masked from prompt-copy keys.
2. **Write**: GEN restarts from the native query. Native prompt K is unchanged;
   only prompt V rows become `V₀ + α(V_dyn - V₀)` at the matching layer.

The anchor cache is never mutated. `alpha=0` bypasses Read and calls native
`_forward_flow`, so V0 compares the exact original path rather than a nominally
equivalent reconstruction.

Default prototype:

```text
body                 [12,20)
enabled trajectory   early 35%
alpha sweep          0, +0.1, -0.1, +0.2
prompt length        native full length
K intervention       none
cross-step state     none (fresh Read at each enabled t)
```

## Validation order

1. V0: `alpha=0` native parity.
2. V1/V2: fixed-`x_t` sensitivity, approximate magnitude scaling, and
   `cos(Δv(+a), -Δv(-a))`.
3. V3/V4: zero/shuffled residual controls and Read usefulness.
4. V5: early/mid/late body and early/late/all step localization.
5. V6: paired-seed hard prompts, then full/easy prior regression.

Only V0–V2 and end-to-end alpha arms are automated in the first runner. Shuffled
delta is implemented in the model API and intentionally requires a packed batch
with at least two equal-length prompts.

## Commands

Fixed-state numerical probe:

```bash
python scripts/evaluate/bagel_dynamic_prompt_phase05.py \
  --mode probe \
  --model-path /data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT \
  --output-dir /data/zyd_workspace/outputs/phase05_dynamic_prompt_probe \
  --prompt-file experiments/data/geneval2_hard_16.txt \
  --alphas 0,0.1,-0.1,0.2 \
  --probe-timestep 0.8 \
  --body-start 12 --body-end 20 \
  --max-prompts 16
```

End-to-end hard16:

```bash
python scripts/evaluate/bagel_dynamic_prompt_phase05.py \
  --mode generate \
  --model-path /data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT \
  --output-dir /data/zyd_workspace/outputs/phase05_dynamic_prompt_hard16 \
  --prompt-file experiments/data/geneval2_hard_16.txt \
  --alphas 0,0.1,-0.1,0.2 \
  --body-start 12 --body-end 20 \
  --step-fraction 0.35 \
  --max-prompts 16
```

Numerical tests (not run during the refactor):

```bash
pytest -q tests/test_dynamic_prompt_numeric.py
```

Outputs are written to `manifest.json`. Probe mode contains relative velocity
change, direction cosine, scale ratio, and layerwise residual norms. Generate
mode stores one image per alpha plus per-step diagnostics.

## Explicit exclusions

No prompt compression, learnable encoder, LoRA, K rewrite, prompt-KV removal,
persistent cross-timestep memory, full-layer rewrite, or Phase 1 supervision is
part of this implementation.
