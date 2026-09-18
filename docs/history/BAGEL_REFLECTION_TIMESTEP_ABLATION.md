# BAGEL Reflection Timestep-Only Ablation

## Active forced-count positive control

The active run is now a causal edit-capacity test rather than an accuracy
improvement test.  The original prompt asks for two green pyramids; every
treatment receives the mandatory counterfactual instruction:

```text
Remove both green pyramids. The final image must contain exactly zero green
pyramids. Preserve the red cube, blue sphere, yellow cylinder, wooden table,
and all of their existing attributes and relations.
```

The instruction is inserted into the critic system prompt and query.  After UND
decoding, it is also appended through native text cache update to the exact
full cache.  Therefore a generated `No correction needed` cannot remove the
intervention from the generation condition.

This positive control must not be scored against the original GenEval target:
success deliberately violates the original two-pyramid requirement.  Its gate
is visible count execution versus the paired `prompt_d1` image.

## Question

At which BAGEL shifted model-time can a read-only preview, mandatory count
instruction, native UND reflection, full-cache reuse, and depth-2 boundary loop
still change final object count within one continuous denoising trajectory?

This experiment does not optimize architecture or compare feedback transports.
Its only independent variable is `reflection_t`.

## Paired contract

For each prompt and seed, generate the native `prompt_d1` trajectory once and
retain the exact pre-Euler state and native depth-1 velocity at every requested
trigger.  Every treatment starts from one of those retained states.

```text
prompt_d1 baseline (one per seed)
       |
       +-- x_t@0.90 -> preview -> UND reflection -> full cache -> depth2 suffix
       +-- x_t@0.85 -> preview -> UND reflection -> full cache -> depth2 suffix
       +-- x_t@0.80 -> preview -> UND reflection -> full cache -> depth2 suffix
       +-- x_t@0.75 -> preview -> UND reflection -> full cache -> depth2 suffix
       +-- x_t@0.70 -> preview -> UND reflection -> full cache -> depth2 suffix
       +-- x_t@0.60 -> preview -> UND reflection -> full cache -> depth2 suffix
       +-- x_t@0.50 -> preview -> UND reflection -> full cache -> depth2 suffix
```

Fixed across candidates: checkpoint, prompt, seed/noise, Euler schedule, CFG,
reflection prompt, full-cache transport, loop depth/layers/state mode/residual
scale, and `post_loop_stop_t=0.35`.

## Metrics

`relative_l2` is only a mechanism diagnostic.  The primary metric is paired
GenEval2 Soft-TIFA log-GM versus the same seed's `prompt_d1` image.

1. Reflection gate: whether the critic emits a concrete correction rather than
   `No correction needed`.
2. Trigger mechanism: full treatment velocity versus prompt depth 1 at the same
   `(x_t,t)`, plus loop-only velocity delta under the same full cache.  Use
   `relative_l2 >= 0.20` as the minimum movement gate because the previous
   `0.1143` changed appearance without changing semantic layout.
3. End-state movement: final latent relative L2 and RGB pixel MAE versus the
   paired baseline.
4. Semantic outcome: paired GenEval2 log-GM delta and per-atom score changes.

## Selection and stop rules

Choose the earliest/highest `t` that has a readable preview, a concrete
preview-specific correction, and positive paired semantic delta.  Do not choose
a timestep from flow L2 or pixel difference alone.

- High flow delta with unchanged atom scores: timing is not the missing
  mechanism; frozen BAGEL is moving in a non-semantic direction.
- All reflections say no correction: the critic/preview contract is the
  blocker; the timestep sweep is not interpretable.
- Low flow delta at every timestep: full-cache conditioning is too weak before
  considering RL.
- Positive mean delta driven by only one seed: repeat on more failed prompts
  before selecting the training window.

The one-seed pilot is an execution/interpretability gate.  Run four seeds only
if at least one timestep produces a real correction and a changed GenEval2 atom
score.
