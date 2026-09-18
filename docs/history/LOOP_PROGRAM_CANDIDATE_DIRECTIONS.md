# Loop Program: Candidate Directions

Recorded 2026-09-15, after the single-trajectory zero-shot edit dead end
(`SINGLE_TRAJECTORY_EDIT_FEASIBILITY.md`). Decision owner: user.

## Standing constraints

- Zero-shot first: the structure must show a visible zero-shot causal effect
  (a lever RL can amplify), before investing in training.
- The eventual goal remains single-trajectory semantic accuracy / editing;
  RL is the designated performance mechanism, not SFT.
- Candidate 1 is deferred by decision (2026-09-15): no training-objective
  work for now.

## Candidates

### C1. SenseTime triad on the existing same-timestep loop — DEFERRED

Deep Supervision (flow loss at every loop pass), Loop Distillation (final
pass as stop-gradient teacher), Self-Modulating Attention (orthogonal
projection + input-dependent modulation to stop local-information erosion).
Architecture unchanged ([10,18) same-timestep loop). Directly treats the
training-side mirror of the observed zero-shot blur. Reference: SenseTime
Looped MMDiT blog (2026-09-08), code expected open-sourced end of September.

### C2. Cross-timestep recurrent state (Looped Flows port) — PASSED GO/NO-GO (2026-09-15)

The loop now runs across denoising steps: state z_t from the GEN body exit
at step t, merged (RMS-capped residual, sg(z)) into the body entry of step
t+1 over VAE+boundary positions (1026). The same-timestep executor was
destructively removed; zero-shot inference is Looped-Flows/RLT-aligned.

Probe verdict (`cross_step_state_probe_v1`, 2026-09-15, frozen checkpoint,
same noise/prompt/schedule across arms):

- parity exact (double-run assert passed); bootstrap step delta exactly 0.
- state_s0.2: max velocity delta 0.264 / mean 0.168 per step, final latent
  rel-L2 0.415 vs parity — 13x the old same-timestep floor (~0.02) while
  the image stays semantically clear and prompt-accurate (user gallery
  inspection).
- state_s0.5 / state_s1.0: velocity deltas 0.76/0.98 but images fully
  collapse — the coherent-lever ceiling lies between 0.2 and 0.5.
- Windowed arm (state only for t >= 0.75) matches the full-window result:
  the lever is concentrated in the high-noise layout phase.

**Deployment constraints:** loop_state_scale ~ 0.2, window t >= 0.75, body
layers [10,18). The zero-shot lever exists, is continuous in scale, and does
not break coherence — the precondition for RL to shape it into semantic
editing is satisfied.

**Next phase (unlocked):** warm-up with `BagelCrossStepFlowModule`
(Looped-Flows local flow losses along a shared-noise rollout; keeps state
consumption on-manifold), then GRPO with the semantic reward; the UND
reflection write (`loop_external_state`, the multi-round editing event)
joins in the RL phase with loop LoRA learning to read it.

### C3. Full RLT structure — LONG TERM

Learnable gated merge (W_g, W_s) replacing the fixed-gate residual merge,
plus layerwise KV carry across loop passes (RLT's C^D). Highest ceiling,
touches attention wiring; depends on C1/C2 training infrastructure.

### C4. Native-condition fallbacks — RECORDED BACKUP

1. VAE source-image edit format: preview latent as source-image VAE
   condition + edit text (BAGEL's trained editing pathway; violates the
   current hard boundary; by construction has a large zero-shot causal
   effect).
2. Two-stage regenerate: preview -> reflect -> rewrite prompt -> regenerate
   from the same noise (validated by the `target_from_start` arm).

## C2 go/no-go probe — RESOLVED (2026-09-15, PASSED)

Executed as `scripts/evaluate/bagel_cross_step_state_probe.py` with output
artifacts at
`vr.turbo-ai.com:/private/yida_workspace/outputs/cross_step_state_probe_v1/`
(manifest + gallery + per-arm PNGs). Pass rule was met at scale 0.2: the
per-step velocity effect cleared the 0.02 floor by an order of magnitude
with finals visibly moved and semantically coherent. See the C2 section
above for the recorded numbers and deployment constraints.
