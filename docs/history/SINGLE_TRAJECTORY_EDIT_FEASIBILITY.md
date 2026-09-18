# Single-Trajectory Mid-Denoise Edit Feasibility: Negative Conclusion

Status: **concluded 2026-09-14** — single-trajectory mid-denoise editing on
frozen native BAGEL is a dead end. Recorded after the zero-shot prompt-switch
probe closed the last open hypothesis.

## Verdict

On frozen BAGEL-7B-MoT, changing any text condition (or injecting any
reflection hidden state) after the first few Euler steps of one continuous
denoising trajectory cannot change the image's semantic layout. The ODE
trajectory locks into its attractor basin within the earliest steps; all
later conditioning — clean prompt replacement, edit-instruction append,
full-cache reflection transport, boundary/all_vae hidden bridges — produces
at most appearance-level perturbation (blur, pixel drift), never a semantic
edit.

## Evidence chain

All runs share the same frozen checkpoint
(`/private/yida_workspace/models/BAGEL-7B-MoT`), 512x512, 50-step shifted
Euler schedule (shift 3.0), CFG text 4.0, and (except the prompt-switch
probe) the same two-green-pyramid prompt family.

1. Forced-count positive control, full-cache + boundary hidden bridge
   (`bagel_forced_count_t_sweep_pilot_v2`, `bagel_und_hidden_bridge_zs_v1`):
   correction_rate 1.0 at every t in {0.5..0.9}; trigger-step flow
   relative-L2 up to 0.36 (t=0.9); final pixel MAE at most 11/255. The
   trigger velocity responds, the final semantics do not change. Gallery
   inspection: pyramids persist.

2. all_vae loop state mode (1024 VAE + 2 boundary injection sites,
   `zs_edit_allvae_s02_v1` / `zs_edit_allvae_s05_v1`, 2026-09-14): widening
   the hidden-bridge channel from 2 to 1026 positions produces visible blur
   (global appearance perturbation) and no semantic edit. Mechanism: the
   external state is a single broadcast UND text vector — rank-1,
   position-independent, off-manifold for GEN; frozen GEN has no reader for
   it; the bridge lives only in the CFG conditional branch and is amplified
   4x, which amplifies noise (blur), not semantics.

3. Prompt-switch probe, clean replacement condition, no reflection, no
   loop (`pswitch_t*_v1`, `zero_shot_mid_denoise_prompt_switch.py`):
   replace switches the entire text cache to a pyramid-free prompt P1 at the
   switch step; append adds the explicit removal instruction. Same initial
   noise per run; `target_from_start` (P1 from t=1.0) confirms P1 is
   generatable (latent cosine vs P0 baseline 0.712).

   | switch_t | replace latent cos | append latent cos |
   |---|---|---|
   | 0.95 | 0.9785 | 0.9678 |
   | 0.92 | 0.9888 | 0.9883 |
   | 0.90 | 0.9968 | 0.9889 |
   | 0.85 | 0.9984 | 0.9975 |
   | 0.75 | 0.9995 | 0.9992 |

   Even switching after only ~2-3 Euler steps of the original prompt (t=0.95)
   leaves the trajectory in the original attractor (0.9785 vs the 0.712
   target bound). Monotone lock-in: the earlier the switch, the more movement,
   but never beyond appearance level. The edit window is empty at every
   meaningful time.

## Hypotheses eliminated (in order)

- reflection quality — probe 3 has no reflection, still fails;
- condition contradiction / dilution / preview anchoring — probe 3 replace
  is a clean swap with no preview and no contradiction, still fails;
- CFG axis (text-removed branch = visual-only) — probe 3 uses the standard
  empty-prompt negative branch, still fails;
- loop injection width (2 boundary tokens vs 1026 all_vae sites) — evidence
  2 shows width changes blur magnitude, not semantics;
- remaining root cause: **frozen BAGEL's flow ODE is attractor-locked from
  the first steps; text guidance after that point cannot overcome the
  image's self-guidance**, and no zero-shot state carrier exists that frozen
  GEN can read as an edit program.

## Consequences for the program

- The loop (boundary or all_vae) cannot be a zero-shot edit carrier. Its
  value must be established by training (see Looped MMDiT / Looped Flows /
  RLT references below).
- Remaining zero-shot routes, both recorded as deliberately violating or
  reinterpreting the current hard contract:
  1. Native VAE source-image edit format (preview latent as source-image VAE
     condition + edit text; BAGEL's trained editing pathway; violates the
     "no source-image VAE latent in KV" boundary).
  2. Two-stage regenerate (preview -> reflect -> rewrite prompt -> regenerate
     from the same noise; `target_from_start` arm already validates the
     mechanism; not an edit of one trajectory).

## References

- `refs/loop/2609.11801v1.pdf` — Looped Flows: recurrent state across
  denoising timesteps, trained with temporally aligned local flow losses.
- `refs/recurrent-looped-tranformer/` — RLT: full-state gated recurrence,
  one transition for prompt and response.
- SenseTime Looped MMDiT blog (2026-09-08): deep supervision, loop
  distillation, self-modulating attention; latent self-correction between
  loops is a trained capability, not zero-shot.
