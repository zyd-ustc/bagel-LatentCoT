# Phase 1 Structured Reflection Checklist

## Identity

- run id: `phase1_structured_reflection_fresh_v1`
- idea id: structured text reflection → latent Δv correction
- stage: implementation

## Planning

- [x] selected idea summarized in 1–2 sentences
- [x] baseline and comparability contract confirmed
- [x] code touchpoints listed
- [x] smoke plan written
- [x] full run plan written
- [x] fallback options written

## Implementation

- [x] per-call K=0/8 contract implemented
- [x] selected Euler state capture implemented
- [x] Δv replay/loss module implemented
- [x] Phase 1.1 trainer and configs implemented
- [x] unrelated changes avoided or justified
- [x] risky logic guarded or sanity-checked

## Reliability Fixes

- [x] direction activation uses teacher correction RMS
- [x] replay states backward sequentially to release each BAGEL graph
- [x] training data requires a valid teacher score contract
- [x] noop/non-noop semantic constraints are validated
- [x] Base/Teacher/Student CFG shape and K contracts are tested
- [x] Q-only trainable/gradient allowlist is tested
- [x] strict Read/Write adapter gates are tested
- [x] Phase-2 GRPO explicitly uses Base K=0 and Loop K=8
- [x] Phase-2 GRPO accepts the Phase-1 v8 adapter contract

## Pilot / Smoke

- [x] unit tests pass (140 passed locally)
- [ ] one-update model smoke executed
- [ ] outputs and gradients look valid
- [ ] comparability still holds

## Main Run

- [ ] 100–500 sample overfit launched
- [ ] monitoring cadence started
- [ ] health signals confirmed
- [ ] runtime deviations reflected in plan

## Validation

- [ ] outputs exist
- [ ] metrics are complete
- [ ] baseline delta is comparable
- [ ] claim classified as supported / refuted / inconclusive
- [ ] result recorded durably

## Closeout

- [ ] experiment summarized in 1–2 sentences
- [ ] next action explicit
