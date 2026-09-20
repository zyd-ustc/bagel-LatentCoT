# Phase 0.5 T2I 128-Prompt Checklist

## Identity

- run id: `phase05_t2i_128_body_persistence`
- stage: implementation / pre-run validation

## Planning

- [x] selected idea and baseline contract recorded
- [x] code touchpoints, smoke path, full path, and fallback recorded

## Implementation

- [x] matched fresh/persistent body-window arms implemented
- [x] balanced 128-prompt GenEval2 subset materialized
- [x] integrated scoring path implemented
- [x] unrelated working-tree changes excluded

## Pilot / Smoke

- [x] local unit and static checks pass (`115 passed`)
- [ ] one-prompt NPU smoke produces all requested arms
- [ ] smoke score outputs are interpretable

## Main Run

- [ ] 128-prompt generation launched on 16 NPUs
- [ ] all 896 images and manifests exist
- [ ] Soft-TIFA scoring completes for all seven arms

## Validation And Closeout

- [ ] AM/GM, skill, atomicity, and mechanism metrics are complete
- [ ] matched persistence deltas are comparable
- [ ] result is classified as supported, refuted, or inconclusive
- [ ] next action is explicit
