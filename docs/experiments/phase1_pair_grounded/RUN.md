# Phase 1 Pair-Grounded Smoke Run

Date: 2026-09-22
Host: Ascend NPU ModelArts
Code checkout: `/root/bagel-LatentCoT-phase1`

## Data validation

- Manifest: `smoke4k_seed42_v2/train_candidates.jsonl`
- Stage-A records accepted: 2,402
- Reflection and teacher-score fields were not used.

## Tests

```text
38 passed, 1 warning
```

The warning is the existing `torch_npu` shared-library ownership warning.

## Phase 1.1 one-step smoke

```text
sample_id=sharegpt4o_15497__noop
loss=30390.794922
cosine=-0.4640
relative_error=34.1076
```

Output:

```text
/data/outputs/bagel_loop_pair_memory_api_smoke_v1/
  pair_memory_adapter_step_0000001.safetensors
```

This is a connectivity/gradient smoke, not a quality result.  It verified the
Read-only STOP path, UND-Q-only gradients, optimizer step, and route-filtered
checkpoint save.

## Phase 1.2A one-step smoke

```text
sample_id=sharegpt4o_15497__noop
flow_loss=0.001597
```

Output:

```text
/data/outputs/bagel_loop_pair_flow_api_smoke_v1/
  pair_flow_adapter_step_0000001.safetensors
```

It verified loading/freeze of the Phase 1.1 UND-Q checkpoint, GEN-Q-only
gradients, conditional `cfg_text=cfg_img=1` Read-Write forward, target-flow MSE,
and combined checkpoint save.

## 2026-09-23 gradient-calibration pilots

See `CALIBRATION_2026-09-23.md`. Two 64-step Phase 1.1 NPU pilots calibrated
the no-op weight to `0.002`; the final no-op/edit median pre-clip gradient-norm
ratio was `0.425` with no clipping. A 32-step Phase 1.2A pilot loaded the
step-64 Read adapter and saved a combined checkpoint. All pilot losses and
gradient norms were finite. The pilots verify numerical training paths, not
held-out editing quality or the Phase 1.1 Go condition.

## 2026-09-23 calibrated Phase 1.1 full run

The 1000-step NPU run completed at
`/data/outputs/bagel_pair_memory_calibrated_f783989`. Its configuration,
grouped training metrics, final checkpoint hash, and pending quality gates are
recorded in `RESULT_1000_2026-09-23.md`.
