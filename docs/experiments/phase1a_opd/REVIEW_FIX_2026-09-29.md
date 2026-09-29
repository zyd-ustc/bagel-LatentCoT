# Phase 1A T0 review corrections — 2026-09-29

The 2026-09-28 validation note is a historical snapshot of commit `95d24aa`.
This follow-up fixes its two identified pre-training risks without claiming a
full BAGEL-weight, H200, or semantic evaluation run.

1. Frozen strict Read now records raw body-entry memory at every layer in
   `[12,20)`. The corresponding Student reader consumes only that layer's
   detached memory. It applies the layer's frozen UND input RMSNorm before
   native UND K/V. The residual branch remains position-free and starts with
   zero output; native attention and the writer remain frozen.
2. Formal training requires scored, disjoint held-out teacher-vs-native
   evidence with `teacher_score > native_score`. The baseline field RMS is
   diagnostic only. Field-only training is explicitly debug-labelled and
   capped at 10 steps.
3. Teacher `Counts:` must preserve every explicit numeric count from the
   prompt and may not introduce another numeric count.

Local CPU suite: 283 tests passed. The added regression compares Student
reader inputs against a separately executed frozen Read capture and checks
that UND K projection receives the layer-normalized memory. GPU numerical
stability, teacher quality, and image-level semantic gain remain unverified.
