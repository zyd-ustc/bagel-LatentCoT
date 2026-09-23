# Phase 1 pair-grounded gradient calibration

## Observed NPU smoke

The two-record smoke at `/data/outputs/bagel_phase1_pair_smoke_4qTcGq` completed
both training stages. Phase 1.1 recorded:

| Group | Loss | Pre-clip gradient norm | Student delta RMS | Target delta RMS |
|---|---:|---:|---:|---:|
| no-op | 30324.96875 | 19.0480709 | 174.14066 | 0 |
| edit | 0.0449910 | 0.00725017 | 16.04067 | 15.91936 |

The no-op relative error has a zero target denominator and is undefined. The
trainer now writes `null` for no-op cosine and relative error, retaining
`student_rms` as the no-op diagnostic.

## Calibration runs

The no-op objective remains `RMS(D_S)^2`. Its multiplier changes from `1.0`
to `4e-4` for a first 64-step NPU pilot: the observed two-record no-op
gradient norm would become about `0.00762`, near the edit norm `0.00725`,
before accounting for model updates. The pilot completed at
`/data/outputs/bagel_phase1_calibration_pilot_EChVDt/phase11`: 16 no-ops and
48 edits, all finite and none clipped. Median pre-clip gradient norms were
`0.000676` and `0.007919`, respectively; their ratio was only `0.0854`.
Median no-op RMS was `11.71`, with a large tail reaching `115.57`.

The second 64-step pilot uses `lambda_noop_mem=0.002` (5x) at
`/data/outputs/bagel_phase1_calibration_v2_pilot_P3G7vU/phase11`. It also
completed with 16 no-ops and 48 edits, all finite and none clipped. Median
gradient norms were `0.003370` and `0.007920`, ratio `0.425`; the largest
no-op gradient was `0.586`. The step-64 UND-Q checkpoint exists. Compared
with the first run, the edit cosine/relative-error medians stayed nearly
unchanged (`0.7462` / `0.9505`). These are per-sample training diagnostics,
not held-out gains or evidence of learning.

The NPU checkout used for the smoke iterated the manifest in file order. On the
full train manifest, the Stage-A filter retained 11,487 rows, including 3,983
no-ops (34.7%); 707 of its first 1,000 rows were no-ops. The local trainer's
deterministic sampling order targets 20% no-ops and must be included in the
pilot and deployed before a full run.

## Pilot gate

The Phase 1.1 numerical pilot passed: both groups were visited, all losses
and gradient norms were finite, none clipped at norm 1, and the no-op/edit
median gradient-norm ratio was within 0.1–10. This gate permits a longer
Phase 1.1 run; edit quality, no-op RMS reduction, instruction retrieval, and
held-out flow quality require later evaluation.

The 32-step Phase 1.2A connectivity pilot using the step-64 Read adapter also
completed at `/data/outputs/bagel_phase1_calibration_v2_pilot_P3G7vU/phase12a`.
It visited 8 no-ops and 24 edits; all flow losses and gradient norms were
finite and nonzero, and the combined step-32 checkpoint exists. Median flow
losses were `0.00321` for no-ops and `0.39482` for edits. These do not measure
held-out flow quality, and the short Read adapter is not a production winner.
