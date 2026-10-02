# Reader warm-up offline diagnosis

CPU-only analysis of original logs. No model inference, no gate/objective change.

| Step | Correct MSE | Shuffled MSE | Zero MSE | Correct < zero |
|---|---:|---:|---:|---|
| 0 | 7.139190 | 7.515929 | 2.704639 | False |
| 250 | 5.464242 | 5.911821 | 2.704639 | False |
| 500 | 4.551529 | 5.040373 | 2.704639 | False |
| 750 | 3.728385 | 4.258073 | 2.704639 | False |

## Latest paired comparisons

Negative delta favors correct. Average within prompt, then bootstrap prompts.

| Group | Control | Prompts | States | ΔMSE | 95% CI | Prompt win fraction |
|---|---|---:|---:|---:|---|---:|
| all | initial | 8 | 16 | -3.4108 | [-4.0326, -2.7765] | 1.000 |
| all | shuffled | 8 | 16 | -0.5297 | [-1.5266, 0.6089] | 0.750 |
| all | zero | 8 | 16 | 1.0237 | [0.3101, 1.8875] | 0.125 |
| category:count | initial | 2 | 4 | -3.3354 | [-4.0008, -2.6701] | 1.000 |
| category:count | shuffled | 2 | 4 | -1.3475 | [-2.5360, -0.1591] | 1.000 |
| category:count | zero | 2 | 4 | -0.1658 | [-0.4010, 0.0694] | 0.500 |
| category:spatial_relation | initial | 6 | 12 | -3.4359 | [-4.2247, -2.6452] | 1.000 |
| category:spatial_relation | shuffled | 6 | 12 | -0.2571 | [-1.4317, 1.0888] | 0.667 |
| category:spatial_relation | zero | 6 | 12 | 1.4203 | [0.6925, 2.3757] | 0.000 |
| step_bucket:early | initial | 7 | 11 | -3.2811 | [-4.0199, -2.5748] | 1.000 |
| step_bucket:early | shuffled | 7 | 11 | -0.2514 | [-1.2759, 0.9352] | 0.714 |
| step_bucket:early | zero | 7 | 11 | 1.2220 | [0.5167, 2.1382] | 0.000 |
| step_bucket:late | initial | 1 | 2 | -4.0008 | not estimable | 1.000 |
| step_bucket:late | shuffled | 1 | 2 | -2.5360 | not estimable | 1.000 |
| step_bucket:late | zero | 1 | 2 | -0.4010 | not estimable | 1.000 |
| step_bucket:middle | initial | 3 | 3 | -2.7088 | [-3.5601, -1.8796] | 1.000 |
| step_bucket:middle | shuffled | 3 | 3 | -1.0420 | [-1.5454, -0.3320] | 1.000 |
| step_bucket:middle | zero | 3 | 3 | 0.6554 | [0.3863, 0.8997] | 0.000 |

## Layer readout diagnostics

These are correct-arm aggregate diagnostics, not paired layer-control results.

| Layer | MSE | Cosine | Readout RMS | Target RMS | Adapter RMS | Effective slots |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 1.6894 | 0.4961 | 1.4859 | 0.8110 | 1.0946 | 7.0228 |
| 13 | 1.9694 | 0.5867 | 1.7343 | 1.1217 | 0.5986 | 6.6759 |
| 14 | 2.1691 | 0.6483 | 1.9082 | 1.4375 | 1.0977 | 6.8752 |
| 15 | 1.9085 | 0.5858 | 1.6687 | 1.2099 | 2.1333 | 6.4985 |
| 16 | 2.6861 | 0.6576 | 2.1706 | 1.5361 | 1.2054 | 6.2906 |
| 17 | 2.9887 | 0.6491 | 2.2254 | 1.8088 | 1.4992 | 5.3395 |
| 18 | 3.2652 | 0.7187 | 2.4316 | 2.3388 | 1.9574 | 6.8655 |
| 19 | 13.1506 | 0.2306 | 3.3729 | 2.2454 | 2.0957 | 6.5727 |

## Interpretation

Verdict: `reconstruction_improves_but_zero_control_is_better`. Original operational gate: `True`.
Zero control is diagnostic only and was not a training loss or gate criterion.
No semantic memory-usefulness or image-quality claim follows from these metrics.
Training log: 755 updates; latest heldout checkpoint: 750.

## Limitations

- One training seed; bootstrap covers these heldout prompts, not training-seed variation.
- Only 8 of 64 heldout prompts evaluated; no new forward passes.
- Layer diagnostics are existing aggregate means; no per-state layer CI is recoverable.
- Step buckets use rollout-index boundaries 40%/75%, not shifted timestep magnitudes.
- Reader reconstruction MSE and gate passage are not image semantic scores.
- No multiplicity correction: bucket/CI analysis is exploratory.
- Training-window loss averages involve different prompts; not a heldout convergence test.
