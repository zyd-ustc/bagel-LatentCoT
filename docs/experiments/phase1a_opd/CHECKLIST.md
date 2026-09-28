# Phase 1A T0 implementation checklist

- [x] Read plan and existing implementation; preserve local archive.
- [x] Implement prompt-hidden M0 and independent zero-effect memory reader.
- [x] Integrate frozen Read and optional GEN residual with exact O=0 parity.
- [x] Implement T0 teacher cache, OPD runner and evaluation controls.
- [x] Pass targeted and full CPU tests; document remaining GPU/semantic gates.

Next experiment gate: curate disjoint semantic train/held-out prompts and run
the offline BAGEL teacher cache + pre-training teacher baseline. No OPD training
has been launched by this implementation pass.

No GPU training, remote upload or research metric claim is in this task.
