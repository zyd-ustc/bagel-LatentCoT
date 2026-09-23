# Write Sensitivity T2I checklist

- [x] Contract and baseline defined in `PLAN.md`.
- [x] User constraint recorded: provide NPU commands, do not run them.
- [x] Add Read→Write memory-source switch with batch-size/R guards.
- [x] Add paired hard-16 runner and reconstructable manifests/gallery.
- [x] Add unit tests for derangement, paired packing, probes, and per-sample CFG.
- [x] Run local syntax/whitespace checks.
- [ ] Run PyTorch unit tests (local PyTorch install lacks `libtorch_cpu.dylib`; defer to user-run NPU checkout, without launching evaluation here).
- [x] Push implementation to GitHub `main` and verify remote SHA (`2e3ff70`).
- [ ] User runs the unit-test command and one-pair smoke on NPU.
- [ ] User runs full 16-prompt NPU experiment and records results.
