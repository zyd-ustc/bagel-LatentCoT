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

## v2 prompt-as-memory refactor (2026-09-23)

- [x] Capture causal prompt EOS hidden at the configured body-entry depth while retaining normal prompt KV.
- [x] Replace conditional body-entry memory with deterministic prompt-aware `M_init` each timestep; `m0` Write arm uses exactly this initializer.
- [x] Remove the adapter option from the training-free script; keep the separate trained-checkpoint evaluator unchanged.
- [x] Version manifest schema and document that v1/v2 arms are not directly poolable.
- [x] Run local syntax and targeted static checks.
- [ ] Run PyTorch unit tests and the one-pair NPU smoke; local PyTorch runtime is unavailable.
- [ ] Run the full 16-prompt v2 evaluation on NPU after smoke passes.
