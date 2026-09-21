# Phase 0.5 full-800 multi-seed checklist

- [x] Import and verify the official 800-prompt GenEval2 source.
- [x] Remove Z5 and Z7 from the default arm matrix while retaining Z6.
- [x] Add isolated scoring-Python support to the launcher.
- [x] Add a deterministic three-run orchestration script.
- [x] Add a Z0 seed-variability summary.
- [x] Run local unit and shell-syntax tests (117 passed).
- [x] Push the implementation to `bagel/main` (`19f11b9`).
- [x] Run a bounded remote smoke test (1 prompt, Z0/Z3/Z6, 3/3 images).
- [x] Start the full remote experiment and verify all 16 workers are live.
- [ ] Inspect the final summaries after completion.
