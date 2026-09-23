# Phase 1 Pair-Grounded Checklist

- [x] Add a real Read-only model API that never executes suffix.
- [x] Add paired source/instruction/target validation; remove teacher gate dependency.
- [x] Add pair-memory delta loss and native flow-state helper.
- [x] Add Phase 1.1 UND-Q-only trainer and locked config.
- [x] Add Phase 1.2A GEN-Q-only target-flow trainer and locked config.
- [x] Add unit tests for loss, data contract, route policy, and STOP semantics.
- [x] Run local compile/diff checks and NPU unit tests (`37 passed`).
- [x] Validate NPU smoke manifest (`2,402` Stage-A records).
- [x] Run 1-step NPU Phase 1.1 and Phase 1.2A model smokes.
- [x] Calibrate no-op gradient weight with two 64-step NPU Phase 1.1 pilots.
- [x] Run a 32-step Phase 1.2A pilot with the calibrated Read adapter.
- [x] Complete and record the calibrated 1000-step Phase 1.1 NPU run.
- [ ] Compare fixed held-out memory metrics and instruction retrieval against
  the frozen/reference controls before selecting a Phase 1.1 winner.
- [x] Inspect final diff; unrelated dirty files and experiment outputs preserved.
