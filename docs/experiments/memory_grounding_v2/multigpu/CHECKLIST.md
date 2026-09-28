# Multi-GPU implementation checklist

- [x] Lock protocol, read current runner and record backup/recovery plan.
- [x] Implement rank/device binding, gradient averaging and disjoint sampling.
- [x] Implement primary-only saves, global logs and foreground launch command.
- [x] Pass multiprocess equivalence/error tests and existing test suite (local: 272 passed).
- [x] Back up/upload to 20474; verify remote CPU tests (272 passed) and eight-process data preflight.

Real BAGEL training remains unlaunched; the user will execute it in tmux.
