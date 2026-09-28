# Implementation checklist

- [x] Read plan, lock scope and preserve source backup.
- [x] Implement Write-only controls and full per-Write outputs.
- [x] Implement stage losses, runtime, training/evaluation commands and configs.
- [x] Test native parity, fixed-state isolation, gradients, caches and CLI validation.
- [x] Document results and limitations. Real GPU training is not part of this turn.

Validation: 255 passed (3.72 s), compileall passed, A/eval preflight and B/C/D CLI
help passed. See VALIDATION.md. H200/NPU pilot and scientific gates remain pending.
