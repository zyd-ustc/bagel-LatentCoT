# Prompt-KV visibility hard-16 checklist

- [x] Define same-run keep/mask comparison and fixed hard-16 contract in `PLAN.md`.
- [x] Implement per-sample prompt-KV mask in MoT attention for every generation layer.
- [x] Route mask to conditional branch only and add eight-cell paired evaluator.
- [x] Add targeted mask/routing tests; local syntax, Ruff F rules, and diff checks pass.
- [ ] Run PyTorch tests on NPU checkout (local macOS PyTorch lacks `libtorch_cpu.dylib`).
- [x] Push code to GitHub main and verify remote SHA.
- [ ] User runs 2-prompt pilot on NPU and checks complete manifest/gallery.
- [ ] User runs full hard-16 comparison on NPU and records findings.
