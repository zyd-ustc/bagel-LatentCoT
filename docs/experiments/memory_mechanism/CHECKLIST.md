# Implementation checklist

- [x] Read the full source plan; confirm `bagel-LatentCoT-v1` target.
- [x] Lock six-arm, same-state, fresh-memory protocol and suffix semantics.
- [x] Save previous inference entry points outside the working tree.
- [x] Implement strict-null and six-mode decoder/velocity path.
- [x] Implement paired probe, independent images, traces and gallery.
- [x] Replace superseded zero-shot CLI routes with migration errors.
- [x] Add and execute CPU tests; syntax/dry-run checks (196 tests passed).
- [x] Document NPU smoke/main commands and remaining verification limits.
- [ ] NPU smoke / main experiment (not requested for this implementation turn).

## H200 / CUDA adaptation

- [x] Default to current Python, CUDA and repository-local model path.
- [x] Preserve allocated GPU visibility; one paired worker per device.
- [x] Validate requested backend/index/BF16 before model loading; record hardware.
- [x] Keep the six-arm scientific protocol unchanged; 204 local CPU tests pass.
- [ ] Real H200 weights / kernels / memory / timing validation (not run).
