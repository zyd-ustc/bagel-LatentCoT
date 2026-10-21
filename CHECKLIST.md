# Execution checklist
- [x] Read old implementation and protect untracked notes and historical artifacts.
- [x] Select direct spatial UND correction; initialize from native UND, not old C2/Stage A.
- [x] Replace implementation, entry points, configs, scripts, docs, and tests.
- [x] Pass CPU contract tests (10 tests, real tiny MoT, exact resume).
- [x] Pass full-model NPU training/save/reload/sampling pilot (4 updates, 41.1 GiB).
- [x] Audit/stop old training on all 4 nodes (no old training remained).
- [x] Launch main 64-rank training and verify finite, advancing updates (observed through step 84, checkpoint 10 saved).
