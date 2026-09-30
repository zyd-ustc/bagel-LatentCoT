# H200 20470 Reader Warm-up execution checklist

- [x] Host, eight GPUs, model, Python and source manifests inspected.
- [x] Run scope/data selection/global-batch change recorded in PLAN.md.
- [x] Distributed adapter helper and real two-rank tiny runner tests passed.
- [x] Code committed/pushed and isolated remote directory verified.
- [x] Prompt-only export and preflight passed; source provenance recorded.
- [x] Remote retry-2 8-card smoke completed with finite metrics and native parity=0.
- [x] Main 8-card run launched; 183 actual optimizer steps confirmed on 2026-09-30.
- [x] User-requested stop at step 208; original artifacts preserved.
- [ ] Main completion/gate not reached: fresh run continues on 20474 (see ../h200_20474).
