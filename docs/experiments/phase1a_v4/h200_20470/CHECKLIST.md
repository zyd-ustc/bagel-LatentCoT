# H200 20470 Reader Warm-up execution checklist

- [x] Host, eight GPUs, model, Python and source manifests inspected.
- [x] Run scope/data selection/global-batch change recorded in PLAN.md.
- [x] Distributed adapter helper and real two-rank tiny runner tests passed.
- [x] Code committed/pushed and isolated remote directory verified.
- [x] Prompt-only export and preflight passed; source provenance recorded.
- [ ] Remote 8-card smoke completed with finite metrics and native parity.
- [ ] Main 8-card run launched; first actual optimizer step confirmed.
- [ ] Main completion and heldout gate evaluated (not part of launch claim).
