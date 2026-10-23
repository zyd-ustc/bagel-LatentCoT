# Restart checklist
- [x] Preserve old run, stop only its processes; four nodes zero old workers
- [x] New branch and explicit actual-state edit contract
- [x] CPU regression: state input/update, role gradients, cross-round gradient, native K0, exact resume
- [x] Full512 NPU pilot: finite caption/edit gradients, memory, checkpoint restore, actual images
- [x] Deploy pinned source snapshot, launch64 ranks, verify finite metrics/checkpoint
- [x] Quiet-on-healthy scheduled follow-up for new run
- [ ] Caption reading1000/2000/4000/8000 inspected
- [ ] Semantic1100/2000/4000/8000 inspected, readable HTML
- [ ] Step8000, final checkpoint and32×three-arm images, four exit0
