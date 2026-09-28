# Normal R ablation checklist

- [x] Lock normal-only R2/4/6/8, first 8 of hard128, unchanged noise and CFG.
- [x] Implement round recurrence with R2 parity and no extra non-memory carry.
- [x] Replace active six-arm CLI/gallery/metrics with R2-referenced four-arm run.
- [x] Pass unit, artifact-pipeline, launcher, syntax and dry-run checks (226 tests).
- [x] Document runnable GPU command and metric/schema changes.
- [ ] Real GPU smoke/main run (not authorized or executed in this turn).
