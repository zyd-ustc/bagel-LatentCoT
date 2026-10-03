# Fixed reader injection checklist

- [x] Parent checkpoint, controls and user-selected intervention specified.
- [x] Plan and evidence boundary recorded; existing data reused.
- [x] Implementation and tiny-model/CPU tests passed; gate-zero trajectories,
  per-step Read, B=0 control, independent selection and corruption rejection verified.
- [ ] Main branch published and remote CPU preflight checked.
- [ ] User-run H200 smoke/main completed (not launched by the assistant).
- [ ] Automated semantic scoring and comparison interpreted.

Validation on 2026-10-03: final full CPU suite: 425 passed in 38.29 s.
Reader-injection targeted suite: 20 passed. Shell syntax and diff checks passed.
No full-model GPU generation or semantic scoring has run.
