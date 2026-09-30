# Fresh restart checklist

Run: `phase1a0_reader8_20474_20260930_115000`; idea: Phase1A.0 Reader Warm-up.

- [x] Fresh restart authorized; no resume; preserve old outputs/unrelated jobs.
- [x] Baseline, shared code/model/data, 8-card budget and metric contract locked.
- [x] Target 8 H200 idle, Python and data hashes verified.
- [x] Existing identical-code 8-card smoke passes; native parity zero.
- [x] PLAN.md, exact launcher and recovery conditions written before launch.
- [x] Old 20470 training stopped at step 208; interruption marker recorded.
- [x] Fresh 20474 main launched: torchrun PID 2641732; durable command/log.
- [x] 10 actual updates confirmed at 11:57:56 UTC; all loss/grad finite.
- [x] Initial native parity max abs=0; all eight scoped workers alive.
- [ ] Full 5000 updates completed, final metrics/gate validated and recorded.
