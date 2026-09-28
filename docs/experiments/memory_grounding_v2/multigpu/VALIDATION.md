# Validation record — 2026-09-27

## Completed

- Local full suite: **272 passed in 5.28s**. Python environment:
  `/tmp/bagel-mechanism-tests.WOiF4o/bin/python` (PyTorch 2.8 CPU).
- Remote full suite on port 20474: **272 passed in 15.62s**, with
  `CUDA_VISIBLE_DEVICES=""`, using `/private/software/conda/envs/lcot/bin/python`
  (PyTorch 2.5.1+cu124). Includes real two-process Gloo/tiny-BAGEL backward and
  Adam equivalence against the corresponding mean-loss reference, synchronized
  initialization, rank-safe saves and injected-error propagation.
- Shell syntax and changed Python module compilation passed. The actual remote
  launcher ran `torchrun` with eight workers and `--validate-only`, exit 0.
  All workers validated the full **115,883-record** deduplicated manifest;
  rank zero reported local batch 2/global batch 16 and the expected v2 protocol.
- Validation-only launcher logs are under
  `/private/yida_workspace/.bagel-v2-multigpu-validation-c09ThS/`.
  No BAGEL weights, NCCL process group or real training were started.
- Scoped source/config/test/document files were uploaded using checksum-based
  rsync, without deletion. No datasets, models, previous outputs or .git changed.

## Recovery

Local pre-change archive:
`/Users/zyd/Documents/LCoT-codex/bagel-v2-multigpu-backup.qCALVo/before-multigpu.tar.gz`.

Remote pre-change archive (776K):
`/private/yida_workspace/.bagel-v2-multigpu-backup-jG3WR6/previous-code.tar.gz`.

## Limits / next action

CPU equivalence and eight-process preflight do not validate actual NCCL transport,
H200 numerical parity, full checkpoint integrity, peak training memory or loss
convergence. Existing GPU processes were observed using approximately 71–94 GB per
card during implementation and were not stopped. Check resource availability
before executing the documented foreground launcher in tmux. Every rank loads a
complete frozen backbone; this is not a sharded-memory training strategy.

This record is implementation evidence, not a training result or a semantic
quality claim. No real GPU training has been launched by the assistant.
