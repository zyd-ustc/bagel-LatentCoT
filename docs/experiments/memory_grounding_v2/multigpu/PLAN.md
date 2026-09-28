# Eight-GPU synchronous training implementation

## Objective / user constraints
Enable the user to launch Stage A with eight H200 GPUs on port 20474 in tmux.
Modify only v1, upload verified code, do not launch real BAGEL training. No data
or existing output modification. This is implementation validation, not research evidence.

## Baseline and comparability
Baseline: current single-device v2, 260 tests including export checks.
Preserve native teacher, local whole-prompt correct/shuffle/zero counterfactuals,
K8/body[12,20), Write-only mask curriculum and learning rate 5e-6.
Each rank uses two distinct prompts; eight ranks mean global batch 16. This is
an explicit budget change from global batch 2; learning rate is not scaled.
The eight-GPU config enables deterministic epoch shuffle; each global batch is
partitioned across ranks, never eight replicas processing the same pair.
The last epoch batch cyclically pads within its permutation if needed; N>=global
batch guarantees no duplicates inside a global batch. Record this policy.

## Code translation
Add a small distributed utility: torchrun rank/device setup, NCCL process group,
adapter initialization broadcast, synchronous mean LoRA gradient all-reduce,
collective error checks, rank-zero writes, global metrics and deterministic sampler.
Use explicit all-reduce because BAGEL has multiple public forward methods and
frozen teacher/read branches; do not pretend eight independent jobs are DDP.
Extend A/B/D shared runner; keep C GRPO and eval single-process and fail closed.
Add an eight-GPU Stage-A config and a tmux-friendly foreground launcher.

## Validation / execution design
Unit tests for rank parsing, disjoint sampling, metric reduction, primary-only
output and failure propagation. Real two-process CPU/Gloo backward/all-reduce
must match a single-process global-batch reference; test actual tiny MoT gradients
where practical. Full local/remote CPU suites, shell syntax, torchrun multi-process
validate-only on the real manifest. No real GPU training or image generation.
Main GPU run is the user's next action, not part of this implementation turn.

## Recovery / tooling
Local backup: `/Users/zyd/Documents/LCoT-codex/bagel-v2-multigpu-backup.qCALVo/before-multigpu.tar.gz`.
Back up remote touched files before upload. Preserve .git, environments, datasets,
models and output directories; use checksum-based rsync without --delete.
The local .git points at an unavailable Linux worktree; do not repair it or claim commits.
Experiment skill bash_exec/artifact/memory tools are unavailable, so use the
available terminal, this plan/checklist and durable validation notes instead.
Stop if gradient equivalence, rank safety or unit tests fail; fix before transfer.

## Checklist / revision
See CHECKLIST.md. 2026-09-27: initial contract; scoped to implementation and upload.
