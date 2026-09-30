# Phase 1A v4 local validation — 2026-09-30

Status: code implementation verified on CPU. No BAGEL-7B weights, H200/NPU
training, heldout semantic score, or full-model readability result was produced.

## Environment and result

The existing local Python environments could not import PyTorch because their
`libtorch_cpu.dylib` was missing. Tests used an isolated temporary virtualenv;
the existing environments and repository dependency declarations were preserved.

Python 3.13, PyTorch 2.8.0, torchvision 0.23.0, transformers 4.56.2.

```text
python -m pytest -q --tb=short
301 passed in 8.14s
```

`git diff --check` also passed. No remote training or Git push was executed.

## Mechanism coverage

| Contract | Verified behavior |
|---|---|
| Exact strict Read KV | Captured K/V equal the memory rows consumed by native attention, including nonzero RoPE; entries are detached and match their layer; cache remains unchanged; Read stops before suffix |
| Query and target | Memory and prompt bank use the exact same post-norm/RoPE GEN Q; native GQA repetition matches manual attention; prompt target is detached |
| Warm-up isolation | Changing trained A/B leaves native velocity exactly unchanged; only A/B receive loss gradients; rollout states and prompt cache receive no warm-up gradient; B=0 gives the expected initial A=0-gradient/B=nonzero-gradient behavior |
| OPD initialization | Nonzero warmed adapter plus zero gates equals native exactly; only gates are trainable; activation checkpointing preserves all gate gradients after gates open |
| Artifact and readiness | Adapter-only warm-up artifact, compatible OPD loading, schema/names/shapes/finite-value checks, checkpoint/source hash provenance, disjoint train/heldout prompts, all four readiness conditions, and CLI preflight without BAGEL loading |

The CPU runner smoke executes **two training steps on a real tiny MoT**, saving
checkpoints, optimizer state, per-layer metrics, heldout diagnostics, readiness
report and completion status. Frozen backbone/writer/gate weights remain
unchanged. Shuffled/zero overrides occur only for heldout evaluation; none enter
the training loss. This is a wiring test, not a scientific pilot.

## Remaining full-model checks

Before formal H200 OPD, run a short reader warm-up pilot on curated semantic
train/heldout data, review memory/attention utilization curves, and obtain a
checkpoint-bound passing `warmup_gate.json`. Confirm full-model native parity
with the actual CUDA/FlashAttention stack. Retain the independent heldout
Teacher>Native semantic evidence for OPD. The code enforces artifact and score
provenance checks; it does not substitute for the external scorer's validity.

The first implementation is B=1 and single-device. Exact resume, distributed
training, adapter/Q capacity relaxation, Draft-Verify and Phase 2 loop
supervision remain outside this implementation.
