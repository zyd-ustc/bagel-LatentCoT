# Phase 1A v4 — Reader Warm-up before Self-CoT OPD

## Objective and boundaries

Implement the user-supplied v4 mechanism on a child branch of `600d195`.
Phase 1A.0 must train only a side-head memory translation adapter against a
frozen native prompt-bank readout; generation remains native. Phase 1A.1 must
load that adapter and initially train only zero-initialized injection gates
against the existing same-state Self-CoT velocity teacher. This is an
auxiliary/dev **code implementation**, not an H200 experiment result.

Null: K=8 frozen dynamic memory is not naturally readable as a prompt-like
condition bank. Alternative: held-out correct-memory readout MSE improves and
is below shuffled-memory MSE without a ranking loss.

## Baseline and comparability

- Base code: `codex/phase1a-selfcot-opd` at `600d195`; 283 CPU tests passed.
- Train/held-out semantic T2I prompt split and CFG=1, NFE=50 remain unchanged.
- Primary warm-up keys: held-out correct/initial/shuffled/zero MSE, native
  parity, slot utilization. OPD retains plain same-state velocity MSE and the
  independent held-out Teacher>Native semantic gate.
- No new dataset or metric result is claimed in this implementation pass.

## Code translation

| Area | Change | Main risk |
|---|---|---|
| Strict Read | Capture layer-entry hidden and the exact native UND K/V used by Read | Misaligned packed indexes or RoPE |
| Prompt bank | Read native prompt K/V with the same frozen GEN Q and O | Including non-prompt cache rows |
| Reader | Frozen GEN-O plus zero-effect low-rank translation adapter, side-head mode | Accidental injection during warm-up |
| Warm-up | Native rollout, detached states, per-layer MSE, adapter-only optimizer | Teacher gradient leakage |
| OPD | Load warm-up adapter; zero gate per body layer; gate-only training | Step-0 parity or checkpoint mismatch |
| Eval | Correct/shuffled/zero held-out diagnostics only | Donor or timestep mismatch |

## Execution contract

- Local smoke: tiny-MoT exact-K/V, per-layer matching, no-injection parity,
  gradient isolation, checkpoint/CLI contract tests; then full CPU suite.
- H200 pilot/main run: **not authorized by this request**. Warm-up and OPD
  commands are documented but are not executed in this pass.
- Stop if any native parity, strict Read K/V identity, or gradient isolation
  assertion fails. Do not loosen those assertions to make tests green.
- Last-known-good baseline remains the untouched `600d195` branch.
- Code fallback: preserve frozen native attention path, isolate side-head
  capture; if exact K/V cannot be captured, stop rather than recompute from
  mismatched hidden states.

## Revision log

- 2026-09-29: Initial v4 implementation contract; no GPU run or metric claim.
- 2026-09-30: Implemented exact Read bank, symmetric prompt/memory side head,
  reader warm-up runner/eval/checkpoints, checkpoint-bound readiness guard,
  warmed-reader gate-only OPD and T0-v2 deterministic CoT. 301 CPU tests pass;
  full-model CUDA/semantic checks remain for a separately executed pilot.

Checklist: [CHECKLIST.md](CHECKLIST.md).
