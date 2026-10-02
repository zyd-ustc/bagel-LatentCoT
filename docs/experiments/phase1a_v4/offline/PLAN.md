# CPU-only follow-up plan

## 1. Objective

Parent: `phase1a0_reader8_20474_20260930_115000`, code baseline `fb60890`.
User order: existing-results diagnosis -> image-scoring tests -> data audit.
Question: does reader reconstruction improve over initialization, shuffled memory
and zero readout, and can future semantic comparisons be scored reproducibly?
This is development/error analysis, not a manuscript campaign or a new GPU run.

## 2. Boundary and comparability

Preserve original logs, checkpoints, manifests, train/heldout splits and training
objective/gate. Read-only SSH retrieval; all analysis runs locally on CPU.
Same recipient state comparisons only. Resample prompt clusters, not individual
timesteps/layers as independent observations. Image metrics are separate from
reader MSE; CPU fixtures are not model scores. No new datasets.

## 3. Slice plan

| Order | Slice | Question | Evidence | Completion |
|---|---|---|---|---|
| 1 | offline-reader | Correct beats which controls, at which layers/states/prompts? | Original metrics and heldout JSONL, paired prompt bootstrap | Reproducible JSON/Markdown report and CPU tests |
| 2 | image-scoring | Are image/score coverage, alignment and AM/GM aggregation reliable? | Existing GenEval2 heldout, synthetic scoring fixtures | Evaluation contract + scoring regression tests; no VLM inference |
| 3 | data-audit | What coverage, duplicate/template and overlap risks exist? | Current reader train/heldout and export provenance | Read-only audit report, CPU tests and current-entry documentation |

## 4. Hypotheses and claim boundary

Correct below initial supports reconstruction learning. Correct below shuffled
supports prompt specificity. Correct worse than zero contradicts a claim of
useful nonzero reconstruction under this metric, not automatically image harm.
Eight evaluated prompts and one training seed limit generalization.

## 5. Assets and dependencies

Remote main: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000/main`.
Reader splits: `/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/data`.
Existing GenEval2 heldout: `experiments/data/geneval2_hard_heldout.jsonl`.
Local CPU Python: `/tmp/bagel-v4-tests.nWADtD/venv/bin/python`.
Dedicated skill artifact/memory/bash tools unavailable: use checked-in plan,
checklist, SHA-256 provenance, ordinary execution logs and reports instead.

## 6. Execution strategy

Retrieve small logs/data only; no model download. Unit-test each tool, then run
against the real snapshot. Separate generated reports from original inputs.
No long GPU jobs, polling automations, schema migrations or destructive edits.
If retrieval fails, record partial status rather than substituting synthetic
results. Scoring service availability is not required for mock-backed tests.

## 7. Reporting

Report support, contradiction and ambiguity separately. Retain per-prompt/state/
layer evidence, split SHA-256 and explicit score units. Never silently turn
missing images or absent skill groups into successful/zero scores.

## 8. Checklist

See [CHECKLIST.md](CHECKLIST.md); initial frontier is original-asset retrieval.

## 9. Revision log

- 2026-10-02: scoped to CPU-only implementation and existing evidence; no new
  training, no existing training gate change, no Git push requested.
- 2026-10-02: SSH/SFTP transfer slow; run identical stdlib analysis read-only on
  remote CPU, verify independent SHA-256 snapshot, then reproduce locally after
  transfer. No remote files written or model loaded.
- 2026-10-02: existing 80-row semantic benchmark is atomicity ordered. Replace
  naive first-64 selection (20/20/20/4) with deterministic equal atomicity quotas
  (16 each at 7/8/9/10), retaining VQA annotations and original output order.
