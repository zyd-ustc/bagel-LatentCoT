# CPU follow-up validation, 2026-10-02

## Execution boundary

Baseline branch `codex/run/phase1a0-h2008-20260930`, base commit `fb60890`.
No training/inference objective, native model forward, existing gate criteria,
checkpoint or source split changed. Original SIGTERM-interrupted run remains
interrupted; no continuation, image generation or real semantic scoring started.
No Git commit/push or remote deployment requested/performed this pass.

Skill tooling fallback: analysis-campaign's bash/artifact/memory services are not
available in this desktop session. PLAN/CHECKLIST, SHA-bound JSON reports and
these execution records preserve scope and evidence instead. No paper outline
or manuscript claim was created. Ordered slices: reader -> scoring tests -> data.

## 1. Existing-reader evidence

Original source: `/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000/main`.
Split source: `/private/yida_workspace/outputs/phase1a0_reader8_20260930_052322_retry2/data`.
Nine logs/config/data files fetched to ignored local
`outputs/phase1a_v4_offline_20261002/source/{main,data}`. All local SHA-256 values
matched a separate read-only SSH `sha256sum` snapshot. No weights downloaded.
SSH/SFTP required explicit source paths; brace expansion was unsupported.
Slow transfer recovered with compression; only our own local download processes
were terminated/restarted, not any remote task.

The identical stdlib reader analyzer first ran read-only on remote `/usr/bin/python3`
against originals, then locally after transfer. Checkpoints0/250/500/750 and
training755-update summaries matched exactly between remote and local analysis.
The checked-in reader report preserves remote original source paths/hashes.

Local commands from repository root:

```bash
PYTHONPATH="$PWD" /tmp/bagel-v4-tests.nWADtD/venv/bin/python \
  scripts/evaluate/analyze_reader_warmup.py \
  --run-dir outputs/phase1a_v4_offline_20261002/source/main \
  --heldout-data outputs/phase1a_v4_offline_20261002/source/data/reader_heldout.jsonl \
  --output-dir outputs/phase1a_v4_offline_20261002/reader
```

Verdict: `reconstruction_improves_but_zero_control_is_better`.
Correct7.1392 ->3.7284; zero2.7046. Correct-vs-shuffled Δ=-.5297,95% interval
[-1.5266,+.6089]; correct-vs-zero Δ=+1.0237,[+.3101,+1.8875]. Conditional on8
prompts/16 states, one training seed; prompt-cluster bootstrap4000/seed42.
No new layer-control forwards; layer evidence is correct-arm aggregate only.
Do not elevate these exploratory intervals into a population/semantic claim.

## 2. Image-scoring contract and tests

Added `prepare_phase1a_score_inputs.py` and lightweight `image_scoring.py`:
Phase1A `pNNN/arm.png` plus exact prompt manifest, verified PNG, matching geometry,
complete coverage, hash-bound maps. Soft-TIFA transport validates batch/atom
coverage, finite probabilities, order echo and existing clamped log-GM.
Scoring output written only after full success; existing output refused.

Aggregation preserves official prompt-mean AM/GM versus separately reported
atom-weighted AM. Added benchmark/score provenance, prompt-order checks, duplicate
run rejection and JSON null for absent skills. Report output must be fresh.
Original official bare-list cache format remains accepted, with weaker order/
provenance guarantees documented. Trusted-local pickle protocol unchanged.

Tests use8x8 synthetic PNGs and fake HTTP scoring responses: no VLM service or
real image judgment. Includes missing/corrupt/geometry mismatch, prompt/order
errors, duplicate/extra/missing maps, partial batches, invalid atom probabilities,
log-GM mismatch, source hash mismatch, AM/GM arithmetic and fresh-output guards.
Existing operational reader gate test explicitly demonstrates that correct can
be worse than zero without changing that gate.

## 3. Reader data audit and semantic evaluation pack

Final CPU run:

```bash
PYTHONPATH="$PWD" /tmp/bagel-v4-tests.nWADtD/venv/bin/python \
  scripts/data/audit_reader_data.py \
  --train-data outputs/phase1a_v4_offline_20261002/source/data/reader_train.jsonl \
  --heldout-data outputs/phase1a_v4_offline_20261002/source/data/reader_heldout.jsonl \
  --export-report outputs/phase1a_v4_offline_20261002/source/data/export_report.json \
  --semantic-benchmark experiments/data/geneval2_hard_heldout.jsonl \
  --output-dir outputs/phase1a_v4_offline_20261002/data_audit_balanced
```

Original train24547 and heldout64 retained byte-for-byte; exact/normalized/id
overlaps and duplicates zero. Defined count/color lexical template overlap zero;
word-set Jaccard>=.85 candidate pairs zero. These are lexical checks, not proof
against all semantic leakage. Existing reader labels cover only count/spatial;
train spatial21004/24547 (85.6%). Word counts>128:1020,>256:69,max858; heldout
none>128. No automatic truncation, reannotation, balancing or split rewrite.

Prepared `experiments/data/phase1a_semantic_hard64.jsonl` from existing independent
80-row GenEval2 heldout, with source SHA, original indices, prompt IDs and VQA/
skills unchanged. Count/color or semantic labels not invented from a model.
Naive first64 would have atomicity20/20/20/4; final deterministic stratified pack
uses16 per7/8/9/10. Zero normalized exact overlap with reader train and heldout.
Source selection and manifest hash are checked by a repository regression test.

Final manifest SHA-256:
`fc8639160731361cc05037813fa698afc45f53b419af4d872a9945a93c805bef`.
No teacher reasoning, image or semantic score exists for this new pack yet.

## Regression evidence and next route

Final command:
`PYTHONPATH="$PWD" /tmp/bagel-v4-tests.nWADtD/venv/bin/python -m pytest -q`.
Result: **375 passed in21.05s** (previous baseline329). `git diff --check` passed.
A temporary test-edit placement error was corrected: VQA alignment assertions
belong to the benchmark fixture test, not the geometry test. Final full suite
includes both guards passing.

Claim update: reconstruction learning has partial support, shuffled specificity
is fragile, nonzero usefulness versus zero is contradicted on the observed small
heldout set. Image semantic benefit remains unresolved, not measured.
Next route: preserve current checkpoints/protocol; inspect zero-control/readout
scale evidence before claiming useful memory. When GPUs return, complete the
independent teacher semantic baseline on the fixed64 manifest before formal OPD.

Reusable lesson: passing plumbing tests or an operational gate is not semantic
success. Cluster repeated timestep measurements by prompt, expose a stronger
null comparator, retain original split provenance, and never silently drop
missing images/scores from a comparison.
