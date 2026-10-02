# CPU-only follow-up: reader evidence, image scoring and data

## Current result and boundary

Parent: `phase1a0_reader8_20474_20260930_115000`, interrupted at update755;
latest saved/evaluated checkpoint750. Original snapshots/checkpoints unchanged.
No generation, VLM scoring, GPU training, checkpoint conversion or Git push.

| Evidence at step750 | Mean paired ΔMSE | Prompt-bootstrap 95% interval | Scope |
|---|---:|---|---|
| correct - initial | -3.4108 | [-4.0326, -2.7765] | 8/8 prompt means improved |
| correct - shuffled | -0.5297 | [-1.5266, +0.6089] | 6/8 improved; interval crosses zero |
| correct - zero | +1.0237 | [+0.3101, +1.8875] | only1/8 improved; zero better overall |

Negative favors correct. Bootstrap unit is prompt (two states kept clustered),
4000 resamples, seed42. These are exploratory intervals conditional on just8
heldout prompts, one training seed and one deterministic shuffled-donor cycle.
Not a population significance claim or image semantic metric. Category and
step-bucket slices are even smaller; late bucket has one prompt, no CI.

Original operational warm-up gate remains true: it never required beating zero.
The test suite now explicitly preserves that boundary. No silent gate or loss
change. Recommendation: do not treat gate passage as proof of useful memory or
start OPD solely on these reader metrics; Teacher>Native semantic evidence is
still required. Warm-up has no generation injection, so warmed A/B alone is not
an image-quality improvement; gate=0 remains native generation.

- [Reader report](reader_diagnosis.md), [full evidence JSON](reader_diagnosis.json).
- [Data audit](data_audit.md), [full audit JSON](data_audit.json).
- [Plan](PLAN.md), [checklist](CHECKLIST.md).

## 1. Reproduce existing-results diagnosis, no GPU

Run from the repository root with a CPU Python. Use a fresh output directory.
Local downloaded originals are under `outputs/phase1a_v4_offline_20261002/source`;
that ignored directory is not committed. Remote originals remain at the paths
in PLAN.md. All nine fetched file hashes matched an independent remote SHA-256
snapshot; the same analyzer ran read-only on remote CPU and then locally.

```bash
PYTHONPATH="$PWD" python scripts/evaluate/analyze_reader_warmup.py \
  --run-dir /path/to/original/main \
  --heldout-data /path/to/reader_heldout.jsonl \
  --output-dir /path/to/new_reader_report
```

Requires original `resolved_config.json`, `run_manifest.json`, `metrics.jsonl`,
`heldout_diagnostics.jsonl` and `warmup_gate.json`. Rejects split hash mismatch,
non-finite metrics, missing/duplicate states, aggregate discrepancies, changed
state/donor/layer grids and training-step gaps. `num_steps=50` is50 schedule grid
points and49 native Euler advances; buckets use the existing49-step sampler's
40%/75% index boundaries, not shifted timestep magnitudes.
Only correct-arm aggregate layer diagnostics exist in these logs; paired
per-state layer control errors cannot be reconstructed without another forward.

## 2. Prepared semantic test file and image scoring

`experiments/data/phase1a_semantic_hard64.jsonl` is both an OPD prompt-data input
and a GenEval2 benchmark input. Contains stable prompt IDs, heldout split,
category, original prompt/VQA/skills/atomicity and original source indices/hash.
Source: existing `geneval2_hard_heldout.jsonl`80 rows. Selection:16 eligible
prompts each at atomicity7/8/9/10; first16 per group, retain source order.
Zero casefold/whitespace-normalized exact overlap with reader train or heldout.
Do not assume this proves independence from any future OPD training dataset:
check that dataset separately before its semantic evaluation.

`category=multi_object_composition` denotes this compositional benchmark;
original per-question skill labels are retained, not replaced or reannotated.
No reasoning text generated. Teacher cache building and actual generation still
require a GPU. Keep the exact same manifest/order for generation and scoring.
Use `--max-prompts 64` with the existing OPD eval entry; a partial8-image run
must not be scored against this64-row benchmark.

After images exist in the existing Phase1A layout `p000/native.png`,
`p000/teacher.png`, `p000/prompt.txt`, etc.:

```bash
PYTHONPATH="$PWD" python scripts/evaluate/prepare_phase1a_score_inputs.py \
  --benchmark-data experiments/data/phase1a_semantic_hard64.jsonl \
  --image-dir /path/to/generated_run \
  --arm native --arm teacher --output-dir /path/to/new_score_inputs

PYTHONPATH="$PWD" python scripts/evaluate/score_geneval2_server.py \
  --benchmark-data experiments/data/phase1a_semantic_hard64.jsonl \
  --image-paths /path/to/new_score_inputs/native_image_map.json \
  --server-url http://127.0.0.1:18086 --output /path/to/native_scores.json
```

Repeat scoring for teacher with `teacher_image_map.json` and a fresh
`teacher_scores.json`. For trained OPD, additionally prepare/score `correct`,
`shuffled`, `zero`; use the same benchmark and generation settings.

```bash
PYTHONPATH="$PWD" python scripts/evaluate/geneval2_report.py \
  --benchmark-data experiments/data/phase1a_semantic_hard64.jsonl \
  --run native=/path/to/native_scores.json \
  --run teacher=/path/to/teacher_scores.json \
  --baseline-run native --output-dir /path/to/new_semantic_report
```

Contract:
- Reject missing/corrupt PNGs, changed prompt/order, mixed image dimensions,
  duplicate paths, missing/extra prompts, partial server batches, wrong VQA
  lengths, bool/NaN/out-of-range probabilities and inconsistent log-GM.
- New server replies echo prompt order; new score files bind benchmark hash,
  prompt order, image-map and image hashes. Older official bare score lists are
  still accepted: they cannot independently prove row order/provenance.
- Overall AM = mean of prompt-level atom AM; overall GM = mean of prompt-level
  atom GM, not GM across all prompts. Atom-weighted AM is separately labeled.
  Summary scores use percent; deltas use percentage points. Absent skills are
  JSON null, not NaN or a zero success/failure score. Zero atom probability gives
  true GM=0; transport log-GM uses the existing1e-8 clamp only.
- Preflight proves prompt/image alignment and geometry, not identical generation
  seeds/NFE/checkpoints. Verify those against the generation protocol separately.
- RPC uses pickle: only connect to a trusted local Soft-TIFA service. Mock-backed
  CPU tests validate transport/aggregation, not VLM accuracy or model scores.

## 3. Reproduce data audit, no split rewriting

```bash
PYTHONPATH="$PWD" python scripts/data/audit_reader_data.py \
  --train-data /path/to/reader_train.jsonl \
  --heldout-data /path/to/reader_heldout.jsonl \
  --export-report /path/to/export_report.json \
  --semantic-benchmark experiments/data/geneval2_hard_heldout.jsonl \
  --semantic-count 64 --output-dir /path/to/new_data_audit
```

Train24547 (spatial21004/count3543); heldout64 (spatial52/count12).
Actual evaluated prefix8 (spatial6/count2), not the whole heldout.
Exact/normalized/id overlap and duplicate counts zero; count/color-template
cross-split hits and Jaccard>=.85 candidates zero under the defined lexical checks.
Other five allowed semantic categories are not covered by this reader export.
Word length diagnostics flag long prompts, not tokenizer lengths or auto-removal.
Original exporter categories/provenance remain heuristic, not human-reviewed.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_reader_offline_analysis.py tests/test_phase1a_image_scoring.py \
  tests/test_reader_data_audit.py tests/test_geneval2_evaluation.py \
  tests/test_reader_warmup_contract.py
```

Full-suite evidence and code/report boundaries are recorded in VALIDATION.md.
Future route: retain current training/checkpoints; revisit useful-vs-zero evidence
and complete independent teacher semantic baseline when GPU resources return.
