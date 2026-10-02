# Reader data audit

Original data unchanged. CPU-only lexical/provenance audit.

| Split | Rows | Count | Spatial | Median words | Exact duplicate extra rows |
|---|---:|---:|---:|---:|---:|
| train | 24547 | 3543 | 21004 | 14 | 0 |
| heldout | 64 | 12 | 52 | 21.0 | 0 |
| evaluated_prefix | 8 | 2 | 6 | 22.0 | 0 |

Exact split integrity: `True`.
Cross-split checks: `{"count_color_template_heldout_hits": 0, "exact_prompt_overlap": 0, "normalized_prompt_overlap": 0, "prompt_id_overlap": 0}`.
Lexical Jaccard >=.85: 0 candidate pairs affecting 0 heldout prompts; not proof of semantic leakage.

## Coverage

Existing train/heldout categories and lexical labels are retained. Missing categories:
attribute_binding, multi_object_composition, action_relation, rare_concept, reasoning_heavy_t2i

## Prepared image-scoring prompts

64 prompts from 80 existing GenEval2 heldout rows.
equal quota per source atomicity; first eligible within group; output source order; no score-based selection.
casefold/whitespace normalized exact prompt overlap with reader train and heldout.
Original VQA questions/answers and skills retained. No new teacher CoT or image scores.

## Limitations

- Labels are existing heuristic/export metadata, not human semantic annotations.
- Word counts are not BAGEL tokenizer lengths.
- Lexical templates and unordered Jaccard are risk flags, not semantic equivalence.
- A prefix of heldout is not a stratified or full-heldout evaluation.
