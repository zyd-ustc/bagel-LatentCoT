# Deployment and dataset audit — 2026-09-27

Target: `node-12` / `root@vr.turbo-ai.com:20474` (H200).
Code: `/private/yida_workspace/bagel-LatentCoT`.
Pre-upload backup:
`/private/yida_workspace/.bagel-memory-v2-backup-20260927-tP3bUg/previous-code.tar.gz`.

Uploaded source/configs/tests/docs using checksum-based rsync without deletion.
Preserved remote-only files, Git metadata, environments, models, datasets and
outputs. Post-upload checksum dry run showed no source differences.

Remote verification:
`CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" /private/software/conda/envs/lcot/bin/python -m pytest -q`
→ **255 passed in 11.32 seconds**. No GPU training or image generation launched.

## Requested dataset

`/private/codes/exp/deep_learning/data/datasets/cort_sft_133k/qwen_latent_cot_v3_adapted_20260816/manifests/stage12_stage2_train.jsonl`

Full read-only JSONL scan: 204,154 valid records; all have nonempty prompt;
115,883 unique prompt strings; 88,271 excess duplicate-prompt rows across 86,830
duplicate groups. All sample_id values are unique.
Sources: sharegpt4o 18,954; unicot 4,776; midjourney_v6 5,570;
cort114_echo4o 135,916; cort114_sharegpt4o 38,938.
Exact prompt overlap with repository hard_heldout (80 prompts): zero. This is
not a near-duplicate/semantic leakage audit.
218,338 image path references; the first 12 sampled paths exist. Not all images
were checked, decoded or downloaded. Stage A native-teacher does not use them.

## Compatibility verdict

- Stage A / B native-teacher: prompt field is compatible, but full direct load
  fails because the loader requires distinct conditioning inputs. Actual remote
  `--validate-only` confirmed `ValueError: need enough distinct conditioning
  inputs for a valid shuffle control`. Recommend an independently exported,
  deduplicated prompt manifest preserving original sample_id as id.
- B target-flow: not directly compatible. Current schema is chains with
  steps/image_paths; loader requires id/source_image/instruction/target_image/
  is_noop/edit_type. Multi-turn chain examples contain fix/output_image, but
  pair mapping and supervision quality must be explicitly adapted/validated.
- C semantic GRPO: lacks vqa_list/skills per-atom reward metadata; cannot be
  substituted directly for the current GRPO training manifest.

No source dataset files, loader code or training configs were changed in this
deployment. No deduplicated dataset has been generated. Await authorization for
that separate conversion before supplying a full-dataset training command.

## Authorized unique-prompt export (subsequent request)

User authorized the export. Created:
`/private/yida_workspace/datasets/memory_grounding_v2/stage12_stage2_prompt_unique_20260927/prompts.jsonl`
with a sibling `summary.json` containing source/output hashes and source counts.
115,883 unique prompts; 88,271 duplicate rows skipped in the new artifact only.
Exact original prompt strings, original order and first-occurrence sample_id as
id are preserved. The exported JSONL is 38,983,113 bytes.

Exporter: `scripts/data/export_unique_prompts.py` (uploaded to 20474); 5 local
tests passed in 0.02 s and 5 remote CPU tests passed in 0.03 s. Full Stage-A
`--validate-only` succeeded with **115,883 records**, without max-prompts truncation.
No training launched and no default training configuration changed.

Source SHA-256 verified unchanged after export:
`344f721727c9359259175a241fafa460df1a5c8a7550c4b4f4a207371872e0a0`.
Output SHA-256 independently verified:
`21110992d64ec31a3c57d8d94e9e602cd920bf7dbe28dfdfc525346f4d6d4ccd`.
Exact overlap against current 80-prompt heldout remains zero. No image files were
copied or altered; this is a prompt-only native-teacher manifest, not B-Edit pairs
or C-stage per-atom reward metadata.
