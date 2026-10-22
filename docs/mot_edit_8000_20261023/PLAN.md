# MoT internal edit: 64 NPUs, 8000 total optimizer steps

User authorization: 2026-10-23, implement the approved MoT continuous feedback plan on a new branch, use `/data/250010019/unimoe_r3/qwen_image_vllm_20261019/pairs/pairs-20261022.jsonl`, train on64 NPUs for8000 steps, schedule checks.

## Fixed contract
- Branch: `run/bagel-mot-edit-64npu-8000-20261023`.
- Native BAGEL frozen; UND observation/feedback and GEN edit use private rank32 Q/K/V/O LoRA. Existing lm_head/llm2vae frozen, no new output head.
- Total8000 updates: A observation caption1000; B guidance-safe two-round editing7000. One optimizer update per global step, no budget hidden in stage transition.
- Batch64, one pair/rank/update;512px. AdamW lr1e-5, weight decay0, clip1, stage warmup100, seed17.
- Exact supplied JSONL is indexed, duplicate UID/image removed; prompt families split train/val/test at98/1/1. No external corpus or automatic semantic filtering. Pair provenance and benchmark labels retained: these data contain benchmark-derived prompts, so benchmark scores after this training are not held-out generalization evidence.
- Feedback nodes are actual shifted grid indices28/35/40, two rounds per node; training samples those times. Clean-proposal observer at t0, edit at actual outer t. Keep native prompt on base GEN; editor receives only feedback KV, not original prompt KV.
- Primary result:32 fixed native/one-round/two-round complete image samples; caption reading and paired objective loss are separate diagnostics.
- Metrics: stage, loss, zero_correction_mse(B), gradient per role, finite loss, negative applied guidance fraction, throughput, memory. Loss alone is not semantic success.

## Implementation and execution
- Add `qwen_latent_cot/mot_edit` and new config/CLI/64-rank launcher; reuse native backbone, codec, distributed reduction and RNG utilities. Keep old paths intact.
- CPU tiny real-MoT checks: native role isolation, native output heads, feedback gradients, stage switch, two-round fixed anchor and checkpoint resume.
- Full512 NPU pilot: caption backward, edit two-round backward, checkpoint restore, image sampling. Then64-rank run.
- Remote code: `/data/zyd_workspace/bagel-LatentCoT/code/bagel-mot-edit-8000-20261023`.
- Run: `/data/zyd_workspace/bagel-LatentCoT/outputs/bagel_mot_edit_64npu_20261023_8000`.
- Python: existing `venvs/bagel-NPU/bin/python`; no shared environment changes.
- Current ended benchmark's dedicated27B server occupies NPU0/1. Save service identity and confirm no in-flight scoring before releasing this task's service. Leave unrelated processes intact.
- Stage checkpoints500/1000/1100/2000/4000/6000/8000. Semantic samples at1100/2000/4000/8000,32 prompts, native cached.
- Atomic trainable-only checkpoints contain all three role adapters, optimizer, sampler, per-rank RNG, configuration and data provenance.
- Health failure: preserve logs and stop this run's workers for nonfinite loss or implementation failure. Fix cause before exact resume; do not change method/data/budget from heartbeat.

## Monitoring and recovery
Every20min heartbeat reads pinned status and four logs. Healthy unchanged state stays quiet. Report stage transition, completed image milestone, failure or completion. Download images and inspect actual native/one/two-round outputs. Completion requires step8000 checkpoint plus semantic008000 records and image coverage. Do not start unrelated SFT or restore stopped old training.

Available harness has no bash_exec/artifact/memory tools from the experiment skill; use actual exec_command sessions and pinned durable files instead.

Checklist: [CHECKLIST.md](CHECKLIST.md). Execution evidence: [RUN.md](RUN.md).
