# UND spatial correction / loop-layer-NPU

User-authorized destructive replacement of the old training architecture on this branch. No new branch. Keep native BAGEL, corpus assets, historical checkpoints and results. Remove old Memory/consumer/reentry/semantic/partial-prompt implementations, scripts, configurations, and tests; do not remove unrelated untracked research notes.

## Method
Frozen native GEN proposes xhat=x_t-t*v0 and same-pass image KV. A separate frozen copy of pretrained native UND layers, adapted with rank-64 Q/K/V/O LoRA, processes one query per spatial patch (clean proposal + original 2D position embedding + current time). It attends to original prompt and same-layer GEN context through ordinary attention. A zero-initialized dense head predicts velocity correction. Only v1=v0+r is applied; no GEN-layer injection, no second GEN correction pass. No C2/Stage A resume. MSE against epsilon-x0-v0 is the only training objective.

## Run contract
Existing paired-generation corpus and inherited train/val families. Full captions, no text CE, Teacher, cosine, ranking, keep loss, or privileged text. Seed 17; 512 px; same 50-node shift-3 schedule; correction trained and sampled at the first 25 updates. Initial main budget 1000 optimizer steps, global batch 256 on 4 nodes x 16 NPUs. Rank-local microbatch 1, accumulation 4. AdamW 1e-4, clip 1, warmup 100. Activation checkpointing disabled for the main run after the NPU profile: 47.95 GiB peak and 0.493 s/sample, versus about 41.15 GiB and 0.55–0.57 s/sample with checkpointing. Save 10/100/500/1000; fixed validation flow diagnostics and native/corrected images.

## Execution
Implement and test the new contract; run a full-model NPU pilot including multiple updates, checkpoint reload, frozen-native checks, and paired image generation. Use pilot memory evidence to set checkpointing for the main run. Audit all four nodes for old project training and idle devices; stop only identified old project training. Deploy an immutable fresh code snapshot, launch 64 ranks detached, then confirm all ranks and finite optimizer-step logs. Save run config, commands, node PIDs/logs, code revision, training/sampler/RNG states. Completion of this assignment means the new training is genuinely advancing, not all 1000 steps completed.
