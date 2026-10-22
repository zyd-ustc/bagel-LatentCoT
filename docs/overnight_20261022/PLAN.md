# 5000-step overnight experiment

User authorization: implement the three recommendations, then start 64 NPU training for 5000 steps. Keep the existing loop-layer-NPU branch and preserve the previous checkpoint and user notes.

## Decisions

1. Native CFG and renorm finish BEFORE the learned correction. Train on that same guided velocity/proposal, rather than a conditional residual that is amplified by CFG at deployment. Add the correction once in FP32, gain=1. Native zero-correction path stays exactly equivalent. Original checkpoints retain their original sampling semantics.
2. Build a separate, traceable photo corpus from existing COCO and Flickr30k generation views. Preserve family splits and original records. Exclude heterogeneous synthetic/edit/OCR pools for this run. Check image readability, caption independence, duplicate pairs, and split conflicts. No generated captions or claim that filtering verifies every object's semantics. COCO/Flickr weights follow their accepted unique image-caption pair counts, rather than sqrt-size over 31 views.
3. Estimate a time-only, spatially constant residual baseline from the SAME training batches using the exact per-time least-squares sample mean. No extra training loss, parameters, or optimizer coupling. Checkpoints retain its statistics. Evaluate native / global-only / full UND on identical held-out inputs and seeds. Include the existing 32 semantic prompts, without training on them.

## Run contract

Fresh native UND initialization; no continuation from the mixed-data 1000 checkpoint. 64 NPU, batch256,512px,rank64 LoRA,lr1e-4,warmup100,clip1,5000 steps. Same residual MSE, now against the deployed guided velocity. Native GEN and copied UND base remain frozen. No contrastive/semantic/preservation loss.

Save steps10,100,500,1000,2000,3000,4000,5000. Fixed natural-pair MSE validation at100,500,1000,2000,3000,4000,5000; 32-prompt three-arm image evaluation at100,1000,3000,5000. Use original benchmark annotations; images and residual metrics are not benchmark success scores.

## Acceptance and operation

Unit checks: native zero-delta parity; delta added after CFG once; guided training/deployment proposal agreement; baseline train-only accumulation/checkpoint/resume; cleaned corpus split/duplicate/caption contract; real tiny MoT optimization and exact resume. Full-model NPU pilot before main launch. Inspect actual pilot images. Verify four nodes/64 ranks, finite loss and nonzero UND gradients after step1, saved step10. Do not terminate unrelated work. Record durable logs, output path, source revision, data inventory, health and resume command. Long run continues detached; request a thread heartbeat for completion/failure and checkpoint images.

## Data correction found during preparation

The inherited family IDs group captions, not necessarily physical images: the same COCO/Flickr image appears under multiple family IDs and across original splits. Keep test > val > train priority, removing conflicting lower-priority records without relabeling. Validation also deduplicates image paths. The initial draft corpus discarded all conflicts including validation; it was rejected before training. The accepted v2 corpus preserves held-out records and removes their train aliases.
