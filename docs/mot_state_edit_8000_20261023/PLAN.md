# Actual-state MoT edit v2: repair and restart

User authorization: 2026-10-23, fix the diagnosed image-unchanged mechanism and restart training. New branch `run/bagel-mot-state-edit-64npu-8000-20261023`. Fresh8000 optimizer updates,1000 caption+7000 edit,64 NPUs/global batch64. Same supplied pairs-20261022.jsonl and existing mot_edit_pairs_20261023 corpus. Same lr1e-5,rank32,512px,feedback indices28/35/40,seed17. No new output head, no ViT input, no changes to base prompt/CFG/native frozen weights.

## Changed computation

For each feedback node, start from the actual noisy state y at time t. Recompute frozen native conditional c and guided gvel on y. UND reads the clean estimate y-t*gvel through frozen GEN source features and trainable observation slots; feedback combines that reading with the goal. Private GEN editor receives **y,t**, not the clean estimate, and predicts w via native llm2vae. Its query retains native start/end-of-image tokens routed through frozen UND, while image patches use private GEN adapters. At zero-adapter initialization, without feedback prefix it matches the native unconditional velocity entry point; the private editor still has no original prompt cache.

Let h=gvel-c; D(d)=d-min(<d,h>/max(||h||²,1e-12),0)*h. Let clean=y-t*gvel, estimated_noise=y+(1-t)*gvel and edited_clean=clean-t*delta, with delta=D(w-c). Re-noise using the same estimated noise: **y_new=(1-t)*edited_clean+t*estimated_noise=y-t*(1-t)*delta**. This retains the flow mixture at the actual outer time. It is a discrete clean-image edit and re-noising operation, not a larger Euler step. Recompute native reading and feedback on y_new for the next round. After both rounds, continue the normal native Euler update from the final committed y. No second delta addition, no fixed-anchor overwrite, no single-Euler dt/t attenuation; the physical clean-mixture coefficient1-t remains.

The intermediate preview is a clean estimate, not decoded ground truth. Recomputed native proposals may change after each committed edit; the next UND reads that actual new-state estimate.

## Training and gradients

At each actual state y, target physical velocity toward the paired x0 is (stopgrad(y)-x0)/t; target w=c+D(truth-c). Average the two same-state MSEs and backward once. Keep the first-to-second editor state graph. Native prediction, clean-source feature extraction and targets remain stop-gradient. Cross-round gradient flows through the next editor's noisy input, not through frozen GEN source extraction. Observation/feedback/editor adapters all train in edit stage. Caption stage still updates observation only.

zero_correction_mse averages ||target-c||² on the same visited states. State distribution changes across rounds; old v1 loss series and v2 series have different inputs. Compare model loss with its own same-state zero baseline. Guidance-safe applies to the committed edit delta at each state; it is not a guarantee about later native dynamics or image quality.

## Execution and acceptance

Stop only the prior run's verified supervisor/process groups; preserve its checkpoint/log/output. It stopped atStep7024, latest durable checkpoint6000 (see RUN.md for final audited value). Old heartbeat deleted. New code snapshot and output directory; fresh initialization and optimizer, no v1 adapter/optimizer reuse. Checkpoint kind `bagel_mot_state_edit_v2` prevents accidental v1 resume.

CPU real tiny-MoT checks cover actual-state input, committed displacement, cross-round gradient, all three role gradients, unchanged native K0 and exact optimizer/sampler/RNG resume. NPU512 pilot must exercise caption and two-round backward, checkpoint reload, sampling and finite images before64-rank launch.

Run `/data/zyd_workspace/bagel-LatentCoT/outputs/bagel_mot_state_edit_64npu_20261023_8000`; code `/data/zyd_workspace/bagel-LatentCoT/code/bagel-mot-state-edit-8000-20261023`. Existing bagel-NPU Python and existing native weights. Data and budget stay fixed during monitoring.64 ranks and all four exit logs checked. Save checkpoints10/100/500/1000/1100/2000/4000/6000/8000; semantic32 three arms1100/2000/4000/8000; caption reading8 cases1000/2000/4000/8000.

Completion:8000 updates, final checkpoint, final32-case native/one/two-round real image coverage, four exit0, all final images and caption records inspected and local HTML delivered. Finite loss or image changes alone do not establish semantic improvement. The inherited corpus contains benchmark-derived prompts; report that provenance in downstream evaluation.
