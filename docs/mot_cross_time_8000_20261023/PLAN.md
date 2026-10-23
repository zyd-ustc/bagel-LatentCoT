# Cross-time MoT v3: implement and restart

User authorized implementation and restart on2026-10-23, following CROSS_TIME_FEEDBACK.md. New branch run/bagel-mot-cross-time-64npu-8000-20261023. Fresh8000 optimizer steps:1000 caption +7000 edit,global batch64,512px,seed17,rank32,lr1e-5,clip1,wd0,warmup100 per stage. Exact same supplied pairs-20261022.jsonl and mot_edit_pairs_20261023 index. Old v2 stopped with checkpoint/logs/images retained; no old optimizer/adapters reused.

## Model and state machine

Observation writing and causal caption reading share the same observation UND LoRA, with explicit caption instruction. Native lm_head/llm2vae, native backbone,VAE remain frozen; no new head, no ViT. Observer extracts GEN features at clean proposal with neutral prompt; original target is introduced only in feedback.16 observation slots and16 feedback slots, per-layer KV.

Feedback combines original prompt, current observation and previous plan into a refreshed16-slot plan. Editor retains original prompt KV and feedback prefix, native image boundary tokens/UND routing and image GEN routing. All exported KV is unrotated and gets legal prefix positions on consumption. Current execution time is embedded every step.

Sampling: actual native trajectory saves index12(t=.902439), observes index33(t=.592593), then restores the saved early latent. The plan persists over ALL21updates12..32. Recompute current-state vc/vg and feedback-conditioned w each step; use vg+H(w-vc), where H removes negative guidance projection. Observe the resulting index33 state, refresh plan with previous plan, and repeat once. Continue final state to0.49+42=91 Euler updates for two rounds,70 for one; no decoded intermediate image or external image. Native0-round remains native byte-identical. No same-time x-t*(1-t)delta commit in this pipeline.

## Training and compute

Caption stage: true paired x0 only, original supplied caption; no generated-failure proposal labelled as target prompt. Edit stage retains caption CE on every true x0, weight1.0, separate metrics from edit MSE. All observation/feedback/editor adapters train; original heads stay frozen.

For each pair, construct early paired state x_early=(1-t_early)x0+t_early*epsilon. Roll exact native grid to read_index, observe and construct plan. Restore early state and replay every step with persistent feedback. Uniformly sample one write index per pair; at that actual visited state backprop MSE(w,vc+H((state-x0)/t-vc)), averaged over two rounds. Complete actual replay, refresh late plan for next round and repeat. All rollout states and previous plans are detached; one selected write-step loss per round trains current observation/feedback/editor. This is truncated local gradient, not full21-step or cross-round BPTT. The sampled index is checkpointed sampler RNG. No sparse/coarse substitute for actual forward trajectories.

This adds native rollouts and editor forwards: speed must be measured in the full NPU pilot. Extra computation is recorded, not hidden as the old5s/update budget. Training starts at paired early state; production sampling reaches early state from original noise. Both subsequently use actual replay states; this entrance distribution difference remains explicit.

## Completion and monitoring

Pilot:512px caption +2edit updates,finite all-three gradients and rollout,checkpoint reload,native andtwo-round real PNG inspection. Then64workers on4nodes,master29673,new pinned snapshot. Required final:step8000,step_008000.pt,observation_008000/records.json,semantic_008000/records.json with32×native/one/two actual PNGs and4exit0. Caption1000/2000/4000/8000; semantic1100/2000/4000/8000. Inspect all32cases and8captions,build local HTML. Only edit MSE compares with zero_correction_mse; total_loss additionally includes caption CE. Applied negative fraction remains0. Same budget/config/data/world on resume with optimizer,sampler,64RNG entries.
