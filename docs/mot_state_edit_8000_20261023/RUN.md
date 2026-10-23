# Execution record

## Old run stopped, new implementation prepared

Old fixed run stopped intentionally atStep7024. Audited all four nodes: zero remaining workers of that run. Its checkpoint6000, logs and images retained. Supervisor identities verified before signalling; torchrun shut down its workers. Stop audit: `artifacts/mot_state_edit_8000_20261023/old_run_stop_status.json`; remote old launch `intentional_stop_20261023.json`. Old20-minute heartbeat `mot-64-8000` deleted because its pinned run is obsolete.

The old implementation fails the new regression for committed displacement (expected y-t*delta; old returned y-t*(vg+delta)). V2 changes state semantics, actual noisy input to GEN editor, observation learning in edit, cross-round gradients and per-state target recomputation. It keeps the supplied data, frozen native heads and64-card8000-step budget.

## CPU and deployment

Six real tiny-MoT checks passed (0.343s): native head/frozen weight preservation, actual noisy-state editor input, observation/feedback/editor gradients in edit, cross-round w2→w1 gradient, committed clean-image displacement, K0 byte equality, activation-checkpointed forward equivalence, exact optimizer/sampler/RNG resume. The committed-displacement regression failed on v1 before the change. Log: `artifacts/mot_state_edit_8000_20261023/cpu_tests.log`.

Pinned remote source b4bb9c7 deployed to the new code directory, separate from the old run. NPU512 pilot launched on node0 NPU15 with one caption and two edit optimizer updates; awaiting evidence and images before64-rank launch.

## NPU candidate rejected before64-rank launch

The first NPU512 pilot (one caption/two edit updates) completed with finite three-role gradients and29.35GiB peak. Actual image inspection showed severe colored grid artifacts after direct full-displacement latent commits. It was not launched on64 cards. Evidence retained in `artifacts/mot_state_edit_8000_20261023/pilot/` and the remote `_pilot` output.

The direct update y-t*delta preserved a velocity reference, not the estimated noise. Replaced it with flow-consistent re-noising: clean=y-t*vg, noise=y+(1-t)*vg, edited_clean=clean-t*delta, then y_new=(1-t)*edited_clean+t*noise = y-t*(1-t)*delta. The next round reads the actual new state. Added a regression checking this full mixture identity; cross-round and observation gradients still pass. This is a physical mixture coefficient, not a tuned injection gain. A separate32-edit-update single-NPU pilot will check the corrected implementation before launch.
