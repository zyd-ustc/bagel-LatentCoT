# Execution record

## Old run stopped, new implementation prepared

Old fixed run stopped intentionally atStep7024. Audited all four nodes: zero remaining workers of that run. Its checkpoint6000, logs and images retained. Supervisor identities verified before signalling; torchrun shut down its workers. Stop audit: `artifacts/mot_state_edit_8000_20261023/old_run_stop_status.json`; remote old launch `intentional_stop_20261023.json`. Old20-minute heartbeat `mot-64-8000` deleted because its pinned run is obsolete.

The old implementation fails the new regression for committed displacement (expected y-t*delta; old returned y-t*(vg+delta)). V2 changes state semantics, actual noisy input to GEN editor, observation learning in edit, cross-round gradients and per-state target recomputation. It keeps the supplied data, frozen native heads and64-card8000-step budget.
