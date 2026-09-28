# Normal-only R ablation — implementation contract

User scope: modify local bagel-LatentCoT-v1 inference only. Remove all other arms
from the active evaluation, run normal with total R=2,4,6,8 on 8 hard prompts.
Do not launch experiments, upload files, or change training APIs implicitly.

## Baseline and comparability

Use the first 8 entries of geneval2_hard_128.jsonl, i.e. the first 8 prompts of
the completed hard64 run. Preserve its noise schema, seed=42, K=8, body=[12,20),
512x512, 49 denoising evaluations, CFG, initialization and fresh-memory policy.
R counts 1 strict Read and R-1 Writes. Prefix/suffix execute once; each Write
resets every non-memory row to the original body-entry state and recycles only
the preceding round's memory. Same behavior in each independent CFG branch.

Normal/R=2 replaces native as the same-state reference trajectory and metric
denominator. All four R values see the same reference x_t per probe step. The
reference trajectory's final latent supplies the R=2 image; other R values have
independent trajectories from identical initial noise. Never generate native,
null, zero, shuffled or frozen images in this runner. Output 8x4=32 images/seed.
Old six-arm model helpers remain for historical regression coverage, not CLI use.

## Code / outputs

Extend opt-in decoder and velocity engine with validated total round count.
Replace active bagel_memory_mechanism.py protocol with a new normal-R schema,
four-column HTML, per-step velocity/hidden deltas against R2, attention round
labels, manifests/source hashes and strict merge. Retain launch filename and
GPU visibility protections; default 8 prompts with fixed four R values.
Packed pairs remain an execution unit for comparability (8 prompts -> max 4 GPUs).

## Verification and resource boundary

CPU tests: counts/order for R2/4/6/8, non-memory reset, memory recurrence, suffix
once, R2 numerical parity, round argument reaches conditional/unconditional
branches, reference-state probe, fresh starts, 32-image fake pipeline, shard
merge rejects incomplete/stale/wrong-schema outputs. Full local regressions,
syntax and dry-run. User-run GPU smoke: 2 prompts, 3 schedule points; full run
only after that passes. Stop on nonfinite tensors or missing outputs.
No quality claim: velocity and pixel MAE remain behavioral distances. Larger R
costs more transformer FLOPs; this is not compute-matched. Runtime not measured.

Existing .git points outside this machine; no new branch or commit is claimed.
Skill bash_exec/artifact/memory tools unavailable; use existing tools and these
durable local notes. Previous reports/results remain unchanged. Next action
after implementation is user-approved transfer and a bounded GPU smoke.
