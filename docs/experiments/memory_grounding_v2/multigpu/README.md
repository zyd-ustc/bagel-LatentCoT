# Stage A: H200 8-GPU synchronous training

Run the following inside tmux on port 20474. This launches training in the
foreground; no job is started by the code upload or validation procedure.
Confirm that existing GPU jobs can coexist before launching. This is replicated
data parallelism, not model sharding: every rank loads the full frozen backbone.

```bash
cd /private/yida_workspace/bagel-LatentCoT
export PYTHON_BIN=/private/software/conda/envs/lcot/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
export DATA_PATH=/private/yida_workspace/datasets/memory_grounding_v2/stage12_stage2_prompt_unique_20260927/prompts.jsonl
export OUTDIR="/private/yida_workspace/outputs/memory_reader_v2_8gpu_$(date +%Y%m%d_%H%M%S)"
export NUM_PROCESSES=8 MAX_STEPS=5000 BATCH_SIZE=2
bash scripts/train/run_memory_reader_8gpu.sh
```

## Fixed protocol

- Stage A trains GEN-Q only; K=8, body=[12,20), one Read and one Write,
  fresh memory, same-depth, CFG=1. Learning rate stays 5e-6 (no linear scaling).
- Batch size 2 is **per rank**; global batch is 16. Gradients are averaged across
  ranks before clipping and AdamW. There is no gradient accumulation.
- Correct/shuffled/zero counterfactuals are evaluated within each rank: shuffled
  memory comes from the other local prompt, not from a global 16-way permutation.
  The optimized loss is the mean of eight local paired objectives.
- Each native rollout supplies 3 states reused for 3 optimizer updates. All ranks
  share the state-selection seed/timestep; prompt noise is rank/sample-specific.
  Deterministic epoch shuffle uses seed+epoch, with cyclic padding in the last
  global batch. No duplicates occur within a global batch; padded samples can
  recur across batches. 5,000 updates consume 1,667 global prompt batches (26,672
  prompt draws), not a complete pass through all 115,883 unique prompts.
- Rank zero alone writes checkpoints every 100 updates and at the last update.
  It writes global scalar means plus full `rank_metrics` (sample/donor provenance).
  Relative delta-v is a mean of rank-local relative norms, not one global norm.

## Outputs and checks

`$OUTDIR` contains `resolved_config.json`, `run_manifest.json`,
`trainable_routes.json`, `metrics.jsonl`, checkpoints and final `status.json`.
`${OUTDIR}.launcher.log` is the combined console log; `${OUTDIR}.workers` holds
individual rank stdout/stderr. The launcher refuses to overwrite these paths.
`--adapter-path` remains a weight warm-start, not exact optimizer/data resume.

```bash
tail -n 40 "${OUTDIR}.launcher.log"
```

For configuration/data-only validation, use a fresh OUTDIR and append
`--validate-only` to the launcher. It checks GPU visibility, then runs preflight
in all eight workers, without loading BAGEL weights or initializing NCCL.
It does **not** prove model-file integrity, free memory, NCCL health or convergence.

CPU tests cover actual tiny BAGEL multi-process backpropagation, averaged Adam
updates versus the corresponding mean-loss reference, disjoint batches, shared
artifact writes and propagated failures. Real H200/NCCL training is deliberately
left to the user. See [validation record](VALIDATION.md) for completed checks.
