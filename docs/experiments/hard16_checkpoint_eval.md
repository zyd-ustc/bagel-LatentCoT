# Unified BAGEL hard-16 checkpoint image evaluation

Use `scripts/evaluate/bagel_hard16_checkpoint.py` for Phase 1.1, 1.2, 1.3,
and 2. Pass the **training YAML belonging to that checkpoint** and the
checkpoint `.safetensors` file. The script reads K, generation loop depth,
recycle mode, persistence, `[start,end)` body, round-0 policy, and LoRA
shape/routes from the YAML. It checks any adjacent JSON checkpoint metadata
against that contract. There is no loop-position ablation in this protocol.

For the completed Phase 1.1 checkpoint, run from the repository root on NPU:

```bash
PYTHON_BIN=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python
PYTHONPATH="$PWD" "$PYTHON_BIN" scripts/evaluate/bagel_hard16_checkpoint.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --adapter /data/outputs/bagel_pair_memory_calibrated_f783989/pair_memory_adapter_step_0001000.safetensors \
  --output-dir /data/outputs/bagel_hard16_phase11_step1000 \
  --device npu:0
```

For a one-prompt connectivity pilot, add `--max-prompts 1` and use a **different,
empty** output directory. For later phases, change only `--training-config`,
`--adapter`, and `--output-dir`; pass `--model-path` only if the model moved.
The default generation protocol is 512×512, 50 steps, timestep shift 3,
text/image CFG 4/1, CFG interval `[0.4,1]`, and seed 42. Overrides are
recorded in `run_manifest.json`; hold them fixed when comparing phases.

`base` disables the loop. `training_free_loop` and `trained_loop` use the
**same** K, depth, body, round policy, m0, CFG, and injected LoRA modules.
Training-free sets LoRA B weights to zero; trained loads the checkpoint.
Both arms use the same per-prompt initial noise. Phase 1.1 only trained UND-Q
Read; its missing GEN-Q Write residual remains zero in the trained arm.
Its sidecar `R=1` is the number of Read rounds, while the training YAML's
`loop_depth=2` controls image generation. A future checkpoint with missing
routes is rejected unless it carries that Phase 1.1 schema.

Open `index.html` for a side-by-side gallery. `run_manifest.json` records
the model, checkpoint/config/benchmark hashes, contract, per-prompt noise
seed/hash, and generation settings. `geneval2/{arm}_image_paths.json` contains image maps for the
existing GenEval2 scorer. Example, after starting the scoring server:

```bash
PYTHON_BIN=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python
for arm in base training_free_loop trained_loop; do
  "$PYTHON_BIN" scripts/evaluate/score_geneval2_server.py \
    --benchmark-data /data/outputs/bagel_hard16_phase11_step1000/benchmark_hard.jsonl \
    --image-paths "/data/outputs/bagel_hard16_phase11_step1000/geneval2/${arm}_image_paths.json" \
    --output "/data/outputs/bagel_hard16_phase11_step1000/geneval2/${arm}_scores.json"
done
```

This is a text-to-image image-quality/compositionality comparison. Phase 1.1
was trained on paired edit memory, so hard-16 text-to-image output alone does
not prove the Phase 1.1 held-out pair-grounding Go condition.
