# NPU run commands — Write Sensitivity T2I

Run from the repository's NPU checkout. This experiment is **not yet run**;
its four-arm outcomes are unknown. The default is a frozen/training-free loop,
matching the first-round scope of the supplied plan. The existing early-body
training YAML supplies K=8, R=2, `[12,20)`, LoRA shape, and strict Read policy.
All LoRA B residuals are zero in the default run. The optional `--adapter`
loads a matching checkpoint without changing the four-arm protocol.

## 1. Update the calibrated checkout

The other NPU checkouts have local edits; use this separate checkout and do
not delete prior outputs.

```bash
git -C /root/bagel-LatentCoT-phase1-calibrated fetch origin main
git -C /root/bagel-LatentCoT-phase1-calibrated switch --detach origin/main
cd /root/bagel-LatentCoT-phase1-calibrated
```

## 2. Two-prompt / two-step smoke (optional; separate output)

```bash
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -m pytest -q \
  tests/test_bagel_write_sensitivity_t2i.py tests/test_mot_loop_phase0.py

PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -u \
  scripts/evaluate/bagel_write_sensitivity_t2i.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --output-dir /data/outputs/bagel_write_sensitivity_hard16_pair2_smoke \
  --device npu:0 --max-prompts 2 --num-steps 2
```

The smoke must produce 8 PNGs, 4 pair probe files, and
`run_manifest.json` with `complete=true`. It does not establish sensitivity.

## 3. Full 16-prompt run

```bash
PYTHONPATH="$PWD" /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python -u \
  scripts/evaluate/bagel_write_sensitivity_t2i.py \
  --training-config configs/training/loop_pair_memory_early.yaml \
  --output-dir /data/outputs/bagel_write_sensitivity_hard16_pair2 \
  --device npu:0
```

The full run is eight native packed batches of two prompts, four arms per
batch, 50 generation schedule points, 512×512, CFG text/image 4/1. The shuffle
is a deterministic swap **within each pair**, with no identity mapping. It
does not alter prompt/noise/CFG/schedule/K/R/body. The intervention is made
after strict Read and before the sole Write round in each denoising step.
The same source rule is applied to conditional and CFG branches, while only
conditional Read memory is probed. `sample_global` applies the native global
CFG norm formula separately to each sample (equal to `global` for batch size
1), avoiding a second cross-sample path. This is a new paired-batch protocol;
do not compare absolute pixels to the earlier single-sample hard-16 run.

To repeat with Phase 1.1 UND-Q weights, use a **new output directory** and add:

```text
--adapter /data/outputs/bagel_pair_memory_calibrated_f783989/pair_memory_adapter_step_0001000.safetensors
```

Open `index.html` for the four-column gallery. `run_manifest.json` records
pair/donor indexes, noise hashes, checkpoint/config/benchmark hashes, probe
summaries, and per-prompt pixel MAE versus `correct_M`. Raw per-step memory
probes are under `pair_*/`. GenEval2 image maps are under `geneval2/` for
optional later scoring. Pixel MAE measures difference, not quality.
