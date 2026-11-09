"""Measure the inversion round-trip floor before producing an I cache.

The source and target images come from an external model, so a round trip is
not expected to be exact: this script reports the achievable floor per solver
subdivision so the acceptance thresholds can be set from data instead of hope.
"""

import argparse
import json

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import LinearManifest
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, forward_native, phase_grid, preview_image, reverse_heun_states, time_key,
)
from qwen_latent_cot.u15_target_correction.images import load_pair
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.loss import build_perceptual, reconstruction_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--linear", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--substeps", default="1,2,4,8")
    parser.add_argument("--endpoints", default="full", help='"full" or times such as 1.0,0.95')
    parser.add_argument("--time", type=float, default=0.36)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    manifest = LinearManifest(args.linear, rt, cfg)
    grid = phase_grid(rt, cfg)
    perceptual = build_perceptual(device)
    times = [float(args.time)]
    rows = [manifest.rows[index] for index in manifest.train_indices[: args.limit]]
    for row in rows:
        source, target, valid, rect = load_pair(row, rt.width, rt.height, rt.device)
        parent = NativeT2I(rt, row["parent_condition"], cfg)
        tail = NativeT2I(rt, row["tail_condition"], cfg)
        prior = None
        endpoints = [None if value == "full" else float(value) for value in args.endpoints.split(",")]
        for substeps in [int(value) for value in args.substeps.split(",")]:
          for endpoint in endpoints:
            anchor = reverse_heun_states(
                source, 1.0, times, grid, parent, substeps, endpoint=endpoint
            )[time_key(args.time)]
            destination = reverse_heun_states(
                target, 1.0, times, grid, tail, substeps, endpoint=endpoint
            )[time_key(args.time)]
            source_rec = forward_native(anchor, args.time, grid, parent)
            target_rec = forward_native(destination, args.time, grid, tail)
            preview = preview_image(anchor, args.time, grid, parent)
            metrics = dict(
                uid=row["uid"],
                task=row.get("task"),
                substeps=substeps,
                endpoint=endpoint,
                valid_px=float(valid.sum()),
                canvas=[source.shape[-1], source.shape[-2]],
                source=reconstruction_metrics(source_rec, source, valid, perceptual),
                target=reconstruction_metrics(target_rec, target, valid, perceptual),
                state_a_abs_mean=float(anchor.abs().mean()),
                state_b_minus_a_abs_mean=float((destination - anchor).abs().mean()),
                preview_abs_mean=float(preview.abs().mean()),
                anchor_shift_from_prior=None if prior is None else float((anchor - prior).abs().mean()),
            )
            prior = anchor
            print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
