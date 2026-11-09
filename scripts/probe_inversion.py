"""Probe how far the discrete inverse can be solved on real cached pairs.

The residual after a finite Picard budget can mean either "more iterations would
converge" or "the image is not a preimage of this map". This script sweeps the
iteration budget on one side of one pair and reports both the solve residual and
the actual forward round trip, so the two can be told apart before a production
run commits to a budget.
"""

import argparse
import json

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import LinearManifest
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, discrete_inverse_states, forward_native, phase_grid, time_key,
)
from qwen_latent_cot.u15_target_correction.images import load_pair
from qwen_latent_cot.u15_target_correction.loading import load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--linear", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--iterations", default="4,8,16,32")
    parser.add_argument("--damping", default="0.6")
    parser.add_argument("--time", type=float, default=0.36)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    manifest = LinearManifest(args.linear, rt, cfg)
    grid = phase_grid(rt, cfg)
    times = [float(args.time)]
    masks = [float(v) for v in args.damping.split(",")]
    rows = [manifest.rows[index] for index in manifest.train_indices[: args.limit]]
    for row in rows:
        source, target, valid, _ = load_pair(row, rt.width, rt.height, rt.device)
        parent = NativeT2I(rt, row["parent_condition"], cfg)
        for damping in masks:
            for iterations in [int(value) for value in args.iterations.split(",")]:
                snapshots, diagnostics = discrete_inverse_states(
                    source, grid, times, parent, iterations=iterations, damping=damping
                )
                state = snapshots[time_key(args.time)]
                back = forward_native(state, args.time, grid, parent)
                error = float((back - source).abs().mean())
                info = diagnostics[time_key(args.time)]
                print(json.dumps(dict(
                    uid=row["uid"][-10:], damping=damping, iterations=iterations,
                    own_residual=info["residual"], segment_residual=info["segment_residual"],
                    used=info["iterations"], round_trip_l1=error,
                    state_abs_mean=float(state.abs().mean()),
                ), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
