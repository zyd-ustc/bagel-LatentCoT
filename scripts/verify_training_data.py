"""Pre-training data gate: load the configured caches and sample real pairs.

Catches protocol drift, empty role pools and malformed payloads before a training
run starts. It reads exactly what the trainer would read, with the real runtime so
the bound protocol must match.
"""

import argparse
import json

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import (
    LinearManifest, StatePairManifest, TrainingSampler,
)
from qwen_latent_cot.u15_target_correction.loading import load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--linear", help="Linear pair cache; omit to gate the R pools only")
    parser.add_argument("--native", required=True, help="Cache holding R-edit and R-keep records")
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    linear = LinearManifest(args.linear, rt, cfg) if args.linear else None
    native = StatePairManifest(args.native, rt, cfg, "native_parent")
    gate_cfg = cfg
    if linear is None:
        # The linear pool lives outside this cache family; gate the R pools alone.
        roles = dict(cfg["role_weights"])
        roles["R_keep"] = roles.get("R_keep", 0) + roles.pop("L_edit", 0)
        gate_cfg = dict(cfg, role_weights=roles)
    sampler = TrainingSampler(linear, None, native, gate_cfg)
    print(json.dumps(dict(linear=linear.count if linear else None, native=native.count,
                          roles={role: getattr(pool, "count", None) for role, pool in sampler.pools.items()},
                          weights=sampler.weights)))
    for _ in range(args.draws):
        sample = sampler.draw()
        row = dict(
            role=sample["role"],
            a=tuple(sample["state_a"].shape), b=tuple(sample["state_b"].shape),
            preview=tuple(sample["preview"].shape), valid=tuple(sample["valid"].shape),
            finite=bool(torch.isfinite(sample["state_a"]).all()
                        and torch.isfinite(sample["state_b"]).all()),
            displacement=float((sample["state_b"] - sample["state_a"]).abs().mean()),
        )
        print(json.dumps(row))


if __name__ == "__main__":
    main()
