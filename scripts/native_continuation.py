"""Continue a dev prompt's parent trajectory without any edit.

The sampler's passthrough route integrates the same noise over the same phase
grid; the frozen ``t2i_generate`` entry uses its own solver details and lands on
a slightly different image (L1 up to 0.12). This script reproduces the
passthrough exactly so an edited result can be compared with the trajectory it
came from.
"""

import argparse
import json
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import read_rows
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, euler, phase_grid, segment_times,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.loading import load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", help="Comma separated row indices; all rows when omitted")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = None if not args.indices else {int(v) for v in args.indices.split(",")}
    grid = phase_grid(rt, cfg)
    read = float(cfg["read_time"])
    rows = read_rows(args.prompts)
    for index, row in enumerate(rows):
        if wanted is not None and index not in wanted:
            continue
        raw = row.get("raw_prompt") or row.get("prompt")
        seed = int(row.get("seed", cfg["seed"] + index))
        native = NativeT2I(rt, raw, cfg)
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        noise = torch.randn(
            (1, 3, rt.height, rt.width), device=rt.device, generator=generator
        ) * rt.noise_scale()
        state = euler(noise.float(), segment_times(grid, 0.0, read), native)
        final = euler(state, segment_times(grid, read, 1.0), native)
        save_image(out / f"{index:05d}.png", final)
        (out / f"{index:05d}.json").write_text(json.dumps(dict(
            index=index, uid=row.get("uid"), seed=seed, raw_prompt=raw,
            route="native_continuation_same_trajectory",
        ), ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(dict(index=index, seed=seed, uid=row.get("uid"))), flush=True)


if __name__ == "__main__":
    main()
