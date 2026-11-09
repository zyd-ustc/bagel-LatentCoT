"""Where does the texture energy come from: the teacher's own path or the handoff?

For each case this compares three images measured the same way:

``x_src``     the source trajectory's own final image
``x_teacher`` the image-conditioned teacher continued to t=1 with *its own*
              condition (no handoff at all)
``x_eff``     the teacher state at the read time continued by the text-conditioned
              deployment tail (the label we actually build)

If ``x_teacher`` stays near ``x_src`` in high-frequency energy while ``x_eff``
jumps, the cost sits in the IT2I -> T2I handoff, not in the teacher's samples.
"""

import argparse
import json
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.coupled import load_source, run_trajectory
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, euler, phase_grid, segment_times, time_key,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.labels import high_frequency
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.teachers import NativeIT2I


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--teacher-cfg", default="4", help="Comma separated CFG scales for the teacher")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    grid = phase_grid(rt, cfg)
    read = float(cfg["read_time"])
    key = time_key(read)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(v) for v in args.teacher_cfg.split(",")]
    for index, request in enumerate(rows[: args.limit]):
        uid = str(request["uid"])
        source = load_source(args.source_cache, uid, key)
        if source is None:
            continue
        seed = source["seed"]
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
        parent = (request["parent_condition"] or "").strip()
        tail = (request["tail_condition"] or "").strip()
        src = run_trajectory(rt, cfg, parent, noise, grid, [read], read)
        hf_src = high_frequency(src["final"])
        for scale in scales:
            trial = dict(cfg, cfg_scale=scale)
            teacher_state = euler(noise.float(), segment_times(grid, 0.0, read),
                                  NativeIT2I(rt, tail, src["final"], trial))
            eff = euler(teacher_state.float(), segment_times(grid, read, 1.0), NativeT2I(rt, tail, cfg))
            row = dict(uid=uid, teacher_cfg=scale, hf_src=hf_src,
                       hf_eff=high_frequency(eff))
            row["ratio_eff"] = row["hf_eff"] / max(hf_src, 1e-6)
            print(json.dumps(row), flush=True)
            save_image(out / f"{index:03d}_cfg{int(scale)}.png", eff)


if __name__ == "__main__":
    main()
