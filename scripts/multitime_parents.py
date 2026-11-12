"""Render the parent trajectory at several read times for time-general training.

The editor takes the generation time as an input, but an input that exists is not
an ability: a model trained at one time has only seen one kind of starting state.
This script keeps the expensive part shared and repeats only what depends on the
time:

    one native forward  -> snapshot at t in {0.25, 0.36, 0.50}
                        -> each snapshot carries its OWN preview P_t

The completion ``x_src`` and the edit instruction are shared: they describe the
scene, not the moment. Only the student-side state and the preview change.

Usage::

    python scripts/multitime_parents.py --prompts <jsonl> --out <cache> \\
        --times 0.25,0.36,0.50 --limit 50000 --shard-index 0 --shard-count 16
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, native_forward_states, phase_grid, time_key,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.prepare_state_pairs import _safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True,
                        help="jsonl with uid and prompt (or raw_prompt)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--times", help="Comma-separated grid nodes; defaults to state_times")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    default_times = cfg["state_times"] if cfg.get("semantic_weights") else [0.25, 0.36, 0.50]
    times = [float(t) for t in args.times.split(",")] if args.times else default_times
    grid = phase_grid(rt, cfg)
    if any(time not in grid for time in times):
        raise ValueError("Every requested parent state must be a grid node")

    rows = []
    with open(args.prompts) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("prompt", row.get("raw_prompt"))
            if row.get("prompt"):
                rows.append(row)
            if len(rows) >= args.limit:
                break
    rows.sort(key=lambda r: hashlib.sha1(("mt:" + str(r["uid"])).encode()).hexdigest())

    out = Path(args.out).resolve()
    (out / "progress").mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    name = "manifest.json" if args.shard_count <= 1 else f"manifest.shard_{args.shard_index:03d}.json"
    records, written = [], 0

    for position, row in enumerate(rows):
        if position % args.shard_count != args.shard_index:
            continue
        uid = str(row["uid"])
        # Same naming as prepare_state_pairs so coupled.load_source finds it.
        marker = out / "progress" / f"{_safe(uid)}.json"
        if args.skip_existing and marker.exists():
            cached = json.loads(marker.read_text())
            if cached.get("phase_grid") == grid and all(
                cached["states"].get(time_key(t), {}).get("preview_image") for t in times
            ):
                records.append(cached)
                continue
        began = time.perf_counter()
        seed = int(row.get("seed", int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)))
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                            generator=generator) * rt.noise_scale()
        velocity = NativeT2I(rt, row["prompt"], cfg)
        walked = native_forward_states(noise, grid, times, velocity)

        stem = _safe(uid)
        states = {}
        for t in times:
            key = time_key(t)
            if key not in walked["states"]:
                continue
            payload = dict(
                time=t,
                state_a=walked["states"][key].cpu(),
                preview=walked["previews"][key].cpu(),
                valid=torch.ones(1, 1, rt.height, rt.width),
            )
            path = out / "states" / f"{stem}_{key}.pt"
            torch.save(payload, path)
            preview_path = out / "targets" / f"{stem}_{key}_preview.png"
            save_image(preview_path, walked["previews"][key])
            states[key] = dict(path=str(path), preview_image=str(preview_path), accepted=True)
        completion = out / "targets" / f"{stem}_source_final.png"
        save_image(completion, walked["final"])

        prompt_family = str(row.get("prompt_family") or row.get("family") or row["prompt"])
        split_bucket = int(hashlib.sha256(prompt_family.encode()).hexdigest()[:8], 16) % 100
        default_split = "val" if cfg.get("semantic_weights") and split_bucket < 5 else "train"
        record = dict(
            uid=uid,
            source_kind="native_parent",
            split=row.get("split", default_split),
            role="R_active",
            parent_condition=row["prompt"],
            tail_condition=row["prompt"],
            edit_instruction="",
            source_image=str(completion),
            source_family=row.get("family") or hashlib.sha1(uid.encode()).hexdigest()[:12],
            prompt_family=prompt_family,
            seed=seed,
            read_times=times,
            phase_grid=grid,
            states=states,
            seconds=round(time.perf_counter() - began, 2),
        )
        marker.write_text(json.dumps(record, ensure_ascii=False))
        records.append(record)
        written += 1

    header = dict(kind="multi_time_parents", times=times, count=len(records))
    (out / name).write_text(json.dumps(dict(header, records=records), ensure_ascii=False))
    print(json.dumps(dict(shard=args.shard_index, written=written, total=len(records),
                          times=times, path=str(out / name))))


if __name__ == "__main__":
    main()
