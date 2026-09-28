#!/usr/bin/env python3
"""Normal-only R=2/4/6/8 ablation, same-R2-state probes + independent images."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from bagel_common import autocast_for, load_native_bagel, make_noise, pixel_mae, stable_noise_seed
from mechanism_runtime import prepare_runtime
from qwen_latent_cot.bagel.normal_r_inference import (
    NormalRoundEngine, MODES, ROUNDS, REFERENCE, PAIRS,
    generate_trajectory, probe_reference_trajectory,
)

SCHEMA = "bagel-normal-r-ablation-v1"
# Preserve prompt noise from the completed hard64 run, independently of schema.
NOISE_SCHEMA = "bagel-memory-mechanism-v1"
ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", str(ROOT / "models/Bagel-7B-MoT")))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-data", type=Path, default=ROOT / "experiments/data/geneval2_hard_128.jsonl")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default, H200) / npu:0 / auto; explicit backends never fall back")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=50, help="Native schedule points; denoising NFE = num-steps - 1")
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Shard complete prompt PAIRS, not individual prompts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args(argv)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_prompts(path, limit):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [str(json.loads(line)["prompt"]) if path.suffix == ".jsonl" else line.strip() for line in lines]
    if limit < 2 or limit % 2 or limit > len(prompts):
        raise ValueError("--max-prompts must be even, >=2, and not exceed the input dataset")
    prompts = prompts[:limit]
    if len(set(prompts)) != len(prompts) or any(not p.strip() for p in prompts):
        raise ValueError("selected prompts must be nonempty and unique")
    return prompts


def make_plan(args):
    prompts = load_prompts(args.benchmark_data, args.max_prompts)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 for s in seeds):
        raise ValueError("seeds must be unique integers in [0, 2**63)")
    if not 1 <= args.num_shards <= len(prompts) // 2 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shards must partition complete pairs, with at least one pair per shard")
    if min(args.height, args.width) <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("image dimensions must be positive multiples of 16")
    if args.num_steps < 2 or not 0 < args.timestep_shift < float("inf"):
        raise ValueError("invalid native image schedule")
    sources = ["scripts/evaluate/bagel_memory_mechanism.py", "scripts/evaluate/bagel_common.py",
               "scripts/evaluate/mechanism_runtime.py",
               "qwen_latent_cot/bagel/normal_r_inference.py",
               "qwen_latent_cot/bagel/memory_mechanism.py", "qwen_latent_cot/bagel/mechanism_inference.py",
               "qwen_latent_cot/bagel/modeling/bagel/qwen2_navit.py",
               "qwen_latent_cot/bagel/modeling/bagel/bagel.py", "qwen_latent_cot/bagel/backbone.py"]
    return {
        "schema": SCHEMA, "prompts": prompts, "seeds": seeds,
        "benchmark_sha256": sha256(args.benchmark_data),
        "source_sha256": {p: sha256(ROOT / p) for p in sources},
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "image_shape": [args.height, args.width], "num_schedule_points": args.num_steps,
        "denoising_nfe": args.num_steps - 1, "timestep_shift": args.timestep_shift,
        "arms": list(MODES), "mode": "normal", "K": 8, "R_values": list(ROUNDS),
        "body": [12, 20], "persist": False, "reference_arm": REFERENCE,
        "round_semantics": "one_read_then_R_minus_1_writes_reset_nonmemory_each_write",
        "noise_schema": NOISE_SCHEMA,
        "prompt_kv": "visible", "initialization": "native_frozen_soi_eoi",
        "CFG": {"text": 4.0, "image": 1.0, "interval": [0.4, 1.0], "renorm": "sample_global", "min": 0.0},
        "hidden_branch": "conditional", "relative_denominator": "normal_r2_velocity_norm_plus_1e-12",
        "attention_layers": [12, 19], "attention_reduction": "exact_all_queries_and_heads_chunked",
        "num_shards": args.num_shards,
        "pairs": [{"pair_id": i // 2, "prompt_indexes": [i, i + 1], "shard": (i // 2) % args.num_shards}
                  for i in range(0, len(prompts), 2)],
    }


def write_curve(path, rows):
    width, height = 700, 260
    ceiling = max([r[f"relative_{name}"] for r in rows for name in PAIRS] + [1e-12])
    colors = ("#577a89", "#a27850", "#6c885f", "#8b6f96", "#b2686b")
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="45" y="18" font-size="12">Same R2 x_t; relative Δv / ||v_R2||; max={ceiling:.3g}</text>',
              '<path d="M45 32V218H680" fill="none" stroke="#999"/>']
    for idx, (name, color) in enumerate(zip(PAIRS, colors)):
        points = " ".join(f'{45 + 635 * i / max(len(rows)-1, 1):.2f},{218 - 175 * row[f"relative_{name}"] / ceiling:.2f}'
                          for i, row in enumerate(rows))
        pieces += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                   f'<text x="{45 + idx * 125}" y="253" fill="{color}" font-size="12">{name}</text>']
    pieces += [f'<text x="45" y="234" font-size="11">step 0 / t={rows[0]["t"]:.3g}</text>',
               f'<text x="510" y="234" font-size="11">step {rows[-1]["step"]} / t={rows[-1]["t"]:.3g}</text>', '</svg>']
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_gallery(out, rows):
    parts = ['<!doctype html><meta charset="utf-8"><title>Normal memory — R ablation</title>',
             '<style>body{font:14px system-ui}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}img{width:210px}td{vertical-align:top}</style>',
             '<h1>Normal memory: R=2,4,6,8</h1><p>R = one Read + R−1 Writes. Same initial noise. Pixel MAE vs R2 is behavioral distance, not image quality.</p>',
             '<table><tr><th>Prompt / seed / trace</th>' + ''.join(f'<th>{m}</th>' for m in MODES) + '</tr>']
    for row in sorted(rows, key=lambda r: (r["seed"], r["prompt_index"])):
        parts.append(f'<tr><td>{html.escape(row["prompt"])}<br>seed={row["seed"]}<br><a href="{html.escape(row["curve"])}">timestep curve</a><br><a href="{html.escape(row["trace"])}">metrics</a></td>')
        for mode in MODES:
            parts.append(f'<td><img src="{html.escape(row["images"][mode])}"><br>MAE vs R2={row["pixel_mae_vs_r2"][mode]:.3f}</td>')
        parts.append('</tr>')
    parts.append('</table>')
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def merge(out, plan):
    root_manifest = out / "run_manifest.json"
    if root_manifest.exists() and json.loads(root_manifest.read_text())["plan"]["schema"] != SCHEMA:
        raise ValueError("cannot overwrite a previous experiment schema during merge")
    write_json(out / "run_manifest.json", {"plan": plan, "complete": False, "status": "validating_shards"})
    manifests = [json.loads((out / f"shard_{i:03d}/run_manifest.json").read_text()) for i in range(plan["num_shards"])]
    expected = {(seed, p) for seed in plan["seeds"] for p in range(len(plan["prompts"]))}
    rows, seen = [], set()
    for shard_id, manifest in enumerate(manifests):
        if not manifest["complete"] or manifest["plan"] != plan or manifest["shard_id"] != shard_id:
            raise ValueError("cannot merge incomplete, mismatched or stale shards")
        for row in manifest["rows"]:
            key = (row["seed"], row["prompt_index"])
            if key in seen or key not in expected or row["pair_id"] % plan["num_shards"] != shard_id:
                raise ValueError("duplicate or misplaced prompt/seed result")
            if (row["prompt"] != plan["prompts"][row["prompt_index"]]
                    or row["pair_id"] != row["prompt_index"] // 2
                    or row.get("reference_arm") != REFERENCE
                    or set(row.get("pixel_mae_vs_r2", {})) != set(MODES)
                    or set(row["images"]) != set(MODES)):
                raise ValueError("row identity or arm set disagrees with protocol")
            for file in list(row["images"].values()) + [row["curve"], row["trace"]]:
                if not (out / file).is_file():
                    raise FileNotFoundError(out / file)
            trace = [json.loads(line) for line in (out / row["trace"]).read_text().splitlines()]
            if len(trace) != plan["denoising_nfe"] or [r["step"] for r in trace] != list(range(plan["denoising_nfe"])):
                raise ValueError("missing or duplicated timestep metrics")
            for record in trace:
                if record.get("reference_arm") != REFERENCE or set(record.get("arms", {})) != set(MODES):
                    raise ValueError("trace reference or R arm set disagrees with protocol")
                for name in PAIRS:
                    if not math.isfinite(record[f"relative_{name}"]):
                        raise ValueError("nonfinite mechanism trace")
            if any(not math.isfinite(value) for value in row["pixel_mae_vs_r2"].values()):
                raise ValueError("nonfinite image distance")
            pair_dir = (out / row["trace"]).parent
            for step in range(plan["denoising_nfe"]):
                state = pair_dir / "reference_states" / f"step_{step:03d}.pt"
                if not state.is_file():
                    raise FileNotFoundError(state)
            for name in ("attention.jsonl", "memory_checks.jsonl"):
                if not (pair_dir / name).is_file():
                    raise FileNotFoundError(pair_dir / name)
            seen.add(key)
            rows.append(row)
    if seen != expected:
        raise ValueError("missing prompt/seed results")
    write_json(out / "run_manifest.json", {"plan": plan, "complete": True, "rows": rows})
    write_gallery(out, rows)


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    out = args.output_dir.expanduser().resolve()
    if args.merge_only:
        merge(out, plan)
        print(f"[done] {out / 'index.html'}")
        return
    shard = out / f"shard_{args.shard_id:03d}"
    if shard.exists():
        raise FileExistsError(f"refusing to overwrite shard: {shard}")
    shard.mkdir(parents=True)
    manifest = {"plan": plan, "shard_id": args.shard_id, "device": args.device,
                "complete": False, "rows": [], "status": "loading"}
    manifest_path = shard / "run_manifest.json"
    write_json(manifest_path, manifest)
    try:
        manifest["runtime"] = prepare_runtime(args.device)
        manifest["device"] = manifest["runtime"]["device"]
        native_args = SimpleNamespace(
            model_path=plan["model_path"], device=manifest["device"], seed=plan["seeds"][0],
            num_loop_tokens=8, loop_depth=2, loop_recycle_mode="same_depth",
            loop_memory_persist=False, memory_loop_start_layer=12, memory_loop_end_layer=20,
            round0_memory_write_enabled=False,
            vit_max_image_size=980, vit_min_image_size=224, vit_image_stride=14,
        )
        backbone, inferencer = load_native_bagel(native_args)
        manifest["torch_version"] = str(torch.__version__)
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        for seed in plan["seeds"]:
            for pair in plan["pairs"]:
                if pair["shard"] != args.shard_id:
                    continue
                indexes = pair["prompt_indexes"]
                prompts = [plan["prompts"][idx] for idx in indexes]
                noise_seeds = [stable_noise_seed(seed, p, schema=NOISE_SCHEMA) for p in prompts]
                noises = [make_noise(backbone.bagel, tuple(plan["image_shape"]), s) for s in noise_seeds]
                pair_dir = shard / f"seed_{seed}" / f"pair_{pair['pair_id']:03d}"
                states_dir = pair_dir / "reference_states"
                states_dir.mkdir(parents=True)
                traces = [[], []]
                with torch.inference_mode(), autocast_for(inferencer.device):
                    engine = NormalRoundEngine(inferencer, prompts, noises, tuple(plan["image_shape"]))
                    timesteps, dts = engine.model.prepare_image_schedule(args.num_steps, args.timestep_shift, engine.device)
                    def on_step(step, state, t, rows, diagnostics):
                        torch.save({"x_t": state, "t": t, "step": step, "prompt_indexes": indexes,
                                    "reference_arm": REFERENCE}, states_dir / f"step_{step:03d}.pt")
                        for sample, row in enumerate(rows):
                            traces[sample].append(row)
                            with (pair_dir / f"p{indexes[sample]:03d}_trace.jsonl").open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row, allow_nan=False) + "\n")
                        with (pair_dir / "attention.jsonl").open("a", encoding="utf-8") as handle:
                            for mode, diag in diagnostics.items():
                                for record in diag["attention"]:
                                    handle.write(json.dumps({"step": step, "t": t, "arm": mode, **record}, allow_nan=False) + "\n")
                        with (pair_dir / "memory_checks.jsonl").open("a", encoding="utf-8") as handle:
                            for mode, diag in diagnostics.items():
                                handle.write(json.dumps({"step": step, "arm": mode,
                                                         "hidden_max_by_layer": diag.get("memory_hidden_max", [])}, allow_nan=False) + "\n")
                        print(f"[probe] seed={seed} pair={pair['pair_id']} step={step+1}/{len(timesteps)}", flush=True)
                    reference_final = probe_reference_trajectory(engine, timesteps, dts, on_step)
                    images = [{}, {}]
                    for mode in MODES:
                        final = (reference_final if mode == REFERENCE else
                                 generate_trajectory(engine, timesteps, dts, mode))
                        for sample, latent in enumerate(final.split(engine.lengths)):
                            image = inferencer.decode_image(latent, tuple(plan["image_shape"]))
                            image.save(pair_dir / f"p{indexes[sample]:03d}_{mode}.png")
                            images[sample][mode] = image
                        print(f"[images] seed={seed} pair={pair['pair_id']} arm={mode}", flush=True)
                for sample, idx in enumerate(indexes):
                    curve = pair_dir / f"p{idx:03d}_timestep.svg"
                    write_curve(curve, traces[sample])
                    manifest["rows"].append({
                        "prompt_index": idx, "prompt": prompts[sample], "seed": seed,
                        "pair_id": pair["pair_id"], "reference_arm": REFERENCE,
                        "noise_seed": noise_seeds[sample],
                        "noise_sha256": hashlib.sha256(noises[sample].contiguous().numpy().tobytes()).hexdigest(),
                        "curve": str(curve.relative_to(out)),
                        "trace": str((pair_dir / f"p{idx:03d}_trace.jsonl").relative_to(out)),
                        "images": {m: str((pair_dir / f"p{idx:03d}_{m}.png").relative_to(out)) for m in MODES},
                        "pixel_mae_vs_r2": {m: pixel_mae(images[sample][m], images[sample][REFERENCE]) for m in MODES},
                    })
                write_json(manifest_path, manifest)
                del engine
        manifest.update(complete=True, status="complete")
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise
    if args.num_shards == 1:
        merge(out, plan)
    print(f"[done] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
