#!/usr/bin/env python3
"""Four hard prompts: one-pass K=8 memory, with/without body prompt-KV mask."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from bagel_common import autocast_for, load_native_bagel, make_noise, pixel_mae, stable_noise_seed
from bagel_hard16_checkpoint import load_benchmark, resolve_contract, sha256


NOISE_SCHEMA = "bagel-write-sensitivity-t2i-v1"
POLICIES = ("keep", "mask_body")
MEMORY_UPDATE_END = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-data", type=Path, default=Path("experiments/data/geneval2_hard_16.jsonl"))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_protocol(contract: dict, args) -> None:
    if (
        contract["num_loop_tokens"] != 8
        or contract["loop_recycle_mode"] != "same_depth"
        or contract["loop_memory_persist"]
        or not 0 < contract["memory_loop_start_layer"] < MEMORY_UPDATE_END
        or not MEMORY_UPDATE_END < contract["memory_loop_end_layer"]
    ):
        raise ValueError("requires K=8, fresh memory, and a mask body overlapping memory update end=16")
    if not 1 <= args.max_prompts <= 16 or args.height <= 0 or args.width <= 0:
        raise ValueError("requires 1..16 prompts and positive image dimensions")
    if args.num_steps < 2 or args.timestep_shift <= 0:
        raise ValueError("requires at least two schedule points and a positive shift")


def generate(inferencer, prompt: str, noise: torch.Tensor, args, *, mask_body: bool):
    with torch.inference_mode(), autocast_for(getattr(inferencer, "device", "cpu")):
        return inferencer(
            image=None,
            text=prompt,
            image_shapes=(args.height, args.width),
            init_noise=noise.clone(),
            num_timesteps=args.num_steps,
            timestep_shift=args.timestep_shift,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
            cfg_renorm_min=0.0,
            cfg_renorm_type="sample_global",
            enable_taylorseer=False,
            return_loop_diagnostics=False,
            single_pass_memory_update_end=MEMORY_UPDATE_END,
            single_pass_mask_prompt_kv=mask_body,
        )["image"]


def main() -> None:
    args = parse_args()
    config_path = args.training_config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    contract = resolve_contract(config, {})
    validate_protocol(contract, args)
    benchmark_path = args.benchmark_data.expanduser().resolve()
    rows = load_benchmark(benchmark_path, args.max_prompts)
    model_path = Path(args.model_path or config["model_path"]).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.dry_run:
        print(json.dumps({"prompts": len(rows), "memory_update_layers": [0, MEMORY_UPDATE_END],
                          "mask_layers": [contract["memory_loop_start_layer"], contract["memory_loop_end_layer"]]}, indent=2))
        return

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    native_args = SimpleNamespace(
        model_path=str(model_path), device=args.device, seed=args.seed,
        num_loop_tokens=contract["num_loop_tokens"],
        loop_depth=contract["loop_depth"],
        loop_recycle_mode=contract["loop_recycle_mode"],
        loop_memory_persist=False,
        memory_loop_start_layer=contract["memory_loop_start_layer"],
        memory_loop_end_layer=contract["memory_loop_end_layer"],
        vit_max_image_size=int(config.get("vit_max_image_size", 980)),
        vit_min_image_size=int(config.get("vit_min_image_size", 224)),
        vit_image_stride=int(config.get("vit_image_stride", 14)),
    )
    backbone, inferencer = load_native_bagel(native_args)
    model = backbone.bagel
    if model.config.llm_config.layer_module != "Qwen2MoTDecoderLayer":
        raise ValueError("single-pass prompt mask requires the MoT decoder")
    if args.height % int(model.latent_downsample) or args.width % int(model.latent_downsample):
        raise ValueError("height and width must be divisible by latent_downsample")

    manifest = {
        "schema": "bagel_single_pass_prompt_kv_hard4_v1",
        "complete": False,
        "training_config": str(config_path),
        "training_config_sha256": sha256(config_path),
        "benchmark_data": str(benchmark_path),
        "benchmark_sha256": sha256(benchmark_path),
        "model_path": str(model_path),
        "memory_slots": 8,
        "memory_init": "native_frozen_soi_eoi_embedding",
        "memory_update_layers": [0, MEMORY_UPDATE_END],
        "prompt_mask_layers": [contract["memory_loop_start_layer"], contract["memory_loop_end_layer"]],
        "prompt_mask_rows": "non_memory_only",
        "loop_repeats": 1,
        "memory_persist": False,
        "policies": POLICIES,
        "seed": args.seed,
        "noise_seed_schema": NOISE_SCHEMA,
        "image_shape": [args.height, args.width],
        "num_timesteps": args.num_steps,
        "timestep_shift": args.timestep_shift,
        "rows": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gallery_rows = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        noise_seed = stable_noise_seed(args.seed, prompt, schema=NOISE_SCHEMA)
        noise = make_noise(model, (args.height, args.width), noise_seed)
        prompt_dir = output_dir / f"p{index:03d}"
        prompt_dir.mkdir()
        images = {}
        for policy in POLICIES:
            image = generate(inferencer, prompt, noise, args, mask_body=policy == "mask_body")
            image_path = prompt_dir / f"{policy}.png"
            image.save(image_path)
            images[policy] = image
            print(f"[{index + 1}/{len(rows)}] {policy}: {image_path}", flush=True)
        mae = pixel_mae(images["mask_body"], images["keep"])
        manifest["rows"].append({"prompt_index": index, "prompt": prompt,
                                 "noise_seed": noise_seed, "pixel_mae_mask_vs_keep": mae})
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        gallery_rows.append(
            f"<tr><td>{index:02d}</td><td>{html.escape(prompt)}</td>"
            f"<td><img src='p{index:03d}/keep.png'></td>"
            f"<td><img src='p{index:03d}/mask_body.png'><br>MAE={mae:.3f}</td></tr>"
        )
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>BAGEL single-pass prompt KV</title>"
        "<style>body{font:14px system-ui;background:#151515;color:#eee}"
        "table{border-collapse:collapse}td,th{border:1px solid #555;padding:8px;vertical-align:top}"
        "img{width:280px}</style><h1>Single pass: keep vs body-only prompt-KV mask</h1>"
        "<table><tr><th>#</th><th>Prompt</th><th>keep</th>"
        f"<th>mask [{contract['memory_loop_start_layer']},"
        f"{contract['memory_loop_end_layer']})</th></tr>"
        + "".join(gallery_rows) + "</table>", encoding="utf-8",
    )
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    raise SystemExit("Retired single-pass mask protocol. Use scripts/evaluate/bagel_memory_mechanism.py (strict Read + one Write).")
