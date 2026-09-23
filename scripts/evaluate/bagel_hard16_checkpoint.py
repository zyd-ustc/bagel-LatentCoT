#!/usr/bin/env python3
"""Matched base / frozen-loop / checkpoint-loop generation on 16 hard prompts.

The training YAML defines the loop architecture.  All three arms use one
loaded BAGEL, one prompt, one initial noise, and one generation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from safetensors.torch import load_file

from bagel_common import autocast_for, load_native_bagel, make_noise, stable_noise_seed

from qwen_latent_cot.bagel import accelerator
from qwen_latent_cot.bagel.loop import (
    loop_adapter_state_dict,
    load_loop_adapter_state_dict,
)


ARMS = ("base", "training_free_loop", "trained_loop")
DEFAULT_BENCHMARK = "experiments/data/geneval2_hard_16.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default="", help="Override model_path in training YAML")
    parser.add_argument("--benchmark-data", type=Path, default=Path(DEFAULT_BENCHMARK))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=16, help="1 for a pilot; default is all 16")
    parser.add_argument("--arms", default=",".join(ARMS), help="Subset for a pilot; default is all three")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and contract without loading BAGEL")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 16 or len({str(row["prompt"]) for row in rows}) != 16:
        raise ValueError(f"expected exactly 16 unique hard prompts in {path}, got {len(rows)}")
    if not 1 <= limit <= 16:
        raise ValueError("--max-prompts must be in [1,16]")
    return rows[:limit]


def resolve_contract(config: dict, metadata: dict) -> dict:
    """Use training settings, never a hard-coded evaluation body/R.

    Phase 1.1 metadata R=1 describes its single Read, not generation depth.
    """
    contract = {
        "num_loop_tokens": int(config["num_loop_tokens"]),
        "loop_depth": int(config["loop_depth"]),
        "loop_recycle_mode": str(config["loop_recycle_mode"]),
        "loop_memory_persist": bool(config["loop_memory_persist"]),
        "memory_loop_start_layer": int(config["memory_loop_start_layer"]),
        "memory_loop_end_layer": int(config["memory_loop_end_layer"]),
        "round0_memory_write_enabled": bool(config["round0_memory_write_enabled"]),
        "loop_uncond_memory": str(config.get("loop_uncond_memory", "m0")),
        "lora_rank": int(config["lora_rank"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config.get("lora_dropout", 0.0)),
        "gen_attention_o_lora": bool(config.get("gen_attention_o_lora", False)),
        "k_v_lora": bool(config.get("k_v_lora", False)),
    }
    if contract["num_loop_tokens"] < 1 or contract["loop_depth"] < 1:
        raise ValueError("three-arm comparison requires K>=1 and generation loop_depth>=1")
    if contract["loop_recycle_mode"] not in {"same_depth", "full_depth"}:
        raise ValueError("unsupported loop_recycle_mode")
    if not 0 <= contract["memory_loop_start_layer"] < contract["memory_loop_end_layer"]:
        raise ValueError("invalid training loop body")
    for key, value in {
        "K": contract["num_loop_tokens"],
        "num_loop_tokens": contract["num_loop_tokens"],
        "loop_depth": contract["loop_depth"],
        "memory_loop_start_layer": contract["memory_loop_start_layer"],
        "memory_loop_end_layer": contract["memory_loop_end_layer"],
        "lora_rank": contract["lora_rank"],
        "lora_alpha": contract["lora_alpha"],
    }.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"checkpoint {key}={metadata[key]!r} disagrees with training config {value!r}")
    if "body" in metadata and list(metadata["body"]) != [
        contract["memory_loop_start_layer"], contract["memory_loop_end_layer"]
    ]:
        raise ValueError("checkpoint body disagrees with training config")
    if "R" in metadata and metadata.get("schema") != "bagel_pair_grounded_memory_adapter_v1":
        if int(metadata["R"]) != contract["loop_depth"]:
            raise ValueError("checkpoint R disagrees with generation loop_depth")
    for key in ("loop_recycle_mode", "loop_memory_persist", "round0_memory_write_enabled", "loop_uncond_memory", "k_v_lora", "gen_attention_o_lora"):
        if key in metadata and metadata[key] != contract[key]:
            raise ValueError(f"checkpoint {key} disagrees with training config")
    for key, value in {
        "num_read_rounds": 0 if contract["round0_memory_write_enabled"] else 1,
        "num_write_rounds": contract["loop_depth"] if contract["round0_memory_write_enabled"] else contract["loop_depth"] - 1,
    }.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"checkpoint {key} disagrees with training config")
    return contract


def selected_arms(raw: str) -> tuple[str, ...]:
    arms = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError(f"--arms must be a unique subset of {','.join(ARMS)}")
    return arms


def validate_adapter_state(state: dict, model, metadata: dict) -> tuple[str, ...]:
    """Accept only complete routes; Phase 1.1 UND-Q may omit zero GEN-Q."""
    expected = loop_adapter_state_dict(model)
    unknown = sorted(set(state) - set(expected))
    if unknown:
        raise RuntimeError(f"checkpoint has unexpected adapter keys: {unknown[:4]}")
    allowed_partial = metadata.get("schema") == "bagel_pair_grounded_memory_adapter_v1"
    missing = sorted(set(expected) - set(state))
    if missing and (not allowed_partial or any(".q_proj_moe_gen." not in key for key in missing)):
        raise RuntimeError(f"checkpoint is incomplete: {missing[:4]}")
    if not state or any(".lora_B." not in key and ".lora_A." not in key for key in state):
        raise RuntimeError("checkpoint contains no valid loop LoRA tensors")
    for key, value in state.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise RuntimeError(f"adapter shape mismatch: {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}")
    return tuple(missing)


def set_loop(model, contract: dict, enabled: bool) -> None:
    values = {
        "num_loop_tokens": contract["num_loop_tokens"] if enabled else 0,
        "loop_depth": contract["loop_depth"] if enabled else 1,
        "loop_recycle_mode": contract["loop_recycle_mode"],
        "loop_memory_persist": contract["loop_memory_persist"] if enabled else False,
        "memory_loop_start_layer": contract["memory_loop_start_layer"],
        "memory_loop_end_layer": contract["memory_loop_end_layer"],
        "round0_memory_write_enabled": contract["round0_memory_write_enabled"] if enabled else False,
        "round0_gen_reads_memory": contract["round0_memory_write_enabled"] if enabled else False,
        "num_read_rounds": 0 if enabled and contract["round0_memory_write_enabled"] else 1,
        "num_write_rounds": (contract["loop_depth"] if contract["round0_memory_write_enabled"] else contract["loop_depth"] - 1) if enabled else 0,
        "loop_uncond_memory": contract["loop_uncond_memory"],
    }
    for key, value in values.items():
        setattr(model, key, value)
        setattr(model.config, key, value)


def set_adapter(model, state: dict | None) -> None:
    """Zero all residuals for training-free; restore exact checkpoint for trained."""
    if state is None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_B." in name:
                    parameter.zero_()
        return
    missing = load_loop_adapter_state_dict(
        model, state, allow_missing_projections=("q_proj_moe_gen",)
    )
    if missing:
        # Only the Phase 1.1 UND-Q-only adapter reaches here.  Do not retain
        # a previous checkpoint's GEN-Q weights when reusing the same model.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in missing and ".lora_B." in name:
                    parameter.zero_()


def generate(inferencer, prompt: str, noise: torch.Tensor, image_shape: tuple[int, int], args):
    return inferencer(
        image=None, text=prompt, image_shapes=image_shape, init_noise=noise.clone(),
        enable_taylorseer=False, return_loop_diagnostics=True,
        cfg_text_scale=args.cfg_text_scale, cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.4, 1.0], cfg_renorm_min=0.0, cfg_renorm_type="global",
        num_timesteps=args.num_steps, timestep_shift=args.timestep_shift,
    )["image"]


def main() -> None:
    args = parse_args()
    arms = selected_arms(args.arms)
    config_path = args.training_config.expanduser().resolve()
    adapter_path = args.adapter.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training YAML must be a mapping")
    metadata_path = adapter_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    contract = resolve_contract(config, metadata)
    benchmark_path = args.benchmark_data.expanduser().resolve()
    rows = load_benchmark(benchmark_path, args.max_prompts)
    if args.height <= 0 or args.width <= 0 or args.num_steps < 2:
        raise ValueError("image dimensions must be positive and num-steps >= 2")
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    model_path = Path(args.model_path or config["model_path"]).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.dry_run:
        print(json.dumps({"contract": contract, "arms": arms, "prompts": len(rows), "adapter_sha256": sha256(adapter_path)}, indent=2))
        return

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    native_args = SimpleNamespace(
        model_path=str(model_path), output_dir=str(output_dir), device=args.device,
        seed=args.seed, num_loop_tokens=contract["num_loop_tokens"],
        loop_depth=contract["loop_depth"], loop_recycle_mode=contract["loop_recycle_mode"],
        loop_memory_persist=contract["loop_memory_persist"],
        memory_loop_start_layer=contract["memory_loop_start_layer"],
        memory_loop_end_layer=contract["memory_loop_end_layer"],
        round0_memory_write_enabled=contract["round0_memory_write_enabled"],
        round0_gen_reads_memory=contract["round0_memory_write_enabled"],
        vit_max_image_size=int(config.get("vit_max_image_size", 980)),
        vit_min_image_size=int(config.get("vit_min_image_size", 224)),
        vit_image_stride=int(config.get("vit_image_stride", 14)),
    )
    backbone, inferencer = load_native_bagel(native_args)
    model = backbone.bagel
    assert model is not None
    backbone.apply_loop_trainable_policy(
        start_layer=contract["memory_loop_start_layer"],
        end_layer=contract["memory_loop_end_layer"],
        rank=contract["lora_rank"], alpha=contract["lora_alpha"],
        dropout=contract["lora_dropout"],
        gen_attention_o_lora=contract["gen_attention_o_lora"],
        k_v_lora=contract["k_v_lora"],
    )
    model.eval().requires_grad_(False)
    state = load_file(str(adapter_path), device="cpu")
    missing = validate_adapter_state(state, model, metadata)
    if missing and metadata.get("schema") != "bagel_pair_grounded_memory_adapter_v1":
        raise RuntimeError("partial adapters require the Phase 1.1 sidecar metadata")
    if int(args.height) % int(model.latent_downsample) or int(args.width) % int(model.latent_downsample):
        raise ValueError(f"height/width must be divisible by {model.latent_downsample}")
    manifest = {
        "schema": "bagel_hard16_checkpoint_eval_v1", "training_config": str(config_path),
        "training_config_sha256": sha256(config_path), "adapter": str(adapter_path),
        "adapter_sha256": sha256(adapter_path), "adapter_metadata": metadata,
        "model_path": str(model_path), "benchmark_data": str(benchmark_path),
        "benchmark_sha256": sha256(benchmark_path), "contract": contract,
        "arms": [{"id": arm, "slug": arm} for arm in arms],
        "prompts": [str(row["prompt"]) for row in rows], "prompt_count": len(rows),
        "noise_seed_schema": "bagel-hard16-checkpoint-v1", "prompt_runs": [],
        "complete": False,
        "image_shape": [args.height, args.width], "seed": args.seed,
        "hyper": {"num_timesteps": args.num_steps, "timestep_shift": args.timestep_shift,
                  "cfg_text_scale": args.cfg_text_scale, "cfg_img_scale": args.cfg_img_scale,
                  "cfg_interval": [0.4, 1.0], "cfg_renorm_min": 0.0,
                  "cfg_renorm_type": "global", "enable_taylorseer": False},
        "geneval2_image_maps": {arm: f"geneval2/{arm}_image_paths.json" for arm in arms},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "benchmark_hard.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    maps = {arm: {} for arm in arms}
    gallery_rows = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        seed = stable_noise_seed(args.seed, prompt, schema="bagel-hard16-checkpoint-v1")
        noise = make_noise(model, (args.height, args.width), seed)
        noise_hash = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
        prompt_dir = output_dir / f"p{index:03d}"
        prompt_dir.mkdir()
        cells = []
        for arm in arms:
            set_loop(model, contract, arm != "base")
            if arm == "training_free_loop":
                set_adapter(model, None)
            elif arm == "trained_loop":
                set_adapter(model, state)
            with torch.inference_mode(), autocast_for(accelerator.resolve_device(args.device)):
                image = generate(inferencer, prompt, noise, (args.height, args.width), args)
            image_path = prompt_dir / f"{arm}.png"
            image.save(image_path)
            maps[arm][prompt] = str(image_path.resolve())
            cells.append(f"<td><img src='{html.escape(prompt_dir.name + '/' + arm + '.png')}'><br>{html.escape(arm)}</td>")
            print(f"[{index + 1}/{len(rows)}] {arm}: {image_path}", flush=True)
        manifest["prompt_runs"].append({
            "index": index, "prompt": prompt, "noise_seed": seed,
            "noise_sha256": noise_hash,
            "images": {arm: maps[arm][prompt] for arm in arms},
        })
        gallery_rows.append(f"<tr><td>{index:02d}</td><td>{html.escape(prompt)}</td>{''.join(cells)}</tr>")
    maps_dir = output_dir / "geneval2"
    maps_dir.mkdir()
    for arm, image_map in maps.items():
        (maps_dir / f"{arm}_image_paths.json").write_text(json.dumps(image_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    header = "".join(f"<th>{html.escape(arm)}</th>" for arm in arms)
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>BAGEL hard16 checkpoint</title>"
        "<style>body{font:14px system-ui;background:#151515;color:#eee;padding:20px}"
        "table{border-collapse:collapse}td,th{border:1px solid #555;padding:8px;vertical-align:top}"
        "img{width:240px}td:nth-child(2){max-width:260px}</style>"
        f"<h1>BAGEL hard16 checkpoint comparison</h1><table><tr><th>#</th><th>Prompt</th>{header}</tr>"
        + "".join(gallery_rows) + "</table>", encoding="utf-8"
    )
    manifest["complete"] = True
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
