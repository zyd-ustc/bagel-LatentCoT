#!/usr/bin/env python3
"""Prompt-as-memory hard-16 T2I: Write source × prompt-KV visibility.

The eight cells reuse one packed pair of prompts, prompt cache, and noises.
Masking changes only non-memory generation queries' access to cached prompt
keys; memory queries retain access. This does not score semantic image quality.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from bagel_common import autocast_for, load_native_bagel, make_noise, pixel_mae, stable_noise_seed
from bagel_hard16_checkpoint import (
    load_benchmark,
    resolve_contract,
    set_adapter,
    set_loop,
    sha256,
)

from qwen_latent_cot.bagel import accelerator
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache
from qwen_latent_cot.bagel.write_sensitivity import (
    PROMPT_MEMORY_SLOT_BETA,
    prompt_memory_init,
)


ARM_SOURCES = {
    "correct_M": "correct",
    "shuffled_across_sample_M": "shuffle",
    "m0": "m0",
    "zero_M": "zero",
}
KV_POLICIES = {"keep": False, "mask_nonmemory": True}
PAIR_SIZE = 2
NOISE_SCHEMA = "bagel-write-sensitivity-t2i-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default="", help="Override model_path in the training YAML")
    parser.add_argument("--benchmark-data", type=Path, default=Path("experiments/data/geneval2_hard_16.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=16, help="Even number in [2,16]; use 2 for smoke")
    parser.add_argument("--dry-run", action="store_true", help="Validate contract without loading BAGEL")
    return parser.parse_args()


def validate_protocol(contract: dict, args) -> None:
    required = {
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "loop_recycle_mode": "same_depth",
        "loop_memory_persist": False,
        "round0_memory_write_enabled": False,
    }
    mismatches = {key: (contract[key], expected) for key, expected in required.items() if contract[key] != expected}
    if mismatches:
        raise ValueError(f"Write sensitivity requires K=8, strict Read + one Write, same_depth, persist=False: {mismatches}")
    if contract["loop_uncond_memory"] != "m0":
        raise ValueError("this comparison fixes CFG unconditioned loop memory to m0")
    if not 2 <= args.max_prompts <= 16 or args.max_prompts % PAIR_SIZE:
        raise ValueError("--max-prompts must be an even number in [2,16]")
    if args.height <= 0 or args.width <= 0 or args.num_steps < 2 or args.timestep_shift <= 0:
        raise ValueError("image size, num-steps, and timestep-shift must be positive")
    if args.cfg_text_scale != 4.0 or args.cfg_img_scale != 1.0:
        raise ValueError("this experiment fixes CFG text/image scales to 4/1")


def paired_rows(rows: list[dict]) -> list[list[dict]]:
    if len(rows) < 2 or len(rows) % PAIR_SIZE:
        raise ValueError("shuffle requires complete pairs")
    return [rows[index:index + PAIR_SIZE] for index in range(0, len(rows), PAIR_SIZE)]


def move_tensors(values: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def code_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare_pair(inferencer, rows: list[dict], noises: list[torch.Tensor], shape: tuple[int, int]):
    """Build a *real* native packed batch, including prompt KV and CFG layouts."""

    model = inferencer.model
    device = inferencer.device
    prompts = [str(row["prompt"]) for row in rows]
    count = len(prompts)
    if count != PAIR_SIZE or len(noises) != count:
        raise ValueError("the Write-sensitivity batch must contain exactly two prompts/noises")
    full_cache = NaiveCache(model.config.llm_config.num_hidden_layers)
    prompt_input, full_lens, full_ropes = model.prepare_prompts(
        curr_kvlens=[0] * count,
        curr_rope=[0] * count,
        prompts=prompts,
        tokenizer=inferencer.tokenizer,
        new_token_ids=inferencer.new_token_ids,
    )
    full_cache, prompt_body_entry = model.forward_cache_update_text(
        full_cache,
        capture_body_entry_at=int(model.config.memory_loop_start_layer),
        **move_tensors(prompt_input, device),
    )
    memory_init = prompt_memory_init(
        prompt_body_entry,
        slots=int(model.config.num_loop_tokens),
    )
    image_shapes = [shape] * count
    generation = model.prepare_vae_latent(
        curr_kvlens=full_lens,
        curr_rope=full_ropes,
        image_sizes=image_shapes,
        new_token_ids=inferencer.new_token_ids,
        num_loop_tokens=int(model.config.num_loop_tokens),
    )
    initial_noise = torch.cat(noises, dim=0)
    if generation["packed_init_noises"].shape != initial_noise.shape:
        raise ValueError("paired noise shape differs from BAGEL packed latent geometry")
    generation["packed_init_noises"] = initial_noise
    generation = move_tensors(generation, device)
    cfg_text = move_tensors(
        model.prepare_vae_latent_cfg(
            curr_kvlens=[0] * count,
            curr_rope=[0] * count,
            image_sizes=image_shapes,
            num_loop_tokens=int(model.config.num_loop_tokens),
        ), device,
    )
    cfg_img = move_tensors(
        model.prepare_vae_latent_cfg(
            curr_kvlens=full_lens,
            curr_rope=full_ropes,
            image_sizes=image_shapes,
            num_loop_tokens=int(model.config.num_loop_tokens),
        ), device,
    )
    return {
        **generation,
        "past_key_values": full_cache,
        "cfg_text_past_key_values": NaiveCache(model.config.llm_config.num_hidden_layers),
        "cfg_img_past_key_values": full_cache,
        "cfg_text_packed_position_ids": cfg_text["cfg_packed_position_ids"],
        "cfg_text_packed_query_indexes": cfg_text["cfg_packed_query_indexes"],
        "cfg_text_key_values_lens": cfg_text["cfg_key_values_lens"],
        "cfg_text_packed_key_value_indexes": cfg_text["cfg_packed_key_value_indexes"],
        "cfg_img_packed_position_ids": cfg_img["cfg_packed_position_ids"],
        "cfg_img_packed_query_indexes": cfg_img["cfg_packed_query_indexes"],
        "cfg_img_key_values_lens": cfg_img["cfg_key_values_lens"],
        "cfg_img_packed_key_value_indexes": cfg_img["cfg_packed_key_value_indexes"],
        "prompt_body_memory_init": memory_init,
    }


def generate_pair(
    inferencer, bundle: dict, args, source: str, probe: list[dict],
    *, mask_prompt_kv: bool = False,
) -> list:
    with torch.inference_mode(), autocast_for(inferencer.device):
        latents = inferencer.model.generate_image(
            **bundle,
            num_timesteps=int(args.num_steps),
            timestep_shift=float(args.timestep_shift),
            cfg_text_scale=float(args.cfg_text_scale),
            cfg_img_scale=float(args.cfg_img_scale),
            cfg_interval=[0.4, 1.0],
            cfg_renorm_min=0.0,
            cfg_renorm_type="sample_global",
            enable_taylorseer=False,
            return_loop_diagnostics=False,
            memory_write_source=source,
            memory_write_probe=probe,
            mask_prompt_kv_for_nonmemory=mask_prompt_kv,
        )
        if len(latents) != PAIR_SIZE:
            raise RuntimeError(f"BAGEL returned {len(latents)} images, expected {PAIR_SIZE}")
        return [inferencer.decode_image(latent, (args.height, args.width)) for latent in latents]


def summarize_probe(rows: list[dict], sample: int, expected_steps: int) -> dict[str, float | None]:
    selected = [row for row in rows if row["sample_in_pair"] == sample]
    if len(selected) != expected_steps:
        raise RuntimeError(f"memory probe has {len(selected)} steps for sample {sample}, expected {expected_steps}")
    result: dict[str, float | None] = {"steps": len(selected)}
    for key in ("read_l2", "m0_l2", "read_minus_m0_l2", "read_m0_cos", "read_write_input_cos"):
        values = [float(row[key]) for row in selected if row[key] is not None]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"nonfinite memory probe: {key}")
        result[key] = sum(values) / len(values) if values else None
    return result


def write_gallery(output_dir: Path, rows: list[dict], image_maps: dict[str, dict[str, str]]) -> None:
    header = "".join(
        f"<th>{html.escape(policy)}<br>{html.escape(arm)}</th>"
        for policy in KV_POLICIES for arm in ARM_SOURCES
    )
    body = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        cells = "".join(
            f"<td><img src='p{index:03d}/{policy}/{arm}.png'><br>{html.escape(policy)} / {html.escape(arm)}</td>"
            for policy in KV_POLICIES for arm in ARM_SOURCES
        )
        body.append(f"<tr><td>{index:02d}</td><td>{html.escape(prompt)}</td>{cells}</tr>")
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>BAGEL Prompt KV × Write</title>"
        "<style>body{font:14px system-ui;background:#151515;color:#eee;padding:20px}"
        "table{border-collapse:collapse}td,th{border:1px solid #555;padding:8px;vertical-align:top}"
        "img{width:240px}td:nth-child(2){max-width:260px}</style>"
        f"<h1>BAGEL Prompt KV visibility × Write source: paired hard prompts</h1><table><tr><th>#</th><th>Prompt</th>{header}</tr>"
        + "".join(body) + "</table>", encoding="utf-8",
    )
    maps_dir = output_dir / "geneval2"
    maps_dir.mkdir()
    for cell, image_map in image_maps.items():
        (maps_dir / f"{cell}_image_paths.json").write_text(
            json.dumps(image_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    config_path = args.training_config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training YAML must be a mapping")
    contract = resolve_contract(config, {})
    validate_protocol(contract, args)
    benchmark_path = args.benchmark_data.expanduser().resolve()
    rows = load_benchmark(benchmark_path, args.max_prompts)
    pairs = paired_rows(rows)
    model_path = Path(args.model_path or config["model_path"]).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.dry_run:
        print(json.dumps({"contract": contract, "pairs": len(pairs), "memory_init": "prompt_body_entry_eos"}, indent=2))
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
    if model.config.llm_config.layer_module != "Qwen2MoTDecoderLayer":
        raise ValueError("prompt KV masking requires the BAGEL MoT decoder")
    backbone.apply_loop_trainable_policy(
        start_layer=contract["memory_loop_start_layer"],
        end_layer=contract["memory_loop_end_layer"],
        rank=contract["lora_rank"], alpha=contract["lora_alpha"],
        dropout=contract["lora_dropout"],
        gen_attention_o_lora=contract["gen_attention_o_lora"],
        k_v_lora=contract["k_v_lora"],
    )
    model.eval().requires_grad_(False)
    set_loop(model, contract, True)
    set_adapter(model, None)
    if args.height % int(model.latent_downsample) or args.width % int(model.latent_downsample):
        raise ValueError(f"height/width must be divisible by {model.latent_downsample}")

    manifest: dict[str, Any] = {
        "schema": "bagel_prompt_memory_kv_visibility_write_hard16_v3",
        "complete": False,
        "code_commit": code_commit(),
        "training_config": str(config_path),
        "training_config_sha256": sha256(config_path),
        "adapter": None,
        "memory_init": {
            "source": "causal_final_prompt_token_at_body_entry",
            "body_entry_layer": contract["memory_loop_start_layer"],
            "slot_offset": "centered_sin_cos_global_rms_v2",
            "slot_offset_beta": PROMPT_MEMORY_SLOT_BETA,
            "cfg_unconditioned_memory": "boundary_embedding_m0",
        },
        "model_path": str(model_path),
        "benchmark_data": str(benchmark_path),
        "benchmark_sha256": sha256(benchmark_path),
        "contract": contract,
        "arm_sources": ARM_SOURCES,
        "prompt_kv_policies": {
            "keep": "all generation queries see cached prompt KV",
            "mask_nonmemory": "only memory queries see cached prompt KV, at every generation layer",
        },
        "pair_size": PAIR_SIZE,
        "pairing": "adjacent_file_order; shuffle=roll(samples,+1)",
        "prompts": [str(row["prompt"]) for row in rows],
        "prompt_count": len(rows),
        "seed": args.seed,
        "noise_seed_schema": NOISE_SCHEMA,
        "image_shape": [args.height, args.width],
        "hyper": {"num_timesteps": args.num_steps, "timestep_shift": args.timestep_shift,
                  "cfg_text_scale": args.cfg_text_scale, "cfg_img_scale": args.cfg_img_scale,
                  "cfg_interval": [0.4, 1.0], "cfg_renorm_min": 0.0,
                  "cfg_renorm_type": "sample_global", "return_loop_diagnostics": False},
        "pairs": [],
        "geneval2_image_maps": {
            f"{policy}__{arm}": f"geneval2/{policy}__{arm}_image_paths.json"
            for policy in KV_POLICIES for arm in ARM_SOURCES
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "benchmark_hard.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    image_maps: dict[str, dict[str, str]] = {
        f"{policy}__{arm}": {} for policy in KV_POLICIES for arm in ARM_SOURCES
    }
    image_shape = (args.height, args.width)
    for pair_index, pair in enumerate(pairs):
        global_indexes = [pair_index * PAIR_SIZE + position for position in range(PAIR_SIZE)]
        noises = [
            make_noise(model, image_shape, stable_noise_seed(args.seed, str(row["prompt"]), schema=NOISE_SCHEMA))
            for row in pair
        ]
        with torch.inference_mode(), autocast_for(accelerator.resolve_device(args.device)):
            bundle = prepare_pair(inferencer, pair, noises, image_shape)
        memory_init = bundle["prompt_body_memory_init"].detach().float().cpu()
        pair_result: dict[str, Any] = {
            "pair_index": pair_index,
            "prompt_indexes": global_indexes,
            "donor_indexes": list(reversed(global_indexes)),
            "noise_seed": [stable_noise_seed(args.seed, str(row["prompt"]), schema=NOISE_SCHEMA) for row in pair],
            "noise_sha256": [hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest() for noise in noises],
            "prompt_memory_init_sha256": [
                hashlib.sha256(memory.contiguous().numpy().tobytes()).hexdigest()
                for memory in memory_init.reshape(PAIR_SIZE, contract["num_loop_tokens"], -1)
            ],
            "policies": {},
        }
        images: dict[str, dict[str, list]] = {}
        pair_dir = output_dir / f"pair_{pair_index:02d}"
        pair_dir.mkdir()
        for policy, mask_prompt_kv in KV_POLICIES.items():
            images[policy] = {}
            pair_result["policies"][policy] = {}
            for arm, source in ARM_SOURCES.items():
                probe: list[dict] = []
                images[policy][arm] = generate_pair(
                    inferencer, bundle, args, source, probe,
                    mask_prompt_kv=mask_prompt_kv,
                )
                if len(probe) != (args.num_steps - 1) * PAIR_SIZE:
                    raise RuntimeError(f"{policy}/{arm}: incomplete Read→Write probe: {len(probe)} rows")
                for step in range(args.num_steps - 1):
                    for position in range(PAIR_SIZE):
                        probe[step * PAIR_SIZE + position]["step"] = step
                probe_path = pair_dir / f"{policy}__{arm}_memory_probe.json"
                probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
                arm_rows = []
                for position, (global_index, row, image) in enumerate(zip(global_indexes, pair, images[policy][arm])):
                    prompt_dir = output_dir / f"p{global_index:03d}"
                    prompt_dir.mkdir(exist_ok=True)
                    (prompt_dir / "prompt.txt").write_text(str(row["prompt"]) + "\n", encoding="utf-8")
                    policy_dir = prompt_dir / policy
                    policy_dir.mkdir(exist_ok=True)
                    image_path = policy_dir / f"{arm}.png"
                    image.save(image_path)
                    image_maps[f"{policy}__{arm}"][str(row["prompt"])] = str(image_path.resolve())
                    arm_rows.append({
                        "prompt_index": global_index,
                        "image": str(image_path.resolve()),
                        "write_source": source,
                        "mask_prompt_kv_for_nonmemory": mask_prompt_kv,
                        "memory_probe": str(probe_path.resolve()),
                        "probe_mean": summarize_probe(probe, position, args.num_steps - 1),
                    })
                    print(f"[{global_index + 1}/{len(rows)}] {policy}/{arm}: {image_path}", flush=True)
                pair_result["policies"][policy][arm] = arm_rows
        for position, global_index in enumerate(global_indexes):
            for policy in KV_POLICIES:
                for arm in ARM_SOURCES:
                    pair_result["policies"][policy][arm][position]["pixel_mae_vs_correct_M"] = (
                        0.0 if arm == "correct_M" else pixel_mae(images[policy][arm][position], images[policy]["correct_M"][position])
                    )
            for arm in ARM_SOURCES:
                pair_result["policies"]["mask_nonmemory"][arm][position]["pixel_mae_vs_keep_same_arm"] = pixel_mae(
                    images["mask_nonmemory"][arm][position], images["keep"][arm][position]
                )
        manifest["pairs"].append(pair_result)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_gallery(output_dir, rows, image_maps)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
