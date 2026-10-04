#!/usr/bin/env python3
"""Matched-noise topology/depth matrix, timestep logs, and functional readouts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import time
from dataclasses import replace

import torch

from qwen_latent_cot.bagel import BagelBackbone, LoopConfig
from qwen_latent_cot.bagel.anchored_loop import LoopModules
from qwen_latent_cot.bagel.accelerator import (
    autocast_for,
    manual_seed_all,
    resolve_device,
    synchronize,
)
from qwen_latent_cot.bagel.flow_time import shift_flow_timestep
from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
from qwen_latent_cot.bagel.loop_checkpoint import (
    checkpoint_config,
    load_loop_checkpoint,
)


def timestep_bin(index, count):
    fraction = index / max(count, 1)
    return "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"


def arm_config(config, mode, depth, memory_slots, memory_control):
    native_control = mode == "legacy_memory_only" or mode.startswith("direct_native_")
    return replace(
        config,
        enable_t2i_loop=depth > 0,
        runtime_loop_depth=depth,
        loop_mode="gen_only" if mode == "base" else mode,
        memory_slots=0
        if mode in {"base", "gen_only", "direct_native_gen_only"}
        else memory_slots,
        memory_control=memory_control,
        log_loop_stats=True,
        loop_gate_init=0.0 if native_control else config.loop_gate_init,
        loop_output_alpha_init=0.0 if native_control else config.loop_output_alpha_init,
    )


def arm_semantics(runtime):
    if runtime.loop_mode == "legacy_memory_only":
        return {
            "gate": "parent_ungated",
            "readout": "parent_suffix_no_alpha",
            "memory_writer": "parent",
        }
    if runtime.loop_mode.startswith("direct_native_"):
        return {
            "gate": "ungated",
            "readout": "current_gen_no_alpha",
            "memory_writer": "canonical_correct"
            if runtime.memory_control == "shuffled"
            else "shared",
        }
    return {
        "gate": "carry_gen_delta_gate_new_layer_write",
        "readout": "base_plus_alpha_delta",
        "memory_writer": "canonical_correct"
        if runtime.memory_control == "shuffled"
        else "shared",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompts", required=True, help="JSONL with prompt; optional bucket/skills"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--modes",
        default="gen_only,gen_memory_anchored,memory_only,legacy_memory_only,direct_native_gen_only",
    )
    parser.add_argument("--depths", default="0,1,2,3,4")
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument(
        "--memory-control",
        choices=["correct", "zero", "frozen", "shuffled"],
        default="correct",
    )
    parser.add_argument("--start-layer", type=int, default=16)
    parser.add_argument("--end-layer", type=int, default=24)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="manual topology alpha; ignored with trained checkpoint",
    )
    parser.add_argument("--gate", type=float, default=0.02)
    parser.add_argument("--reentry-scale", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--skip-base", action="store_true", help="omit duplicate Base when another worker provides it")
    parser.add_argument(
        "--no-loop-stats", action="store_true",
        help="generate final images without per-round suffix/diagnostic probes",
    )
    parser.add_argument(
        "--save-readouts",
        action="store_true",
        help="save per-round velocity and x0 estimates at early/middle/late probes",
    )
    args = parser.parse_args()
    run_matrix(args)


def run_matrix(args, backbone=None):
    """Run an arm matrix; optionally reuse frozen native weights across jobs."""
    if getattr(args, "no_loop_stats", False) and args.save_readouts:
        raise ValueError("per-round readout probes require loop stats")
    if args.num_timesteps < 2 or args.timestep_shift <= 0:
        raise ValueError("require >=2 timesteps and positive shift")
    rows = [
        json.loads(line)
        for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ][: args.max_prompts]
    if not rows or any(not row.get("prompt") for row in rows):
        raise ValueError("nonempty prompt JSONL is required")
    if args.batch_size < 1 or args.max_prompts < 1:
        raise ValueError("batch-size and max-prompts must be positive")
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    valid_modes = {
        "gen_only",
        "gen_memory_anchored",
        "memory_only",
        "legacy_memory_only",
        "direct_native_gen_only",
        "direct_native_memory",
    }
    if not modes or not set(modes) <= valid_modes:
        raise ValueError("unsupported or empty mode list")
    if {"memory_only", "legacy_memory_only", "direct_native_memory"} & set(
        modes
    ) and args.memory_slots == 0:
        raise ValueError("memory_only requires positive memory slots")
    if "legacy_memory_only" in modes and args.memory_control != "correct":
        raise ValueError(
            "legacy_memory_only is an unchanged control; choose new workspace modes for memory interventions"
        )
    if args.memory_control == "shuffled" and any(
        mode not in {"gen_only", "direct_native_gen_only"} for mode in modes
    ):
        if args.batch_size < 2 or len(rows) % args.batch_size == 1:
            raise ValueError(
                "shuffled workspace requires every batch to contain at least two samples"
            )
    depths = sorted(set(int(value) for value in args.depths.split(",")))
    config = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=args.start_layer,
        loop_end_layer=args.end_layer,
        runtime_loop_depth=max(depths),
        allocated_max_loop_depth=max(4, max(depths)),
        memory_slots=args.memory_slots,
        reentry_adapter_type="fixed",
        fixed_reentry_scale=args.reentry_scale,
        loop_output_alpha_init=args.alpha,
        loop_gate_init=args.gate,
    )
    if args.checkpoint:
        metadata = checkpoint_config(args.checkpoint)
        config = LoopConfig(**metadata["loop_config"])
        anchored_modes = {"gen_only", "gen_memory_anchored", "memory_only"} & set(modes)
        if anchored_modes and max(depths) > config.allocated_max_loop_depth:
            raise ValueError("depth matrix exceeds checkpoint allocation")
        if args.memory_slots > config.memory_slots and any(
            mode in {"gen_memory_anchored", "memory_only", "direct_native_memory"}
            for mode in modes
        ):
            raise ValueError("memory matrix exceeds checkpoint workspace allocation")
    if backbone is None:
        backbone = BagelBackbone({
            "model_path": args.model_path,
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "t2i_loop": config.to_dict(),
        }).load()
    else:
        if Path(backbone.cfg["model_path"]).resolve() != Path(args.model_path).resolve():
            raise ValueError("cached native checkpoint differs from requested model")
        # Each trained arm has its own K and parameter shapes. Reuse only the
        # immutable native tensors, and replace the complete loop module.
        model = backbone.bagel
        model.t2i_loop = LoopModules(model.llm2vae.in_features, config).to(
            device=next(model.parameters()).device, dtype=torch.float32
        )
        model.t2i_loop.initialize_memory_from_boundaries(
            model.language_model.model.embed_tokens.weight,
            [backbone.token_ids["start_of_image"], backbone.token_ids["end_of_image"]],
            seed=int(backbone.cfg.get("memory_init_seed", 0)),
        )
    config = replace(config, log_loop_stats=not getattr(args, "no_loop_stats", False))
    device = resolve_device(args.device)
    manual_seed_all(args.seed)
    model, vae = backbone.bagel.to(device).eval(), backbone.vae_model.to(device).eval()
    model.requires_grad_(False)
    if args.checkpoint:
        load_loop_checkpoint(model, args.checkpoint)
    inferencer = InterleaveInferencer(
        model, vae, backbone.tokenizer, backbone.token_ids
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, arm_metadata = [], {}
    schedule = torch.linspace(1, 0, args.num_timesteps, device=device)
    schedule = shift_flow_timestep(schedule, args.timestep_shift)
    dts = schedule[:-1] - schedule[1:]
    probe_steps = {0, len(dts) // 2, len(dts) - 1}
    arms = ([] if getattr(args, "skip_base", False) else [("base", 0)]) + [
        (mode, depth) for mode in modes for depth in depths if depth > 0
    ]
    with torch.no_grad(), autocast_for(device):
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            condition = inferencer.prepare_condition(
                [row["prompt"] for row in batch],
                [
                    (
                        int(row.get("height", args.image_size)),
                        int(row.get("width", args.image_size)),
                    )
                    for row in batch
                ],
            )
            gen_counts = (condition.inputs["packed_seqlens"] - 2).tolist()
            count = len(condition.inputs["packed_vae_token_indexes"])
            generator = torch.Generator().manual_seed(args.seed + offset)
            initial_noise = torch.randn(
                count, model.patch_latent_dim, generator=generator
            ).to(device)
            noise_hashes = [
                hashlib.sha256(
                    part.float().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                for part in initial_noise.split(gen_counts)
            ]
            for mode, depth in arms:
                runtime = arm_config(
                    config, mode, depth, args.memory_slots, args.memory_control
                )
                runtime = replace(runtime, log_loop_stats=config.log_loop_stats)
                slots = runtime.memory_slots
                arm = f"{mode}_R{depth}_K{slots}_{args.memory_control}"
                coverage = (
                    metadata.get("round_training_steps") if args.checkpoint else None
                )
                depth_status = (
                    "training_free"
                    if not args.checkpoint
                    else "unknown"
                    if coverage is None
                    else "seen"
                    if depth == 0 or (depth <= len(coverage) and all(coverage[:depth]))
                    else "unseen"
                )
                if (
                    depth == 0
                    or mode == "legacy_memory_only"
                    or mode.startswith("direct_native_")
                ):
                    depth_status = "training_free"
                arm_metadata[arm] = {
                    "depth_status": depth_status,
                    "runtime_config": runtime.to_dict(),
                    **arm_semantics(runtime),
                    "learned_alpha": float(model.t2i_loop.output_alpha[0])
                    if args.checkpoint and depth and not (mode == "legacy_memory_only" or mode.startswith("direct_native_")) else None,
                }
                directory = out / arm
                directory.mkdir(exist_ok=True)
                x_t = initial_noise.clone()
                logs = []
                synchronize(device)
                started = time.perf_counter()
                for step, (t, dt) in enumerate(zip(schedule[:-1], dts)):
                    scale = args.cfg_text_scale if 0.4 < float(t) <= 1 else 1.0
                    result = model.forward_t2i_loop(
                        x_t=x_t,
                        timestep=t.expand(count),
                        loop_config=runtime,
                        cfg_text_scale=scale,
                        **condition.inputs,
                    )
                    logs.extend(
                        {
                            "step": step,
                            "timestep": float(t),
                            "bin": timestep_bin(step, len(dts)),
                            **item,
                        }
                        for item in result.stats
                    )
                    if args.save_readouts and step in probe_steps:
                        per_round = [result.base_velocity] + result.velocities
                        torch.save(
                            [value.cpu() for value in per_round],
                            directory
                            / f"batch_{offset:04d}_step_{step:03d}_velocities.pt",
                        )
                        for r, velocity in enumerate(per_round):
                            estimates = (x_t - t * velocity).split(gen_counts)
                            for i, latent in enumerate(estimates):
                                inferencer.decode_image(
                                    latent, condition.shapes[i]
                                ).save(
                                    directory
                                    / f"{offset + i:04d}_step_{step:03d}_r{r}.png"
                                )
                    # Exactly one sampler update, after the final inner readout.
                    x_t = x_t - result.velocity * dt
                synchronize(device)
                elapsed = time.perf_counter() - started
                for i, latent in enumerate(x_t.split(gen_counts)):
                    path = directory / f"{offset + i:04d}.png"
                    inferencer.decode_image(latent, condition.shapes[i]).save(path)
                    records.append(
                        {
                            "index": offset + i,
                            "prompt": batch[i]["prompt"],
                            "arm": arm,
                            "path": str(path.resolve()),
                            "seed": args.seed + offset,
                            "image_shape": condition.shapes[i],
                            "initial_noise_sha256": noise_hashes[i],
                            "elapsed_batch_seconds": elapsed,
                            "semantics": arm_semantics(runtime),
                            "depth_status": depth_status,
                            "benchmark_record": batch[i],
                        }
                    )
                (directory / f"batch_{offset:04d}_loop_logs.json").write_text(
                    json.dumps(logs, indent=2)
                )
                print(f"Generated {arm}: prompts {offset + 1}-{offset + len(batch)}/{len(rows)}", flush=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "allocated_loop_config": config.to_dict(),
                "images": records,
                "arms": arm_metadata,
                "checkpoint_training_coverage": metadata.get(
                    "round_training_steps", "unknown"
                )
                if args.checkpoint
                else "training_free",
                "memory_control_semantics": "fixed_donor_read_only_with_correct_canonical_writer",
                "loop_diagnostics_enabled": config.log_loop_stats,
                "readout_probes_saved": args.save_readouts,
                "compute_note": "Anchored/direct branches run native reference plus R extra bodies; shuffled uses a separate correct writer body each round. Enabled legacy diagnostics add a native reference pass. Timing includes probes only when enabled; FLOPs require accelerator profiling.",
            },
            indent=2,
        )
    )
    return backbone


if __name__ == "__main__":
    main()
