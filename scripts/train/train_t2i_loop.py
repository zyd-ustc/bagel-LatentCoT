#!/usr/bin/env python3
"""Stage 1 only. Later workspace/body stages require measured experiment gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
from contextlib import nullcontext
from dataclasses import replace

import torch
import yaml
from torch.utils.data import DataLoader
from torch import distributed as dist

from qwen_latent_cot.bagel import BagelBackbone, LoopConfig
from qwen_latent_cot.bagel.accelerator import (
    autocast_for,
    manual_seed_all,
)
from qwen_latent_cot.bagel.depth_curriculum import (
    resolve_depth_curriculum,
)
from qwen_latent_cot.bagel.flow_time import sample_native_flow_timestep
from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
from qwen_latent_cot.bagel.loop_checkpoint import save_loop_checkpoint
from qwen_latent_cot.bagel.stage1_distributed import (
    gather_rank_metrics,
    initialize_stage1_distributed,
    stage1_objective,
    synchronized_depth,
    token_weighted_backward_loss,
)
from qwen_latent_cot.data.t2i import (
    BucketBatchSampler,
    T2IDataset,
    collate_t2i,
    patchify_latents,
    sample_flow_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    config = LoopConfig(**cfg["loop"])
    if (
        not config.enable_t2i_loop
        or config.runtime_loop_depth == 0
        or config.reentry_adapter_type != "low_rank"
    ):
        raise ValueError(
            "Stage 1 requires enabled extra loops with a trainable low_rank adapter"
        )
    if (
        config.loop_mode == "legacy_memory_only"
        or config.loop_mode.startswith("direct_native_")
        or config.memory_control != "correct"
    ):
        raise ValueError(
            "Stage 1 trains anchored states with correct workspace; use evaluation for negative controls"
        )
    if config.loop_gate_init == 0:
        raise ValueError(
            "Stage 1 needs positive loop_gate_init so output alpha can learn"
        )
    if config.loop_mode == "gen_only" and config.loop_output_alpha_init == 0:
        raise ValueError(
            "GEN-only needs nonzero loop_output_alpha_init (e.g. 0.01) to train the zero adapter bias; initial velocity remains native"
        )
    if int(cfg.get("save_every", 100)) < 1 or int(cfg.get("batch_size", 1)) < 1:
        raise ValueError("save_every and batch_size must be positive")
    total = int(cfg.get("steps", 1000))
    max_train_depth = int(cfg.get("max_train_loop_depth", config.runtime_loop_depth))
    if not 1 <= max_train_depth <= config.allocated_max_loop_depth:
        raise ValueError("training depth must lie within allocated_max_loop_depth")
    text_dropout = float(cfg.get("text_cond_dropout_prob", 0.1))
    if not 0 <= text_dropout <= 1:
        raise ValueError("text_cond_dropout_prob must lie in [0,1]")
    phases = resolve_depth_curriculum(
        cfg.get("depth_curriculum", list(range(1, max_train_depth + 1))),
        max_train_depth,
        total,
    )
    depth_counts = {str(depth): 0 for depth in range(1, max_train_depth + 1)}
    round_training_steps = [0] * config.allocated_max_loop_depth
    seed = int(cfg.get("seed", 0))
    manual_seed_all(seed)
    device, rank, world_size = initialize_stage1_distributed(cfg.get("device", "auto"))
    backbone = BagelBackbone(
        {
            "model_path": cfg["model_path"],
            "timestep_shift": float(cfg.get("timestep_shift", 1.0)),
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "t2i_loop": config.to_dict(),
        }
    ).load()
    model, vae = backbone.bagel.to(device), backbone.vae_model.to(device)
    # Packed inference retains autograd. Native weights and dropout stay frozen.
    model.eval()
    trainable_names = backbone.apply_stage1_policy()
    objective = stage1_objective(model, device, world_size)
    # The initial loop state is shared, but noise/dropout/time must differ.
    manual_seed_all(seed + rank)
    model.language_model.model.gradient_checkpointing = bool(
        cfg.get("gradient_checkpointing", True)
    )
    inferencer = InterleaveInferencer(
        model, vae, backbone.tokenizer, backbone.token_ids
    )
    size = int(cfg.get("image_size", 512))
    if size % model.latent_downsample:
        raise ValueError("image_size must be divisible by native latent_downsample")
    dataset = T2IDataset(
        cfg["data_path"],
        size,
        stride=model.latent_downsample,
        min_image_size=cfg.get("min_image_size"),
        max_pixels=cfg.get("max_pixels"),
    )
    sampler = BucketBatchSampler(
        dataset,
        int(cfg.get("batch_size", 1)),
        total,
        weights=cfg.get("bucket_weights"),
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_t2i)
    bucket_metrics = {}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    out = Path(cfg["output_dir"])
    distributed_training = {
        "world_size": world_size,
        "batch_size_per_rank": int(cfg.get("batch_size", 1)),
        "global_batch_size": int(cfg.get("batch_size", 1)) * world_size,
        "loss_reduction": "global_packed_token_mean",
        "depth_sampling": "rank0_broadcast",
    }
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "data_policy.json").write_text(
            json.dumps(
                {
                    "effective_bucket_weights": sampler.effective_weights,
                    "text_cond_dropout_prob": text_dropout,
                    "image_preprocessing": "native_bagel_resize_no_crop",
                    "distributed_training": distributed_training,
                },
                indent=2,
            )
        )
        (out / "trainable.json").write_text(json.dumps(trainable_names, indent=2))
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if world_size > 1:
        dist.barrier()
    step = 0
    log_context = (out / "metrics.jsonl").open("w") if rank == 0 else nullcontext(None)
    with log_context as log:
        while step < total:
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), autocast_for(device):
                    clean_parts = [
                        patchify_latents(
                            vae.encode(pixels.unsqueeze(0).to(device)),
                            model.latent_patch_size,
                        ).float()
                        for pixels in batch["pixels"]
                    ]
                    clean = torch.cat(clean_parts)
                    counts = [len(part) for part in clean_parts]
                    drops = (torch.rand(len(counts)) < text_dropout).tolist()
                    condition = inferencer.prepare_condition(
                        batch["prompt"], batch["image_shape"], text_drop_mask=drops
                    )
                    # Prompt caches are anchors, not data for an autograd graph.
                    inputs = {
                        key: value
                        for key, value in condition.inputs.items()
                        if not key.startswith("cfg_")
                    }
                depth, phase_index = synchronized_depth(
                    phases, step, device, rank, world_size
                )
                runtime = replace(config, runtime_loop_depth=depth)
                t = sample_native_flow_timestep(
                    device=device, timestep_shift=model.timestep_shift
                )
                noise = torch.randn_like(clean)
                x_t, target = sample_flow_state(clean, t, noise)
                with autocast_for(device):
                    loss, result = objective(
                        target=target,
                        x_t=x_t,
                        timestep=t.expand(clean.shape[0]),
                        loop_config=runtime,
                        **inputs,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite flow loss at step {step}")
                backward_loss = token_weighted_backward_loss(
                    loss, len(clean), device, world_size
                )
                backward_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(cfg.get("max_grad_norm", 1.0))
                )
                if not torch.isfinite(norm):
                    raise RuntimeError(f"nonfinite loop gradients at step {step}")
                optimizer.step()
                step += 1
                depth_counts[str(depth)] += 1
                for index in range(depth):
                    round_training_steps[index] += 1
                bucket = batch["bucket"][0]
                alpha = model.t2i_loop.output_alpha.detach().float().cpu().tolist()
                gates = (
                    model.t2i_loop.gate_logits.detach().sigmoid().float().cpu().tolist()
                )
                losses_by_condition = {"conditional": [], "text_removed": []}
                offset = 0
                for count, dropped in zip(counts, drops):
                    parts = result.velocities or [result.velocity]
                    losses = [
                        torch.nn.functional.mse_loss(
                            value[offset : offset + count].float(),
                            target[offset : offset + count].float(),
                        )
                        for value in parts
                    ]
                    sample_loss = losses[-1]
                    if runtime.loop_deep_supervision and len(losses) > 1:
                        sample_loss = (
                            sample_loss
                            + runtime.loop_ds_weight * torch.stack(losses[:-1]).mean()
                        )
                    losses_by_condition[
                        "text_removed" if dropped else "conditional"
                    ].append(float(sample_loss.detach()))
                    offset += count
                per_rank = gather_rank_metrics(
                    {
                        "rank": rank,
                        "indices": batch["index"],
                        "bucket": bucket,
                        "flow_loss": float(loss.detach()),
                        "tokens": len(clean),
                        "timestep": float(t),
                        "text_drop_mask": drops,
                        "condition_sample_losses": losses_by_condition,
                        "image_shapes": batch["image_shape"],
                        "original_shapes": batch["original_shape"],
                        "loop_stats": result.stats,
                    },
                    rank,
                    world_size,
                )
                if rank == 0:
                    global_loss = sum(
                        item["flow_loss"] * item["tokens"] for item in per_rank
                    ) / sum(item["tokens"] for item in per_rank)
                    metric = bucket_metrics.setdefault(
                        bucket, {"steps": 0, "loss_sum": 0.0}
                    )
                    metric["steps"] += 1
                    metric["loss_sum"] += global_loss
                    losses_by_condition = {
                        key: [
                            value
                            for item in per_rank
                            for value in item["condition_sample_losses"][key]
                        ]
                        for key in ("conditional", "text_removed")
                    }
                else:
                    global_loss = None
                record = {
                    "step": step,
                    "depth": depth,
                    "curriculum_phase": phase_index,
                    "allowed_depths": phases[phase_index]["depths"],
                    "depth_counts": dict(depth_counts),
                    "round_training_steps": list(round_training_steps),
                    "timestep": float(t),
                    "timestep_shift": model.timestep_shift,
                    "flow_loss": global_loss,
                    "distributed_training": distributed_training,
                    "per_rank": per_rank,
                    "loop_stats_rank": 0,
                    "loop_stats": result.stats,
                    "bucket": bucket,
                    "bucket_metrics": {
                        key: {
                            "steps": value["steps"],
                            "mean_step_loss": value["loss_sum"] / value["steps"],
                        }
                        for key, value in bucket_metrics.items()
                    },
                    "bucket_parameter_metrics": {
                        "bucket": bucket,
                        "alpha": alpha,
                        "gates": gates,
                    },
                    "text_drop_mask": drops,
                    "condition_losses": {
                        key: sum(value) / len(value) if value else None
                        for key, value in losses_by_condition.items()
                    },
                    "image_shapes": batch["image_shape"],
                    "original_shapes": batch["original_shape"],
                    "gates": gates,
                    "grad_norm": float(norm),
                    "alpha": model.t2i_loop.output_alpha.detach()
                    .float()
                    .cpu()
                    .tolist(),
                }
                if rank == 0:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                if step % int(cfg.get("save_every", 100)) == 0 or step == total:
                    if rank == 0:
                        save_loop_checkpoint(
                            model,
                            out / f"step_{step:06d}",
                            step=step,
                            model_path=cfg["model_path"],
                            training_depth_counts=depth_counts,
                            round_training_steps=round_training_steps,
                            depth_curriculum=phases,
                            distributed_training=distributed_training,
                        )
                    if world_size > 1:
                        dist.barrier()
                if step >= total:
                    break
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
