"""One single-step IFM update per draw, with explicit stage budgets."""

import json
import math
import os
import time
from pathlib import Path

import torch

from . import LINEAR_PAIR
from . import distributed as dist
from .checkpoint import canonical_parameters, editor_identity, resume, save
from .config import stage_updates
from .data import LinearManifest, TrainingSampler, data_version as hash_data
from .field import phase_grid
from .loading import load_native
from .loss import state_flow_matching
from .model import StateEditor


def peak_memory_gib(device):
    module = getattr(torch, device.type, None)
    if module is None or not hasattr(module, "max_memory_allocated"):
        return 0.0
    return float(module.max_memory_allocated(device)) / (1024**3)


def _start_wandb(args, cfg, out, version, budget, steps):
    """Rank-0-only tracking. Offline is the safe default on compute nodes."""
    if dist.rank() != 0:
        return None
    mode = os.environ.get("WANDB_MODE", "offline").strip().lower()
    if mode in ("disabled", "off", "none", "false"):
        return None
    try:
        import wandb
    except Exception:
        return None
    project = os.environ.get("WANDB_PROJECT", "isochronic-state-flow")
    entity = os.environ.get("WANDB_ENTITY") or None
    name = os.environ.get("WANDB_RUN_NAME") or f"{Path(args.output).name}-stage{cfg['stage']}"
    directory = os.environ.get("WANDB_DIR") or str(out / "wandb")
    tags = [item for item in os.environ.get("WANDB_TAGS", "").split(",") if item]
    try:
        return wandb.init(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            dir=directory,
            tags=tags or None,
            config=dict(
                cfg,
                data_version=version,
                linear=str(args.linear),
                inverted=None,
                native=None,
                budget=budget,
                steps=steps,
                world=dist.world_size(),
                global_batch=dist.world_size(),
                configured_global_batch=int(cfg["global_batch"]),
            ),
        )
    except Exception as error:
        print("wandb init failed; continuing without tracking: " + str(error), flush=True)
        return None


def learning_rate(step, cfg, total):
    low, high = float(cfg["lr_final"]), float(cfg["lr"])
    warmup = int(cfg["warmup_steps"])
    if warmup > 0 and step <= warmup:
        return low + (high - low) * step / warmup
    if total <= warmup:
        return low
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return low + (high - low) * 0.5 * (1 + math.cos(math.pi * progress))


def _linear_fields(sample, editor, cfg, grid):
    source, target = sample["source"], sample["target"]
    sigma = editor.rt.noise_scale()
    noise = torch.randn(source.shape, device=source.device, dtype=source.dtype)
    time = sample["time"]
    anchor = time * source + (1 - time) * sigma * noise
    destination = time * target + (1 - time) * sigma * noise
    # No lookahead: the editor reads a_t and the instruction, nothing else, so a
    # caption is not needed and no native preview forward has to be paid for.
    return anchor, destination, None, sample["valid"], 0


def _field_update(editor, anchor, destination, time, joint, base, valid):
    """One standard IFM sample: predict the paired displacement at q_s."""
    progress = float(torch.rand((), device=anchor.device))
    state = (1 - progress) * anchor + progress * destination
    predicted = editor.state_velocity(state, time, progress, joint, base)
    return state_flow_matching(predicted, destination - anchor, time, valid), progress


def _update(editor, sample, cfg, grid):
    anchor, destination, _, valid, preview_calls = _linear_fields(sample, editor, cfg, grid)
    time = float(sample["time"])
    prefixes = editor.prefixes(sample["edit_request"])
    base = editor.image_clean(anchor, time, prefixes["image"])
    loss, progress = _field_update(editor, anchor, destination, time, prefixes["joint"], base, valid)
    return loss, dict(time=time, progress=progress, mode="ifm", role=sample["role"],
                      preview_calls=preview_calls, baseline_calls=1, state_calls=1,
                      keep=int(bool(sample["keep"])))


def train(args, cfg, device):
    out = Path(args.output).resolve()
    if dist.rank() == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()

    rt = load_native(args.model, args.source, cfg, device, args.distributed)
    linear = LinearManifest(args.linear, rt, cfg)
    independent = linear.count
    budget = stage_updates(independent, cfg, global_batch=dist.world_size())
    steps = int(args.steps) if args.steps else budget
    if steps < 1:
        raise ValueError("The stage requires at least one update")

    sampler = TrainingSampler(linear, cfg, dist.rank())
    grid = phase_grid(rt, cfg)
    torch.manual_seed(int(cfg["seed"]) + dist.rank())
    editor = StateEditor(rt, cfg).train()
    dist.broadcast_parameters(editor)
    params = canonical_parameters(editor)
    optimizer = torch.optim.AdamW(
        params.values(),
        lr=float(cfg["lr"]),
        betas=tuple(float(v) for v in cfg["betas"]),
        eps=float(cfg["eps"]),
        weight_decay=float(cfg["weight_decay"]),
        foreach=False,
    )
    identity = editor_identity(rt, cfg)
    version = hash_data(linear.path)

    start = 0
    if args.resume:
        meta, _ = resume(
            args.resume, editor, optimizer, sampler, cfg, identity,
            data_version=version, total_steps=steps,
        )
        start = int(meta["step"])

    log = out / "train.jsonl"
    run = _start_wandb(args, cfg, out, version, budget, steps)
    for step in range(start + 1, steps + 1):
        began = time.perf_counter()
        rate = learning_rate(step, cfg, steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        sample = sampler.draw()
        loss, fields = _update(editor, sample, cfg, grid)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite state-flow loss")
        scaled = loss / dist.world_size()
        scaled.backward()
        dist.sum_gradients(editor, allow_unused=False)
        gradient = torch.nn.utils.clip_grad_norm_(list(params.values()), float(cfg["gradient_clip"]))
        if not torch.isfinite(gradient):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        stats = dict(
            step=step,
            lr=rate,
            loss=float(loss.detach()),
            gradient_norm=float(gradient),
            clipped=float(gradient) > float(cfg["gradient_clip"]),
            seconds=time.perf_counter() - began,
            source_kind=sample["source_kind"],
            role=fields["role"],
            semantic_task=sample.get("semantic_task"),
            count_direction=sample.get("count_direction"),
            mode=fields["mode"],
            uid=sample["uid"],
            keep=fields["keep"],
            time=fields["time"],
            progress=fields["progress"],
            preview_calls=fields["preview_calls"],
            baseline_calls=fields["baseline_calls"],
            state_calls=fields["state_calls"],
            input_domains=sampler.input_domains(),
            configured_weights=sampler.configured_weights,
            actual_weights=sampler.actual_weights,
            peak_memory_gib=peak_memory_gib(device),
            world=dist.world_size(),
            global_batch=dist.world_size(),
            configured_global_batch=int(cfg["global_batch"]),
            batch_mismatch=dist.world_size() != int(cfg["global_batch"]),
        )
        for key in ("structure", "perceptual", "preserve", "frequency", "hf_ratio", "edited_fraction", "tail_calls"):
            if key in fields:
                stats[key] = fields[key]
        if dist.rank() == 0:
            with log.open("a") as handle:
                handle.write(json.dumps(stats) + "\n")
            if run is not None:
                run.log(dict(stats, stage=int(cfg["stage"])), step=step)
        if step % int(cfg["save_every"]) == 0 or step == steps:
            save(
                out / f"step_{step:06d}",
                editor,
                optimizer,
                sampler,
                step,
                cfg,
                identity,
                device,
                data_version=version,
                input_domains=sampler.input_domains(),
                total_steps=steps,
                budget=budget,
            )
    if dist.rank() == 0:
        (out / "complete.json").write_text(
            json.dumps(
                dict(
                    stage=int(cfg["stage"]),
                    steps=steps,
                    budget=budget,
                    override=steps != budget,
                    input_domains=sampler.input_domains(),
                    configured_weights=sampler.configured_weights,
                    actual_weights=sampler.actual_weights,
                    data_version=version,
                    world=dist.world_size(),
                    global_batch=dist.world_size(),
                    configured_global_batch=int(cfg["global_batch"]),
                    batch_mismatch=dist.world_size() != int(cfg["global_batch"]),
                )
            )
            + "\n"
        )
        if run is not None:
            run.finish()
