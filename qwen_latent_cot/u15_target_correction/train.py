"""One single-step IFM update per draw, with explicit stage budgets."""

import json
import math
import time
from pathlib import Path

import torch

from . import INVERTED_PAIR, LINEAR_PAIR, NATIVE_PARENT
from . import distributed as dist
from .checkpoint import canonical_parameters, editor_identity, load_stage_weights, resume, save
from .config import stage_updates
from .data import LinearManifest, StatePairManifest, TrainingSampler, data_version as hash_data
from .field import NativeT2I, phase_grid, preview_image
from .loading import load_native
from .loss import state_flow_matching
from .model import StateEditor


def peak_memory_gib(device):
    module = getattr(torch, device.type, None)
    if module is None or not hasattr(module, "max_memory_allocated"):
        return 0.0
    return float(module.max_memory_allocated(device)) / (1024**3)


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
    native = NativeT2I(editor.rt, sample["parent_condition"], cfg)
    preview = preview_image(anchor, time, grid, native)
    return anchor, destination, preview, sample["valid"], native.calls


def _cached_fields(sample):
    return (
        sample["state_a"],
        sample["state_b"],
        sample["preview"],
        sample["valid"],
        0,
    )


def _update(editor, sample, cfg, grid):
    if sample["source_kind"] == LINEAR_PAIR:
        anchor, destination, preview, valid, preview_calls = _linear_fields(
            sample, editor, cfg, grid
        )
    else:
        anchor, destination, preview, valid, preview_calls = _cached_fields(sample)
    time = float(sample["time"])
    prefixes = editor.prefixes(
        preview, sample["edit_request"], sample["tail_condition"], sample["preserve"]
    )
    if sample["base_prediction"] is None:
        base = editor.image_clean(anchor, time, prefixes["image"])
    else:
        base = sample["base_prediction"].detach()
    progress = float(torch.rand((), device=anchor.device))
    state = (1 - progress) * anchor + progress * destination
    target_speed = destination - anchor
    predicted_speed = editor.state_velocity(state, time, progress, prefixes["joint"], base)
    loss = state_flow_matching(predicted_speed, target_speed, time, valid)
    return loss, dict(
        time=time,
        progress=progress,
        preview_calls=preview_calls,
        baseline_calls=1 if sample["base_prediction"] is None else 0,
        state_calls=1,
        keep=int(bool(sample["keep"])),
    )


def train(args, cfg, device):
    if int(cfg["stage"]) == 1:
        if args.inverted or args.native:
            raise ValueError("Stage 1 trains on linear pairs only")
    if int(cfg["stage"]) == 2 and not args.inverted:
        raise ValueError("Stage 2 requires the inverted-pair cache")

    out = Path(args.output).resolve()
    if dist.rank() == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()

    rt = load_native(args.model, args.source, cfg, device, args.distributed)
    linear = LinearManifest(args.linear, rt, cfg)
    inverted = StatePairManifest(args.inverted, rt, cfg, INVERTED_PAIR) if args.inverted else None
    native = StatePairManifest(args.native, rt, cfg, NATIVE_PARENT) if args.native else None

    if int(cfg["stage"]) == 1:
        independent = linear.count
    else:
        independent = (inverted.count if inverted else 0) + (native.count if native else 0)
    budget = stage_updates(independent, cfg)
    steps = int(args.steps) if args.steps else budget
    if steps < 1:
        raise ValueError("The stage requires at least one update")

    sampler = TrainingSampler(linear, inverted, native, cfg, dist.rank())
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
    version = hash_data(linear.path, inverted.path if inverted else None, native.path if native else None)

    start = 0
    if args.resume:
        meta, _ = resume(args.resume, editor, optimizer, sampler, cfg, identity)
        start = int(meta["step"])
    elif args.init_from:
        load_stage_weights(args.init_from, editor, identity)

    log = out / "train.jsonl"
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
        )
        if dist.rank() == 0:
            with log.open("a") as handle:
                handle.write(json.dumps(stats) + "\n")
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
                )
            )
            + "\n"
        )
