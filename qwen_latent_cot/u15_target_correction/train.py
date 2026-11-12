"""One single-step IFM update per draw, with explicit stage budgets."""

import json
import math
import os
import time
from pathlib import Path

import torch

from . import INVERTED_PAIR, LINEAR_PAIR, NATIVE_PARENT, TERMINAL_ROLES
from . import distributed as dist
from .checkpoint import canonical_parameters, editor_identity, load_stage_weights, resume, save
from .config import stage_updates
from .data import LinearManifest, StatePairManifest, TrainingSampler, data_version as hash_data
from .field import NativeT2I, edit_state, euler, phase_grid, preview_image, segment_times
from .loading import load_native
from .loss import state_flow_matching
from .model import StateEditor
from .terminal import loss_options, terminal_loss


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
                inverted=str(args.inverted) if args.inverted else None,
                native=str(args.native) if args.native else None,
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


def _fields(sample, editor, cfg, grid):
    """The native pair and preview for one sample, fresh or cached."""
    if sample["source_kind"] == LINEAR_PAIR:
        return _linear_fields(sample, editor, cfg, grid)
    return _cached_fields(sample)


def _field_update(editor, anchor, destination, time, joint, base, valid):
    """One standard IFM sample: predict the paired displacement at q_s."""
    progress = float(torch.rand((), device=anchor.device))
    state = (1 - progress) * anchor + progress * destination
    predicted = editor.state_velocity(state, time, progress, joint, base)
    return state_flow_matching(predicted, destination - anchor, time, valid), progress


def _recovery_update(editor, anchor, destination, time, joint, base, valid, cfg):
    """Score the remaining displacement after one stop-gradient student step.

    The student walks its own state, so the recovery label covers the drift the
    analytic interpolation never shows. Only the recovery prediction is trained.
    """
    step = float(cfg["recovery_step"])
    index = int(torch.randint(0, int(round(1 / step)) - 1, (), device=anchor.device))
    progress = index * step
    state = (1 - progress) * anchor + progress * destination
    with torch.no_grad():
        walked = state + step * editor.state_velocity(state, time, progress, joint, base)
    remaining = progress + step
    predicted = editor.state_velocity(walked, time, remaining, joint, base)
    loss = state_flow_matching(
        (1 - remaining) * predicted, destination - walked, time, valid
    )
    return loss, progress


def _terminal_update(editor, sample, cfg, grid):
    """Score the finished picture through the frozen continuation.

    The target state is a latent here: nothing is compared in state space, so a
    label state neither has to be inverse solved nor be the unique answer. The
    gradient reaches the editor through the deployment tail.
    """
    anchor, _, preview, valid, preview_calls = _fields(sample, editor, cfg, grid)
    if sample["source_kind"] == LINEAR_PAIR:
        x_source = sample["source"]
        x_teacher = x_source if sample["keep"] else sample["target"]
    else:
        x_source, x_teacher = sample["x_source"], sample["x_teacher"]
    if x_source is None or x_teacher is None:
        raise ValueError("Terminal supervision needs the source and label images")
    time = float(sample["time"])
    prefixes = editor.prefixes(
        preview, sample["edit_request"], sample["tail_condition"], sample["preserve"]
    )
    base = editor.image_clean(anchor, time, prefixes["image"])
    edited, _ = edit_state(
        editor, anchor, time, prefixes, int(cfg["edit_steps"]), base, grad=True
    )
    velocity = NativeT2I(editor.rt, sample["tail_condition"], cfg)
    picture = euler(edited.float(), segment_times(grid, time, 1.0), velocity)
    options = loss_options(cfg["terminal"])
    if cfg["terminal"].get("region_supervision"):
        options.update(
            regions=sample["edit_regions"],
            no_change=sample["keep"] or sample.get("satisfied", False),
            perceptual=editor.terminal_perceptual,
        )
    loss, detail = terminal_loss(picture, x_teacher, x_source, valid, **options)
    return loss, dict(
        time=time, progress=1.0, mode="terminal",
        role=sample.get("role", sample["source_kind"]),
        preview_calls=preview_calls, baseline_calls=1,
        state_calls=int(cfg["edit_steps"]), tail_calls=velocity.calls,
        keep=int(bool(sample["keep"])), **detail,
    )


def _terminal_roles(cfg):
    """Roles trained through the frozen tail. Linear pairs stay on state flow."""
    terminal = cfg.get("terminal") or {}
    if not terminal.get("enabled"):
        return frozenset()
    named = terminal.get("roles")
    return frozenset(named) if named else TERMINAL_ROLES


def _update(editor, sample, cfg, grid, recovery=False):
    if sample.get("role") in _terminal_roles(cfg):
        return _terminal_update(editor, sample, cfg, grid)
    anchor, destination, preview, valid, preview_calls = _fields(sample, editor, cfg, grid)
    time = float(sample["time"])
    prefixes = editor.prefixes(
        preview, sample["edit_request"], sample["tail_condition"], sample["preserve"]
    )
    if sample["base_prediction"] is None:
        base = editor.image_clean(anchor, time, prefixes["image"])
    else:
        base = sample["base_prediction"].detach()
    joint = prefixes["joint"]
    if recovery:
        loss, progress = _recovery_update(
            editor, anchor, destination, time, joint, base, valid, cfg
        )
    else:
        loss, progress = _field_update(
            editor, anchor, destination, time, joint, base, valid
        )
    return loss, dict(
        time=time,
        progress=progress,
        mode="recovery" if recovery else "ifm",
        role=sample.get("role", sample["source_kind"]),
        preview_calls=preview_calls,
        baseline_calls=1 if sample["base_prediction"] is None else 0,
        state_calls=2 if recovery else 1,
        keep=int(bool(sample["keep"])),
    )


def train(args, cfg, device):
    terminal = bool((cfg.get("terminal") or {}).get("enabled", False))
    if int(cfg["stage"]) == 1 and not terminal:
        if args.inverted or args.native:
            raise ValueError("Stage 1 trains on linear pairs only")
    if (
        int(cfg["stage"]) == 2
        and not args.inverted
        and not terminal
        and not cfg.get("role_weights")
    ):
        # A role-weighted run names its own pools; requiring the I cache
        # would block the state-loss control arm, which trains on R-edit only.
        raise ValueError("Stage 2 requires the inverted-pair cache")

    out = Path(args.output).resolve()
    if dist.rank() == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()

    rt = load_native(args.model, args.source, cfg, device, args.distributed)
    linear = LinearManifest(args.linear, rt, cfg) if args.linear else None
    inverted = StatePairManifest(args.inverted, rt, cfg, INVERTED_PAIR) if args.inverted else None
    native = StatePairManifest(args.native, rt, cfg, NATIVE_PARENT) if args.native else None

    if int(cfg["stage"]) == 1:
        if linear is None:
            raise ValueError("Stage 1 requires --linear")
        independent = linear.count
    else:
        independent = (inverted.count if inverted else 0) + (native.count if native else 0)
    budget = stage_updates(independent, cfg, global_batch=dist.world_size())
    steps = int(args.steps) if args.steps else budget
    if steps < 1:
        raise ValueError("The stage requires at least one update")

    sampler = TrainingSampler(linear, inverted, native, cfg, dist.rank())
    grid = phase_grid(rt, cfg)
    torch.manual_seed(int(cfg["seed"]) + dist.rank())
    editor = StateEditor(rt, cfg).train()
    dist.broadcast_parameters(editor)
    if (cfg.get("terminal") or {}).get("region_supervision"):
        from .loss import build_perceptual

        object.__setattr__(editor, "terminal_perceptual", build_perceptual(rt.device))
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
    version = hash_data(linear.path if linear else None, inverted.path if inverted else None, native.path if native else None)

    start = 0
    if args.resume:
        meta, _ = resume(
            args.resume, editor, optimizer, sampler, cfg, identity,
            data_version=version, total_steps=steps,
        )
        start = int(meta["step"])
    elif args.init_from:
        load_stage_weights(args.init_from, editor, identity)

    log = out / "train.jsonl"
    run = _start_wandb(args, cfg, out, version, budget, steps)
    for step in range(start + 1, steps + 1):
        began = time.perf_counter()
        rate = learning_rate(step, cfg, steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        sample = sampler.draw()
        recovery = (
            step > int(cfg.get("recovery_start_step", 0))
            and float(cfg.get("recovery_prob", 0.0)) > 0
            and float(torch.rand(())) < float(cfg["recovery_prob"])
        )
        loss, fields = _update(editor, sample, cfg, grid, recovery=recovery)
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
