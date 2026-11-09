"""Stage checkpoints: strict same-stage resume and weight-only stage handoff."""

import json
import random
from pathlib import Path

import torch

from . import KIND
from . import distributed as dist


def canonical_parameters(module):
    return {
        name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def editor_identity(rt, cfg):
    """The frozen interface every checkpoint must match before weights are loaded."""
    return dict(
        native=rt.identity,
        interface=KIND,
        parameter_layout="gen_lora_all_layers+edit_s_embedder",
        condition_protocol=cfg["condition_protocol"],
    )


def save(
    path, editor, optimizer, sampler, step, cfg, identity, device,
    *, data_version, input_domains, total_steps, budget,
):
    path = Path(path)
    if dist.rank() == 0:
        path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    partial = path / f"rank_{dist.rank():05d}.partial"
    if dist.rank() == 0:
        payload = dict(
            weights=cpu(canonical_parameters(editor)),
            optimizer=cpu(optimizer.state_dict()),
            sampler=sampler.state_dict(),
            rng=rng(device),
        )
    else:
        # Replicated training: weights and optimizer are identical on all
        # ranks. Only the rank-specific RNG/sampler state needs a copy.
        payload = dict(sampler=sampler.state_dict(), rng=rng(device))
    torch.save(payload, partial)
    partial.replace(partial.with_suffix(".pt"))
    dist.barrier()
    if dist.rank() == 0:
        meta = dict(
            kind=KIND,
            stage=int(cfg["stage"]),
            step=int(step),
            world=dist.world_size(),
            config=cfg,
            identity=identity,
            parameter_layout="gen_lora_all_layers+edit_s_embedder",
            data_version=data_version,
            input_domains=list(input_domains),
            total_steps=int(total_steps),
            budget=int(budget),
        )
        (path / "manifest.partial").write_text(json.dumps(meta, indent=2) + "\n")
        (path / "manifest.partial").replace(path / "manifest.json")
    dist.barrier()


def _manifest(path):
    return json.loads((Path(path) / "manifest.json").read_text())


def _copy_weights(path, editor, *, rank_file):
    params = canonical_parameters(editor)
    state = torch.load(
        str(Path(path) / rank_file),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    weights = state["weights"]
    if params.keys() != weights.keys():
        missing = sorted(params.keys() ^ weights.keys())
        raise ValueError("Editor parameter inventory differs: " + ", ".join(missing[:8]))
    with torch.no_grad():
        for name, parameter in params.items():
            if parameter.shape != weights[name].shape:
                raise ValueError("Parameter shape differs: " + name)
            parameter.copy_(weights[name].to(parameter))
    return state


def resume(path, editor, optimizer, sampler, cfg, identity, *, data_version, total_steps):
    """Resume the same stage; cross-stage loading must use load_stage_weights."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    if int(meta.get("stage", -1)) != int(cfg["stage"]):
        raise ValueError("Strict resume cannot cross stages")
    if int(meta.get("world", -1)) != dist.world_size():
        raise ValueError("Checkpoint world size differs")
    if meta.get("config") != cfg or meta.get("identity") != identity:
        raise ValueError("Checkpoint config or identity differs")
    if meta.get("data_version") != data_version:
        raise ValueError("Checkpoint data version differs")
    if int(meta.get("total_steps", -1)) != int(total_steps):
        raise ValueError("Checkpoint total-step budget differs")
    # Weights/optimizer live in rank 0. Each rank restores only its own
    # RNG/sampler state; older checkpoints may still carry full per-rank files.
    base = _copy_weights(path, editor, rank_file="rank_00000.pt")
    optimizer.load_state_dict(base["optimizer"])
    if dist.rank() == 0:
        own = base
    else:
        own = torch.load(
            str(Path(path) / f"rank_{dist.rank():05d}.pt"),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    sampler.load_state_dict(own["sampler"])
    restore_rng(own["rng"], next(editor.parameters()).device)
    return meta, base


def load_stage_weights(path, editor, identity):
    """Load only trainable weights for Stage 2 or sampling."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    stored = meta.get("identity", {})
    for key in ("native", "interface", "parameter_layout", "condition_protocol"):
        if stored.get(key) != identity.get(key):
            raise ValueError("Checkpoint identity differs at " + key)
    _copy_weights(path, editor, rank_file="rank_00000.pt")
    return meta


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu(item) for item in value)
    return value


def rng(device):
    return dict(
        cpu=torch.get_rng_state(),
        python=random.getstate(),
        device=getattr(torch, device.type).get_rng_state(device).cpu() if device.type != "cpu" else None,
    )


def restore_rng(saved, device):
    torch.set_rng_state(saved["cpu"])
    random.setstate(saved["python"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(saved["device"], device)
