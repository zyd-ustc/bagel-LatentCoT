"""Configuration loading and the isochronic-state contract."""

import json
import math
from pathlib import Path

from . import (
    CONDITION_PROTOCOL, KIND, PROTOCOL_VERSION, ROLE_R_ACTIVE, ROLE_R_EDIT,
    ROLE_R_REPAIR, ROLES,
)

# Every name in this set belonged to a previous training or inference design.
OLD_KEYS = frozenset(
    {
        "handoff_time",
        "rho_deep",
        "rho_shallow",
        "freeze_gen_tail",
        "window_steps",
        "tail_steps",
        "edit_times",
        "edit_scale",
        "proposal_proj",
        "start_embedding",
        "lambda_fm",
        "lambda_perceptual",
        "text_drop_prob",
        "lookahead_nodes",
        "history_roles",
        "native_edit",
        "history_edit",
        "history_keep",
        "continue_edit",
        "detach_image_lora",
        "edit_cfg",
    }
)

STAGE_WEIGHTS = {
    1: {"linear_pair"},
    2: {"linear_pair", "inverted_pair", "native_parent"},
}


def load_config(path):
    """Read a stage file, merge its shared core, and validate the result."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    parent = data.pop("extends", None)
    if parent:
        base = json.loads((path.parent / parent).read_text())
        if "extends" in base:
            raise ValueError("A shared core cannot extend another configuration")
        base.update(data)
        data = base
    validate_config(data)
    return data


def validate_config(cfg):
    old = sorted(OLD_KEYS.intersection(cfg))
    if old:
        raise ValueError("Old editor settings are not part of state flow: " + ", ".join(old))
    if cfg.get("kind") != KIND or cfg.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Isochronic state flow requires its own configuration")

    for key in ("width", "height"):
        if not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if cfg["width"] % 8 or cfg["height"] % 8:
        raise ValueError("The canvas must support the native patch grid")
    if int(cfg.get("num_steps", 0)) < 1:
        raise ValueError("num_steps must be positive")
    if float(cfg.get("timestep_shift", 0)) <= 0:
        raise ValueError("timestep_shift must be positive")
    if not 0 < float(cfg.get("t_eps", 0)) < 1:
        raise ValueError("t_eps must lie strictly between zero and one")

    if cfg.get("cfg_norm") != "none":
        raise ValueError("The isochronic contract uses cfg_norm=none")
    if float(cfg.get("image_cfg_scale", 0)) != 1:
        raise ValueError("The image condition is the only condition branch")
    if float(cfg.get("cfg_scale", 0)) != 4:
        raise ValueError("Native T2I is frozen at CFG=4")
    lo, hi = [float(t) for t in cfg.get("cfg_interval", [-1, -1])]
    if not 0 <= lo < hi <= 1:
        raise ValueError("Invalid CFG interval")

    read = float(cfg.get("read_time", -1))
    times = [float(t) for t in cfg.get("state_times", [])]
    probs = [float(p) for p in cfg.get("state_time_probs", [])]
    if not 0 < read < 1:
        raise ValueError("read_time must lie strictly inside the trajectory")
    if not times or any(not 0 < t < 1 for t in times):
        raise ValueError("state_times must be non-empty interior times")
    if min(times) < 0.15:
        raise ValueError("The first version supports state_times at or above 0.15")
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("state_times must be strictly increasing")
    if len(probs) != len(times) or abs(sum(probs) - 1) > 1e-6:
        raise ValueError("state_time_probs must match state_times and sum to one")
    grid_times = [float(t) for t in cfg.get("phase_grid_times", times)]
    if any(not 0 < t < 1 for t in grid_times) or not set(times).issubset(grid_times):
        raise ValueError("phase_grid_times must contain the sampled state times")
    if read not in times:
        raise ValueError("read_time must be one of the supported state times")

    if int(cfg.get("edit_steps", 0)) < 1:
        raise ValueError("edit_steps must be positive")
    if int(cfg.get("lookahead_steps", 0)) != 2:
        raise ValueError("The preview uses two native intervals")
    if int(cfg.get("lora_rank", 0)) < 1:
        raise ValueError("LoRA rank must be positive")
    if not 0 <= float(cfg.get("keep_fraction", -1)) < 1:
        raise ValueError("keep_fraction must lie in [0, 1)")
    keep = str(cfg.get("keep_instruction", "")).strip()
    if not keep:
        raise ValueError("keep_instruction must be non-empty")
    if cfg.get("condition_protocol") != CONDITION_PROTOCOL:
        raise ValueError("condition_protocol must use the frozen template version")
    if not isinstance(cfg.get("allow_derived_tail", False), bool):
        raise ValueError("allow_derived_tail must be a boolean")

    stage = cfg.get("stage")
    if stage is not None:
        if int(stage) not in STAGE_WEIGHTS:
            raise ValueError("stage must be 1 or 2")
        if int(cfg.get("global_batch", 0)) < 1:
            raise ValueError("global_batch must be positive")
        if int(cfg.get("save_every", 0)) < 1:
            raise ValueError("save_every must be positive")
        if float(cfg.get("gradient_clip", 0)) <= 0:
            raise ValueError("gradient_clip must be positive")
        if float(cfg.get("lr", 0)) <= 0 or float(cfg.get("lr_final", -1)) <= 0:
            raise ValueError("Stage learning rates must be positive")
        if float(cfg["lr_final"]) > float(cfg["lr"]):
            raise ValueError("The schedule may only decay")
        if int(cfg.get("warmup_steps", -1)) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if int(cfg.get("max_steps", 0)) < 1:
            raise ValueError("max_steps must be positive")
        if int(cfg.get("updates_per_pair", 0)) < 1:
            raise ValueError("updates_per_pair must be positive")
        weights = cfg.get("source_weights", {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError("source_weights must be a non-empty mapping")
        allowed = STAGE_WEIGHTS[int(stage)]
        if set(weights) - allowed:
            raise ValueError("source_weights contains a source outside this stage")
        if abs(sum(float(v) for v in weights.values()) - 1) > 1e-6:
            raise ValueError("source_weights must sum to one")
        if any(float(v) < 0 for v in weights.values()):
            raise ValueError("source_weights must be non-negative")
        if int(stage) == 1 and set(weights) != {"linear_pair"}:
            raise ValueError("Stage 1 trains on linear pairs only")
        if int(stage) == 2 and float(weights.get("native_parent", 0)) <= 0:
            raise ValueError("Stage 2 must include native parent pairs")

    roles = cfg.get("role_weights")
    if roles is not None:
        if not isinstance(roles, dict) or not roles:
            raise ValueError("role_weights must be a non-empty mapping")
        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise ValueError("role_weights contains an unknown role: " + ", ".join(unknown))
        values = [float(value) for value in roles.values()]
        if any(value < 0 for value in values):
            raise ValueError("role_weights must be non-negative")
        if abs(sum(values) - 1) > 1e-6:
            raise ValueError("role_weights must sum to one")
        if float(cfg.get("keep_fraction", 0)) != 0:
            raise ValueError("role_weights replaces keep_fraction; set keep_fraction to zero")
        if stage is not None and int(stage) == 2:
            # Any one real-history edit pool can carry stage 2: a directly
            # specified edit, a repair of a real failure, or an active edit.
            edit_roles = (ROLE_R_EDIT, ROLE_R_REPAIR, ROLE_R_ACTIVE)
            if not any(float(roles.get(role, 0)) > 0 for role in edit_roles):
                raise ValueError(
                    "Stage 2 role weights must include one of " + ", ".join(edit_roles))

    semantic = cfg.get("semantic_weights")
    if semantic is not None:
        tasks = {"count_increase", "count_reduce", "spatial", "attribute", "keep", "satisfied"}
        if (not isinstance(semantic, dict) or not semantic or set(semantic) - tasks
                or any(not math.isfinite(float(w)) or float(w) < 0 for w in semantic.values())
                or not math.isclose(sum(map(float, semantic.values())), 1.0)):
            raise ValueError("semantic_weights must assign normalized weights to semantic tasks")
        if roles or cfg["keep_fraction"] or stage != 2:
            raise ValueError("semantic_weights require Stage 2 without role_weights or keep_fraction")
        if not 0 <= float(cfg.get("repair_cap", -1)) <= 1:
            raise ValueError("repair_cap must lie between zero and one")
        terminal_options = cfg.get("terminal") or {}
        if not (terminal_options.get("enabled") and terminal_options.get("region_supervision")):
            raise ValueError("Semantic Stage 2 requires region supervision")

    terminal = cfg.get("terminal")
    if terminal is not None:
        if not isinstance(terminal, dict):
            raise ValueError("terminal must be a mapping")
        if terminal.get("enabled"):
            for key in ("preserve_weight", "frequency_weight", "pixel_weight"):
                if float(terminal.get(key, 0)) < 0:
                    raise ValueError("terminal." + key + " must be non-negative")
            low = float(terminal.get("mask_low", 0.6))
            high = float(terminal.get("mask_high", 0.95))
            if not 0 <= low < high <= 1:
                raise ValueError("terminal mask quantiles must satisfy 0 <= low < high <= 1")
            if float(cfg.get("recovery_prob", 0)) > 0:
                raise ValueError("Terminal supervision replaces the recovery bridge")
            named = terminal.get("roles")
            if named is not None:
                if not isinstance(named, list) or not named:
                    raise ValueError("terminal.roles must be a non-empty list")
                unknown = sorted(set(named) - set(ROLES))
                if unknown:
                    raise ValueError("terminal.roles has an unknown role: " + ", ".join(unknown))

    start = int(cfg.get("recovery_start_step", 0))
    probability = float(cfg.get("recovery_prob", 0.0))
    recovery_step = float(cfg.get("recovery_step", 0.0))
    if start < 0:
        raise ValueError("recovery_start_step must be non-negative")
    if not 0 <= probability < 1:
        raise ValueError("recovery_prob must lie in [0, 1)")
    if probability > 0:
        if not 0 < recovery_step < 1:
            raise ValueError("recovery_step must lie in (0, 1)")
        subdivision = round(1 / recovery_step)
        if subdivision < 2 or abs(subdivision * recovery_step - 1) > 1e-9:
            raise ValueError("recovery_step must divide one into equal steps, e.g. 0.125")

    inversion = cfg.get("inversion", {})
    if inversion:
        solver = str(inversion.get("solver", "reverse_heun"))
        if solver not in ("reverse_heun", "discrete_fixpoint"):
            raise ValueError("inversion.solver must be reverse_heun or discrete_fixpoint")
        if int(inversion.get("substeps", 0)) < 1:
            raise ValueError("Inversion substeps must be positive")
        if solver == "discrete_fixpoint":
            if int(inversion.get("iterations", 0)) < 1:
                raise ValueError("Inversion iterations must be positive")
            damping = float(inversion.get("damping", 0.0))
            if not 0 < damping <= 1:
                raise ValueError("Inversion damping must lie in (0, 1]")
            if float(inversion.get("max_residual", 0)) <= 0:
                raise ValueError("max_residual must be positive")
        for key in ("source_l1_max", "target_l1_max"):
            if float(inversion.get(key, 1)) <= 0:
                raise ValueError(key + " must be positive")
        if float(inversion.get("target_lpips_max", 1)) <= 0:
            raise ValueError("target_lpips_max must be positive")
    return cfg


def stage_updates(records, cfg, global_batch=None):
    """Eight passes over independent records at the actual global batch."""
    records = int(records)
    if records < 1:
        raise ValueError("A stage needs at least one independent training record")
    batch = int(global_batch or cfg["global_batch"])
    if batch < 1:
        raise ValueError("global_batch must be positive")
    return min(
        int(cfg["max_steps"]),
        math.ceil(int(cfg["updates_per_pair"]) * records / batch),
    )
