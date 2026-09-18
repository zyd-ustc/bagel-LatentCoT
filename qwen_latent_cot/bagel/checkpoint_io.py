"""Shared BAGEL checkpoint loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import torch


logger = logging.getLogger(__name__)


def load_finetuned_weights(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    *,
    require_complete: bool = False,
    required_keys: Sequence[str] = (),
    source_prefix: str = "",
) -> int:
    from safetensors.torch import load_file

    path = Path(checkpoint_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing BAGEL checkpoint model.safetensors: {path}")
    raw_state = load_file(str(path), device="cpu")
    model_state = model.state_dict()
    state: Dict[str, torch.Tensor] = {}
    prefix = str(source_prefix).strip(".")
    prefix_with_dot = f"{prefix}." if prefix else ""
    normalized_raw_keys = set()
    for raw_key, value in raw_state.items():
        key = raw_key.removeprefix("module.")
        if prefix_with_dot:
            if not key.startswith(prefix_with_dot):
                continue
            key = key[len(prefix_with_dot) :]
        normalized_raw_keys.add(key)
        candidates = [key, key.removeprefix("bagel.")]
        # Older adapters used condition_lift.* for the same input interface.
        # Do not silently replace that interface with a fresh random map.
        for candidate in tuple(candidates):
            if candidate.startswith("condition_lift."):
                candidates.append(
                    "dino_to_hidden." + candidate.removeprefix("condition_lift.")
                )
        target_key = next((candidate for candidate in candidates if candidate in model_state), None)
        if target_key is not None:
            state[target_key] = value

    input_interface_present = any(
        key.startswith(("dino_to_hidden.", "condition_lift."))
        for key in normalized_raw_keys
    )
    input_scale_present = any(
        key in {"dino_to_hidden.output_scale", "condition_lift.output_scale"}
        for key in normalized_raw_keys
    )
    scale_key = "dino_to_hidden.output_scale"
    if (
        scale_key in model_state
        and scale_key not in state
        and input_interface_present
        and not input_scale_present
    ):
        # Preserve the effective scale of checkpoints created before
        # output_scale was not persisted by older adapters; preserve their
        # effective scale when loading them into the current interface.
        state[scale_key] = torch.ones_like(model_state[scale_key])
        logger.info(
            "Legacy condition input interface has no output_scale; using 1.0"
        )
    for key, value in list(state.items()):
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            continue
        can_prefix_copy = (
            value.ndim == target.ndim == 2
            and int(value.shape[1]) == int(target.shape[1])
            and int(value.shape[0]) <= int(target.shape[0])
        )
        if not can_prefix_copy:
            logger.warning(
                "Skip incompatible checkpoint tensor %s: %s -> %s",
                key,
                tuple(value.shape),
                tuple(target.shape),
            )
            state.pop(key)
            continue
        merged = target.detach().cpu().clone()
        merged[: int(value.shape[0])] = value.to(dtype=merged.dtype)
        state[key] = merged
    missing_required = [str(key) for key in required_keys if str(key) not in state]
    if missing_required:
        raise RuntimeError(
            "Checkpoint is missing required interface tensors: "
            f"{missing_required} path={path}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        if bool(require_complete):
            raise RuntimeError(
                "Checkpoint is incomplete for the active BAGEL architecture: "
                f"missing={incompatible.missing_keys[:8]} "
                f"count={len(incompatible.missing_keys)} path={path}"
            )
        logger.warning(
            "Checkpoint has %d missing model tensors",
            len(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Checkpoint has %d unexpected tensors",
            len(incompatible.unexpected_keys),
        )
    return int(sum(tensor.numel() for tensor in state.values()))


_load_finetuned_weights = load_finetuned_weights


__all__ = ["load_finetuned_weights"]
