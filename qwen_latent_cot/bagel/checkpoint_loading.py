"""Strict checkpoint projection for deliberately reduced model variants."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import nn


def project_complete_state_dict(
    module: nn.Module,
    source: Mapping[str, torch.Tensor],
    *,
    source_name: str,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Select every tensor required by ``module`` and ignore disabled branches."""
    target = module.state_dict()
    missing = sorted(key for key in target if key not in source)
    mismatched = sorted(
        key
        for key in target
        if key in source and tuple(source[key].shape) != tuple(target[key].shape)
    )
    if missing or mismatched:
        raise RuntimeError(
            f"Checkpoint is incomplete for the active model: source={source_name} "
            f"missing={missing[:8]} mismatched={mismatched[:8]}"
        )
    selected = {key: source[key] for key in target}
    ignored = sum(1 for key in source if key not in target)
    return selected, ignored


__all__ = ["project_complete_state_dict"]
