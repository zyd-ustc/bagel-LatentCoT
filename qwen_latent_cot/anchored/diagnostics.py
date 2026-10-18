"""Per-step tensor records. These explain a run; they are not quality scores."""
from __future__ import annotations

import torch


def tensor_record(name: str, value: torch.Tensor | None) -> dict:
    if value is None:
        return {"name": name, "missing": True}
    detached = value.detach()
    return {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "max_abs": float(detached.float().abs().max()) if detached.numel() else 0.0,
        "rms": float(detached.float().pow(2).mean().sqrt()) if detached.numel() else 0.0,
    }


def relative_change(current: torch.Tensor, reference: torch.Tensor, eps: float = 1e-6) -> float:
    numerator = (current.detach().float() - reference.detach().float()).pow(2).mean().sqrt()
    denominator = reference.detach().float().pow(2).mean().sqrt()
    return float(numerator / (denominator + eps))
