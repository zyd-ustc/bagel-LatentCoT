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


def round_velocity_report(
    round_velocities: list[torch.Tensor], target: torch.Tensor, eps: float = 1e-6,
) -> dict:
    """Compare each round velocity with one flow target.

    Index 0 is the native velocity. Later indexes are correction rounds.
    ``error_fell`` means that round's mean square error is lower than the
    previous round. ``error_rose`` means it moved farther from this target.
    ``drift`` is the relative change from the native velocity. A round can
    move toward the target while its drift from the native velocity grows.
    This report measures those two changes. It does not say that feedback
    improves generation.
    """
    if not round_velocities:
        raise ValueError("round_velocities is empty")
    native = round_velocities[0]
    mse = []
    drift = []
    for velocity in round_velocities:
        if velocity.shape != target.shape:
            raise ValueError("velocity and target shapes differ")
        mse.append(float((velocity.detach().float() - target.detach().float()).pow(2).mean()))
        drift.append(relative_change(velocity, native, eps=eps))
    corrections = []
    for index in range(1, len(mse)):
        corrections.append({
            "round": index,
            "mse": mse[index],
            "drift": drift[index],
            "error_fell": mse[index] < mse[index - 1],
            "error_rose": mse[index] > mse[index - 1],
        })
    return {"mse": mse, "drift": drift, "corrections": corrections}
