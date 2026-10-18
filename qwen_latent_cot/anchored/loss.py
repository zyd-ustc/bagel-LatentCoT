"""Within-step losses. The anchor term is a small prior, not the objective.

``delta`` is the residual actually added at the body entry, ``Bound(gamma * A_phi)``.
``velocity`` is the sampler velocity, the last round. ``round_loss_weights``
replaces the flow term with one weight per correction. ``None`` keeps the
flow term on that final velocity. Weights may differ, and a zero weight stays
out of the graph. The anchor prior stays on the final velocity. The entry
term stays on the last written residual.
"""
from __future__ import annotations

import torch

from .diagnostics import round_velocity_report


def _flow_mse(velocity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if velocity.shape != target.shape:
        raise ValueError("velocity and target shapes differ")
    return (velocity.float() - target.float()).pow(2).mean()


def anchored_loss(
    velocity: torch.Tensor,
    target: torch.Tensor,
    native_velocity: torch.Tensor,
    delta: torch.Tensor | None,
    anchor: torch.Tensor | None,
    vae_indexes: torch.Tensor | None,
    *,
    lambda_anchor: float,
    lambda_entry: float,
    eps: float = 1e-6,
    round_velocities: list[torch.Tensor] | None = None,
    round_loss_weights: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, dict]:
    if round_loss_weights is None:
        final = _flow_mse(velocity, target)
    else:
        weights = tuple(float(weight) for weight in round_loss_weights)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("round loss weights must be non-negative")
        if round_velocities is None or len(round_velocities) != len(weights) + 1:
            raise ValueError(
                "round_loss_weights needs the native velocity plus one velocity per correction"
            )
        if not torch.equal(velocity, round_velocities[-1]):
            raise ValueError("velocity must be the last round velocity")
        final = None
        for weight, pred in zip(weights, round_velocities[1:]):
            term = _flow_mse(pred, target)
            if weight == 0.0:
                continue
            weighted = term if weight == 1.0 else weight * term
            final = weighted if final is None else final + weighted
        if final is None:
            final = target.float().new_zeros(())
    parts = {"final": final.detach()}
    total = final
    native = native_velocity.detach()
    anchor_term = (velocity.float() - native.float()).pow(2).mean() / (native.float().pow(2).mean() + eps)
    parts["anchor"] = anchor_term.detach()
    if lambda_anchor != 0.0:
        total = total + lambda_anchor * anchor_term
    if delta is None or anchor is None or vae_indexes is None:
        entry = velocity.new_zeros(())
    else:
        raw = delta[vae_indexes].float()
        base = anchor[vae_indexes].float().detach()
        entry = (raw.pow(2).sum(-1) / (base.pow(2).sum(-1) + eps)).mean()
    parts["entry"] = entry.detach()
    if lambda_entry != 0.0:
        total = total + lambda_entry * entry
    if round_velocities is not None:
        report = round_velocity_report(round_velocities, target, eps=eps)
        for index, value in enumerate(report["mse"]):
            parts[f"mse_{index}"] = value
        for item in report["corrections"]:
            parts[f"drift_{item['round']}"] = item["drift"]
            parts[f"error_fell_{item['round']}"] = 1.0 if item["error_fell"] else 0.0
            parts[f"error_rose_{item['round']}"] = 1.0 if item["error_rose"] else 0.0
    parts["total"] = total.detach()
    return total, {key: float(value) for key, value in parts.items()}
