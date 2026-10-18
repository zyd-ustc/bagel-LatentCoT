"""Within-step losses. The anchor term is a small prior, not the objective.

``delta`` is the residual actually added at the body entry, ``gamma * Bound(A_phi)``.
"""
from __future__ import annotations

import torch


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
) -> tuple[torch.Tensor, dict]:
    if velocity.shape != target.shape:
        raise ValueError("velocity and target shapes differ")
    final = (velocity.float() - target.float()).pow(2).mean()
    parts = {"final": final.detach()}
    total = final
    native = native_velocity.detach()
    anchor_term = (velocity.float() - native.float()).pow(2).mean() / (native.float().pow(2).mean() + eps)
    parts["anchor"] = anchor_term.detach()
    total = total + lambda_anchor * anchor_term
    if delta is None or anchor is None or vae_indexes is None:
        entry = velocity.new_zeros(())
    else:
        raw = delta[vae_indexes].float()
        base = anchor[vae_indexes].float().detach()
        entry = (raw.pow(2).sum(-1) / (base.pow(2).sum(-1) + eps)).mean()
    parts["entry"] = entry.detach()
    total = total + lambda_entry * entry
    parts["total"] = total.detach()
    return total, {key: float(value) for key, value in parts.items()}
