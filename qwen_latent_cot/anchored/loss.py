"""Within-step losses. The anchor term is a small prior, not the objective.

``delta`` is the residual actually added at the body entry, ``Bound(gamma * A_phi)``.
``velocity`` is the sampler velocity, the last round. The flow term weights each
image token by the detached native error on that token, then clamps the weight.
``round_loss_weights`` replaces that term with one weight per correction.
``None`` keeps it on the final velocity. Weights may differ, and a zero weight
stays out of the graph. The anchor prior stays on the final velocity and pulls
harder where the native error is small. ``lambda_anchor`` still scales it.
The entry term stays on the last written residual.

``lambda_improve`` scales a one-sided margin on the same final velocity. It is
0 unless a caller sets it, so the focal term stays the whole flow objective.
The margin compares summed squares, not the mean over channels. It does not
show that the correction improves a generated image.
"""
from __future__ import annotations

import torch

from .diagnostics import round_velocity_report


def native_error_weights(
    native: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token flow weight and anchor protection from the detached native error.

    The flow weight is the token error divided by its mean, clamped to
    ``[0.2, 3]``. The protection weight is larger where that flow weight is
    smaller, and is clamped to ``[0.3, 2]``. Neither weight receives gradient.
    """
    if native.shape != target.shape or native.ndim != 2:
        raise ValueError("native velocity and target must be [tokens, channels]")
    with torch.no_grad():
        token_err = (native.float() - target.float()).pow(2).mean(dim=-1, keepdim=True)
        focal = (token_err / (token_err.mean() + eps)).clamp(min=0.2, max=3.0)
        protect = (1.0 / (focal + 0.5)).clamp(min=0.3, max=2.0)
    return focal, protect


def focal_flow_loss(
    velocity: torch.Tensor,
    target: torch.Tensor,
    native: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Token-weighted flow loss. Gradient reaches ``velocity`` only."""
    if velocity.shape != target.shape:
        raise ValueError("velocity and target shapes differ")
    focal, _protect = native_error_weights(native, target, eps)
    return (focal * (velocity.float() - target.float()).pow(2)).mean()


def improvement_margin_loss(
    velocity: torch.Tensor,
    target: torch.Tensor,
    native: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Mean over tokens of ``relu(||v_corr - target||_2^2 - ||v_native - target||_2^2 + m)``.

    The squared norm sums over channels. The native term is detached. A token
    whose corrected error is lower than the native error by at least ``margin``
    contributes nothing. ``margin`` is an absolute gap on that sum, not a
    tuned image metric.
    """
    if velocity.shape != target.shape or velocity.shape != native.shape or velocity.ndim != 2:
        raise ValueError("improvement margin expects [tokens, channels] velocities")
    if velocity.shape[0] == 0:
        raise ValueError("improvement margin needs at least one token")
    gap = float(margin)
    if gap < 0.0 or gap != gap:
        raise ValueError("improvement margin must be non-negative")
    corrected = (velocity.float() - target.float()).pow(2).sum(dim=-1)
    with torch.no_grad():
        native_error = (native.float() - target.float()).pow(2).sum(dim=-1)
    return (corrected - native_error + gap).clamp(min=0).mean()


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
    lambda_improve: float = 0.0,
    improvement_margin: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    focal, protect = native_error_weights(native_velocity, target, eps)
    if round_loss_weights is None:
        final = focal_flow_loss(velocity, target, native_velocity, eps)
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
            if weight == 0.0:
                continue
            term = (focal * (pred.float() - target.float()).pow(2)).mean()
            weighted = term if weight == 1.0 else weight * term
            final = weighted if final is None else final + weighted
        if final is None:
            final = target.float().new_zeros(())
    parts = {"final": final.detach()}
    total = final
    native = native_velocity.detach()
    anchor_term = (protect * (velocity.float() - native.float()).pow(2)).mean()
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
    improve_scale = float(lambda_improve)
    if improve_scale < 0.0:
        raise ValueError("lambda_improve must be non-negative")
    if improve_scale != 0.0:
        improve = improvement_margin_loss(velocity, target, native_velocity, improvement_margin)
        parts["improve"] = improve.detach()
        total = total + improve_scale * improve
    else:
        with torch.no_grad():
            improve = improvement_margin_loss(velocity, target, native_velocity, improvement_margin)
        parts["improve"] = improve.detach()
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
