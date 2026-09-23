"""Pair-grounded latent-memory and target-flow objectives for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryReadOutput:
    """Memory state produced by prefix -> strict Read -> STOP."""

    memory_read: torch.Tensor


@dataclass(frozen=True)
class PairMemoryLossOutput:
    loss: torch.Tensor
    direction_loss: torch.Tensor
    magnitude_loss: torch.Tensor
    regression_loss: torch.Tensor
    noop_loss: torch.Tensor
    cosine: torch.Tensor
    relative_error: torch.Tensor
    student_rms: torch.Tensor
    target_rms: torch.Tensor


def tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def prepare_flow_training_state(
    clean_latent: torch.Tensor,
    timestep: torch.Tensor | float,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply BAGEL's native rectified-flow convention.

    BAGEL integrates from noise at ``t=1`` to data at ``t=0``.  The training
    path is therefore ``x_t=(1-t)x_1+t*epsilon`` with velocity
    ``epsilon-x_1``.
    """

    if clean_latent.shape != noise.shape:
        raise ValueError("clean_latent and noise must have identical shapes")
    t = torch.as_tensor(
        timestep, device=clean_latent.device, dtype=clean_latent.dtype
    )
    if t.numel() != 1:
        raise ValueError("pair-grounded training requires one shared timestep")
    if not 0.0 <= float(t.detach().float()) <= 1.0:
        raise ValueError("timestep must lie in [0, 1]")
    x_t = (1.0 - t) * clean_latent + t * noise
    return x_t, noise - clean_latent


def sample_weighted_timestep(
    generator: torch.Generator,
    *,
    bucket_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Sample early/mid/late denoising time with a 50/30/20 prior.

    Early denoising corresponds to noisy ``t in [2/3, 1]`` in BAGEL's
    reverse integration convention.
    """

    weights = torch.tensor(bucket_weights, dtype=torch.float64)
    if tuple(weights.shape) != (3,) or bool((weights < 0).any()) or not float(
        weights.sum()
    ) > 0.0:
        raise ValueError("bucket_weights must be three non-negative values")
    bucket = int(torch.multinomial(weights / weights.sum(), 1, generator=generator))
    unit = float(torch.rand((), generator=generator))
    ranges = ((2.0 / 3.0, 1.0), (1.0 / 3.0, 2.0 / 3.0), (0.0, 1.0 / 3.0))
    lower, upper = ranges[bucket]
    return lower + unit * (upper - lower)


def pair_memory_loss(
    *,
    student_memory: torch.Tensor,
    source_reference: torch.Tensor,
    target_reference: torch.Tensor,
    is_noop: bool,
    lambda_mem_dir: float = 1.0,
    lambda_mem_mag: float = 0.1,
    lambda_mem_reg: float = 0.25,
    lambda_noop_mem: float = 1.0,
    normalization_floor: float = 1.0e-4,
) -> PairMemoryLossOutput:
    """Match instruction-predicted memory delta to the visual pair delta."""

    if not (
        student_memory.shape == source_reference.shape == target_reference.shape
    ):
        raise ValueError("student/source/target memory shapes must match")
    if student_memory.ndim != 2:
        raise ValueError("memory tensors must have shape [K, hidden_size]")
    if float(normalization_floor) <= 0.0:
        raise ValueError("normalization_floor must be positive")

    source = source_reference.detach().float()
    target_delta = (target_reference.detach().float() - source).detach()
    student_delta = student_memory.float() - source
    target_rms = tensor_rms(target_delta)
    student_rms = tensor_rms(student_delta)
    zero = student_delta.sum() * 0.0

    if is_noop:
        direction_loss = zero
        magnitude_loss = zero
        regression_loss = zero
        noop_loss = student_rms.square()
        loss = float(lambda_noop_mem) * noop_loss
    else:
        direction_loss = (
            1.0
            - F.cosine_similarity(
                student_delta,
                target_delta,
                dim=-1,
                eps=1.0e-8,
            )
        ).mean()
        magnitude_loss = F.smooth_l1_loss(student_rms, target_rms)
        sigma = target_rms.clamp_min(float(normalization_floor))
        regression_loss = F.smooth_l1_loss(
            student_delta / sigma,
            target_delta / sigma,
        )
        noop_loss = zero
        loss = (
            float(lambda_mem_dir) * direction_loss
            + float(lambda_mem_mag) * magnitude_loss
            + float(lambda_mem_reg) * regression_loss
        )

    flat_student = student_delta.reshape(1, -1)
    flat_target = target_delta.reshape(1, -1)
    cosine = F.cosine_similarity(
        flat_student, flat_target, dim=1, eps=1.0e-8
    ).mean()
    relative_error = tensor_rms(student_delta - target_delta) / target_rms.clamp_min(
        float(normalization_floor)
    )
    return PairMemoryLossOutput(
        loss=loss,
        direction_loss=direction_loss,
        magnitude_loss=magnitude_loss,
        regression_loss=regression_loss,
        noop_loss=noop_loss,
        cosine=cosine,
        relative_error=relative_error,
        student_rms=student_rms,
        target_rms=target_rms,
    )


__all__ = [
    "MemoryReadOutput",
    "PairMemoryLossOutput",
    "pair_memory_loss",
    "prepare_flow_training_state",
    "sample_weighted_timestep",
    "tensor_rms",
]
