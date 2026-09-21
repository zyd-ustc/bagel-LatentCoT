"""Selected-state replay and losses for Phase-1 delta-velocity distillation.

Rollout is intentionally separate and runs under ``torch.no_grad``.  This
module rebuilds BAGEL's native velocity call at a saved ``(x_t, t, m_in)``;
only the Student K>0 replay should be executed with autograd enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VelocityReplayResult:
    velocity: torch.Tensor
    memory: Optional[torch.Tensor]
    memory_text: Optional[torch.Tensor]
    memory_img: Optional[torch.Tensor]


@dataclass(frozen=True)
class DeltaVelocityLossResult:
    loss: torch.Tensor
    delta_v_loss: torch.Tensor
    direction_loss: torch.Tensor
    overshoot_loss: torch.Tensor
    noop_loss: torch.Tensor
    cosine: torch.Tensor
    relative_error: torch.Tensor
    student_rms: torch.Tensor
    teacher_rms: torch.Tensor
    direction_active: bool


def _sample_without_replacement(
    values: Sequence[int], count: int, generator: torch.Generator
) -> list[int]:
    if count <= 0 or not values:
        return []
    order = torch.randperm(len(values), generator=generator).tolist()
    return [int(values[index]) for index in order[: min(count, len(values))]]


def sample_replay_step_indices(
    num_steps: int,
    count: int,
    *,
    generator: torch.Generator,
    early_fraction: float = 0.4,
    mid_fraction: float = 0.75,
    bucket_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> Tuple[int, ...]:
    """Sample unique actual step indexes with the Phase-1 60/30/10 prior."""

    total = int(num_steps)
    wanted = int(count)
    if total < 1:
        raise ValueError("num_steps must be positive")
    if not 1 <= wanted <= total:
        raise ValueError("count must be in [1, num_steps]")
    if not 0.0 < float(early_fraction) < float(mid_fraction) < 1.0:
        raise ValueError("bucket fractions must satisfy 0 < early < mid < 1")
    weights = torch.tensor(bucket_weights, dtype=torch.float64)
    if tuple(weights.shape) != (3,) or bool((weights < 0).any()) or not float(
        weights.sum()
    ) > 0.0:
        raise ValueError("bucket_weights must contain three non-negative values")

    early_end = max(1, min(total - 2, int(round(total * early_fraction))))
    mid_end = max(early_end + 1, min(total - 1, int(round(total * mid_fraction))))
    buckets = (
        tuple(range(0, early_end)),
        tuple(range(early_end, mid_end)),
        tuple(range(mid_end, total)),
    )
    selected: list[int] = []
    available = [list(bucket) for bucket in buckets]
    probabilities = weights.clone()
    for _ in range(wanted):
        for index, bucket in enumerate(available):
            if not bucket:
                probabilities[index] = 0.0
        bucket_index = int(
            torch.multinomial(
                probabilities / probabilities.sum(),
                num_samples=1,
                generator=generator,
            )[0]
        )
        choice = _sample_without_replacement(
            available[bucket_index], 1, generator
        )[0]
        available[bucket_index].remove(choice)
        selected.append(choice)
    return tuple(sorted(selected))


def _per_sample_loop_memory(
    model: nn.Module, condition: Any, x_t: torch.Tensor
) -> torch.Tensor:
    indexes = condition.flow_input["packed_loop_token_indexes"]
    sequence_lengths = condition.flow_input["packed_vae_seqlens"]
    samples = int(sequence_lengths.numel())
    total_slots = int(indexes.numel())
    if samples < 1 or total_slots < 1 or total_slots % samples:
        raise ValueError("loop-token indexes must tile equally across samples")
    slots = total_slots // samples
    loop_memory = getattr(model, "loop_memory", None)
    if loop_memory is None or slots > int(loop_memory.shape[0]):
        raise ValueError("student K exceeds the initialized frozen loop memory")
    return loop_memory[:slots].to(device=x_t.device).repeat(samples, 1)


def replay_velocity(
    model: nn.Module,
    inferencer: Any,
    condition: Any,
    state: Mapping[str, Any],
    *,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "text_channel",
    loop_depth: int = 2,
    loop_recycle_mode: str = "same_depth",
    memory_loop_start: int = 12,
    memory_loop_end: int = 20,
    round0_memory_write_enabled: bool = False,
    loop_uncond_memory: str = "m0",
) -> VelocityReplayResult:
    """Replay Base/Teacher K=0 or Student K>0 on one exact rollout state."""

    sample = state["sample"]
    kwargs = inferencer.build_image_velocity_kwargs(
        x_t=sample,
        timestep=state["timestep"],
        condition=condition,
        cfg_text_scale=float(cfg_text_scale),
        cfg_img_scale=float(cfg_img_scale),
        cfg_interval=tuple(cfg_interval),
        cfg_renorm_min=float(cfg_renorm_min),
        cfg_renorm_type=str(cfg_renorm_type),
    )
    loop_indexes = condition.flow_input["packed_loop_token_indexes"]
    if int(loop_indexes.numel()) == 0:
        velocity = model._forward_flow(**kwargs)
        return VelocityReplayResult(velocity, None, None, None)

    if int(loop_depth) != 2:
        raise ValueError("Phase 1.1 replay requires loop_depth=2")
    if str(loop_recycle_mode) != "same_depth":
        raise ValueError("Phase 1.1 replay requires same_depth recycling")
    if str(loop_uncond_memory) not in ("m0", "zero"):
        raise ValueError("loop_uncond_memory must be 'm0' or 'zero'")

    for key in (
        "within_step_loop_start",
        "within_step_loop_end",
        "within_step_loop_repeat",
        "within_step_loop_damping",
    ):
        kwargs.pop(key, None)
    m0 = _per_sample_loop_memory(model, condition, sample)
    unconditional = torch.zeros_like(m0) if loop_uncond_memory == "zero" else m0
    m_in = state.get("m_in")
    m_in_text = state.get("m_in_text")
    m_in_img = state.get("m_in_img")
    result = model._forward_flow_loop(
        **kwargs,
        packed_loop_token_indexes=loop_indexes,
        loop_memory=m0,
        loop_memory_text=unconditional,
        loop_memory_img=unconditional,
        recycle_mode="same_depth",
        memory_loop_repeat=2,
        memory_loop_start=int(memory_loop_start),
        memory_loop_end=int(memory_loop_end),
        memory_body_in=None if m_in is None else m_in.detach(),
        memory_body_in_text=None if m_in_text is None else m_in_text.detach(),
        memory_body_in_img=None if m_in_img is None else m_in_img.detach(),
        embed_memory=m0,
        round0_memory_write_enabled=bool(round0_memory_write_enabled),
        collect_round_diagnostics=False,
    )
    velocity, memory, memory_text, memory_img = result[:4]
    return VelocityReplayResult(velocity, memory, memory_text, memory_img)


def delta_velocity_distillation_loss(
    *,
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    base_velocity: torch.Tensor,
    is_noop: bool,
    normalization_floor: float = 1.0e-4,
    direction_active_threshold: float = 1.0e-3,
    lambda_delta_v: float = 1.0,
    lambda_dir: float = 0.1,
    lambda_over: float = 0.05,
    lambda_noop: float = 1.0,
    overshoot_gamma: float = 1.5,
) -> DeltaVelocityLossResult:
    """Compute the exact Phase-1.1 correction objective and diagnostics."""

    if not (
        student_velocity.shape == teacher_velocity.shape == base_velocity.shape
    ):
        raise ValueError("base, teacher, and student velocity shapes must match")
    if float(normalization_floor) <= 0.0:
        raise ValueError("normalization_floor must be positive")
    base = base_velocity.detach()
    teacher_delta = (teacher_velocity.detach() - base).float()
    student_delta = (student_velocity - base).float()
    teacher_rms = teacher_delta.square().mean().sqrt()
    student_rms = student_delta.square().mean().sqrt()
    denominator = teacher_delta.square().mean().clamp_min(
        float(normalization_floor)
    ).sqrt()

    zero = student_delta.sum() * 0.0
    delta_loss = F.smooth_l1_loss(
        student_delta / denominator,
        teacher_delta / denominator,
    )
    active = bool(teacher_rms.detach() > float(direction_active_threshold))
    if active:
        direction_loss = 1.0 - F.cosine_similarity(
            student_delta.reshape(1, -1),
            teacher_delta.reshape(1, -1),
            dim=1,
            eps=1.0e-8,
        ).mean()
    else:
        direction_loss = zero
    overshoot_loss = F.relu(
        student_rms - float(overshoot_gamma) * teacher_rms
    ).square()
    noop_loss = student_delta.square().mean() if bool(is_noop) else zero

    # No-op records are a separate restraint objective; fitting tiny teacher
    # numerical differences would defeat their purpose.
    if bool(is_noop):
        loss = float(lambda_noop) * noop_loss
    else:
        loss = (
            float(lambda_delta_v) * delta_loss
            + float(lambda_dir) * direction_loss
            + float(lambda_over) * overshoot_loss
        )

    cosine = F.cosine_similarity(
        student_delta.detach().reshape(1, -1),
        teacher_delta.reshape(1, -1),
        dim=1,
        eps=1.0e-8,
    ).mean()
    relative_error = (
        (student_delta.detach() - teacher_delta).norm()
        / teacher_delta.norm().clamp_min(1.0e-8)
    )
    return DeltaVelocityLossResult(
        loss=loss,
        delta_v_loss=delta_loss.detach(),
        direction_loss=direction_loss.detach(),
        overshoot_loss=overshoot_loss.detach(),
        noop_loss=noop_loss.detach(),
        cosine=cosine.detach(),
        relative_error=relative_error.detach(),
        student_rms=student_rms.detach(),
        teacher_rms=teacher_rms.detach(),
        direction_active=active,
    )


__all__ = [
    "DeltaVelocityLossResult",
    "VelocityReplayResult",
    "delta_velocity_distillation_loss",
    "replay_velocity",
    "sample_replay_step_indices",
]
