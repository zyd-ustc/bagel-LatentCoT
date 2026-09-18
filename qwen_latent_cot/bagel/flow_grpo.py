"""Small, model-independent Flow-GRPO math used by BAGEL loop training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# Selected by the fixed 2-prompt x 4-seed BAGEL quality guardrail on
# 2026-09-12. Keep model-level loop defaults neutral; GRPO rollout callers
# explicitly opt into these values.
DEFAULT_LOOP_TIMESTEP_THRESHOLD = 0.75
DEFAULT_LOOP_RESIDUAL_SCALE = 0.05


@dataclass(frozen=True)
class SDETransition:
    sample: torch.Tensor
    next_sample: torch.Tensor
    mean: torch.Tensor
    log_prob: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class RewardAdvantage:
    semantic_delta: torch.Tensor
    quality_delta: torch.Tensor
    quality_penalty: torch.Tensor
    objective: torch.Tensor
    advantage: torch.Tensor


def sde_step_with_logprob(
    velocity: torch.Tensor,
    *,
    timestep: torch.Tensor | float,
    next_timestep: torch.Tensor | float,
    sample: torch.Tensor,
    next_sample: Optional[torch.Tensor] = None,
    sigma_max: torch.Tensor | float = 0.999,
    noise_level: float = 0.8,
    generator: Optional[torch.Generator] = None,
    noise: Optional[torch.Tensor] = None,
) -> SDETransition:
    """Flow-GRPO SDE transition with a replayable Gaussian log probability."""

    velocity_f = velocity.float()
    sample_f = sample.float()
    device = sample.device
    t = torch.as_tensor(timestep, device=device, dtype=torch.float32)
    next_t = torch.as_tensor(next_timestep, device=device, dtype=torch.float32)
    dt = next_t - t
    if bool(torch.any(dt >= 0)):
        raise ValueError("next_timestep must be smaller than timestep")
    if float(noise_level) < 0.0:
        raise ValueError("noise_level must be non-negative")
    if noise is not None and generator is not None:
        raise ValueError("pass either generator or explicit noise, not both")

    if float(noise_level) == 0.0:
        mean = sample_f + velocity_f * dt
        chosen = mean if next_sample is None else next_sample.float()
        if not torch.allclose(chosen, mean, atol=1e-6, rtol=1e-6):
            raise ValueError("noise_level=0 requires the deterministic Euler transition")
        zero_log_prob = velocity_f.sum() * 0.0
        return SDETransition(
            sample=sample,
            next_sample=chosen.to(sample.dtype),
            mean=mean,
            log_prob=zero_log_prob,
            std=torch.zeros_like(t),
        )

    sigma_max_t = torch.as_tensor(sigma_max, device=device, dtype=torch.float32)
    denominator = 1.0 - torch.where(t == 1.0, sigma_max_t, t)
    std = torch.sqrt(t / denominator) * float(noise_level)
    mean = sample_f * (1.0 + std.square() / (2.0 * t) * dt)
    mean = mean + velocity_f * (
        1.0 + std.square() * (1.0 - t) / (2.0 * t)
    ) * dt
    transition_std = std * torch.sqrt(-dt)
    if next_sample is None:
        if noise is None:
            sampled_noise = torch.randn(
                mean.shape,
                generator=generator,
                device=mean.device,
                dtype=mean.dtype,
            )
        else:
            if noise.shape != mean.shape:
                raise ValueError(
                    f"explicit noise shape {tuple(noise.shape)} does not match "
                    f"sample shape {tuple(mean.shape)}"
                )
            sampled_noise = noise.to(device=mean.device, dtype=mean.dtype)
        chosen = mean + transition_std * sampled_noise
    else:
        chosen = next_sample.float()
    log_prob = -(
        (chosen.detach() - mean).square() / (2.0 * transition_std.square())
    ).mean()
    return SDETransition(
        sample=sample,
        next_sample=chosen.to(sample.dtype),
        mean=mean,
        log_prob=log_prob,
        std=std,
    )


def paired_group_advantages(
    loop_rewards: torch.Tensor,
    base_rewards: torch.Tensor,
    *,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Normalize paired loop-minus-base rewards within one prompt group."""

    if loop_rewards.shape != base_rewards.shape or loop_rewards.numel() < 2:
        raise ValueError("paired rewards require matching shapes with at least two samples")
    delta = loop_rewards.float() - base_rewards.float()
    return (delta - delta.mean()) / (delta.std(unbiased=False) + float(epsilon))


def quality_constrained_advantages(
    loop_semantic_rewards: torch.Tensor,
    base_semantic_rewards: torch.Tensor,
    loop_quality_rewards: torch.Tensor,
    base_quality_rewards: torch.Tensor,
    *,
    quality_tolerance: float = 0.0,
    quality_penalty_weight: float = 1.0,
    epsilon: float = 1e-4,
) -> RewardAdvantage:
    """Normalize paired semantic gain with a one-sided quality penalty."""

    tensors = (
        loop_semantic_rewards,
        base_semantic_rewards,
        loop_quality_rewards,
        base_quality_rewards,
    )
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("semantic and quality reward tensors must have matching shapes")
    if tensors[0].numel() < 2:
        raise ValueError("a GRPO prompt group requires at least two samples")
    if float(quality_tolerance) < 0.0 or float(quality_penalty_weight) < 0.0:
        raise ValueError("quality tolerance and penalty weight must be non-negative")

    semantic_delta = loop_semantic_rewards.float() - base_semantic_rewards.float()
    quality_delta = loop_quality_rewards.float() - base_quality_rewards.float()
    quality_penalty = float(quality_penalty_weight) * torch.relu(
        -float(quality_tolerance) - quality_delta
    )
    objective = semantic_delta - quality_penalty
    advantage = (objective - objective.mean()) / (
        objective.std(unbiased=False) + float(epsilon)
    )
    return RewardAdvantage(
        semantic_delta=semantic_delta,
        quality_delta=quality_delta,
        quality_penalty=quality_penalty,
        objective=objective,
        advantage=advantage,
    )


def gaussian_mean_kl(
    policy_mean: torch.Tensor,
    reference_mean: torch.Tensor,
    std: torch.Tensor | float,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """KL between equal-variance diagonal Gaussians, reduced to one scalar."""

    if policy_mean.shape != reference_mean.shape:
        raise ValueError("policy and reference means must have matching shapes")
    std_tensor = torch.as_tensor(
        std, device=policy_mean.device, dtype=torch.float32
    )
    variance = std_tensor.square().clamp_min(float(epsilon))
    return (
        (policy_mean.float() - reference_mean.float()).square()
        / (2.0 * variance)
    ).mean()


def clipped_grpo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_range: float = 1e-5,
) -> torch.Tensor:
    if float(clip_range) <= 0.0:
        raise ValueError("clip_range must be positive")
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(
        ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)
    )
    return torch.maximum(unclipped, clipped).mean()


__all__ = [
    "DEFAULT_LOOP_RESIDUAL_SCALE",
    "DEFAULT_LOOP_TIMESTEP_THRESHOLD",
    "RewardAdvantage",
    "SDETransition",
    "clipped_grpo_loss",
    "gaussian_mean_kl",
    "paired_group_advantages",
    "quality_constrained_advantages",
    "sde_step_with_logprob",
]
