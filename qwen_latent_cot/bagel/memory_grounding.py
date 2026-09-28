"""Memory dependency objectives. Velocities use [batch, tokens, channels]."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryDependencyLossOutput:
    loss: torch.Tensor
    teacher_loss: torch.Tensor
    shuffle_margin_loss: torch.Tensor
    zero_margin_loss: torch.Tensor
    direction_loss: torch.Tensor
    error_correct: torch.Tensor
    error_shuffled: torch.Tensor
    error_zero: torch.Tensor
    dependency_gap_shuffle: torch.Tensor
    dependency_gap_zero: torch.Tensor
    direction_cosine: torch.Tensor


def memory_dependency_loss(*, correct_velocity, shuffled_velocity, zero_velocity,
                           teacher_velocity, lambda_teacher=1.0, lambda_dep_shuffle=0.5,
                           lambda_dep_zero=0.25, lambda_direction=0.1, margin=0.02,
                           normalization_floor=1e-6):
    values = (correct_velocity, shuffled_velocity, zero_velocity, teacher_velocity)
    if correct_velocity.ndim < 2 or any(v.shape != values[0].shape for v in values):
        raise ValueError("matching batched velocity shapes are required")
    if min(lambda_teacher, lambda_dep_shuffle, lambda_dep_zero, lambda_direction, margin) < 0:
        raise ValueError("loss coefficients and margin must be nonnegative")
    if normalization_floor <= 0:
        raise ValueError("normalization_floor must be positive")
    correct, shuffled, zero, teacher = [v.float().reshape(v.shape[0], -1) for v in values]
    teacher = teacher.detach()
    scale = teacher.square().mean(-1).clamp_min(normalization_floor)
    errors = [(v - teacher).square().mean(-1) / scale for v in (correct, shuffled, zero)]
    ec, es, ez = errors
    shuffle_loss = F.relu(margin + ec - es).mean()
    zero_loss = F.relu(margin + ec - ez).mean()
    desired = (teacher - zero.detach()).detach()
    effect = correct - zero
    cosine = F.cosine_similarity(effect, desired, dim=-1, eps=1e-8)
    active = desired.square().mean(-1) > normalization_floor
    direction = torch.where(active, 1 - cosine, cosine * 0).mean()
    teacher_loss = ec.mean()
    loss = (lambda_teacher * teacher_loss + lambda_dep_shuffle * shuffle_loss
            + lambda_dep_zero * zero_loss + lambda_direction * direction)
    return MemoryDependencyLossOutput(loss, teacher_loss, shuffle_loss, zero_loss, direction,
        ec.detach().mean(), es.detach().mean(), ez.detach().mean(),
        (es - ec).detach().mean(), (ez - ec).detach().mean(), cosine.detach().mean())


def shuffle_across_batch(memory, *, generator=None):
    """Whole-sample cyclic derangement, never slots or token rows.

    Input [B,K,D]. Randomizes sample order before cycling (B=2 is a swap).
    Returns the new tensor and the donor index for reproducibility.
    """
    if memory.ndim != 3 or memory.shape[0] < 2:
        raise ValueError("shuffle requires [B,K,D] with at least two distinct prompts")
    order = torch.randperm(memory.shape[0], generator=generator, device="cpu")
    donors = torch.empty_like(order)
    donors[order] = order.roll(1)
    return memory.index_select(0, donors.to(memory.device)), donors


def write_prompt_mask_probability(step, *, start=0.5, end=0.1, decay_steps=3000):
    if not (0 <= end <= start <= 1) or decay_steps < 1 or step < 0:
        raise ValueError("invalid Write-mask curriculum")
    return start + (end - start) * min(step / decay_steps, 1.0)


@contextmanager
def adapters_off(model):
    """Native teacher/cache creation must not inherit a caller's adapter mode."""
    modules = [m for m in model.modules() if callable(getattr(m, "set_loop_mode", None))]
    states = [m.loop_mode for m in modules]
    try:
        for m in modules:
            m.set_loop_mode("off")
        yield
    finally:
        for m, mode in zip(modules, states):
            m.set_loop_mode(mode)


def causal_reward_advantages(correct, shuffled, quality_correct, quality_native,
                              *, lambda_memory=1.0, lambda_quality=1.0,
                              quality_tolerance=0.0):
    """Within-prompt group advantages: semantic + causal gain - quality penalty."""
    from .flow_grpo import paired_group_advantages, quality_constrained_advantages
    if min(lambda_memory, lambda_quality, quality_tolerance) < 0:
        raise ValueError("reward coefficients must be nonnegative")
    terms = quality_constrained_advantages(correct, shuffled, quality_correct, quality_native,
        quality_tolerance=quality_tolerance, quality_penalty_weight=lambda_quality)
    objective = correct.float() + lambda_memory * terms.semantic_delta - terms.quality_penalty
    advantage = paired_group_advantages(objective, torch.zeros_like(objective))
    return objective.detach(), advantage.detach(), terms
