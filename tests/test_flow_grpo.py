from __future__ import annotations

import pytest
import torch

from qwen_latent_cot.bagel.flow_grpo import (
    DEFAULT_LOOP_RESIDUAL_SCALE,
    DEFAULT_LOOP_TIMESTEP_THRESHOLD,
    clipped_grpo_loss,
    gaussian_mean_kl,
    paired_group_advantages,
    quality_constrained_advantages,
    sde_step_with_logprob,
)


def test_selected_rollout_safety_defaults_are_frozen():
    assert DEFAULT_LOOP_TIMESTEP_THRESHOLD == 0.75
    assert DEFAULT_LOOP_RESIDUAL_SCALE == 0.05


def test_zero_noise_sde_is_exact_euler_step():
    sample = torch.tensor([[1.0, 2.0]])
    velocity = torch.tensor([[0.5, -1.0]], requires_grad=True)
    transition = sde_step_with_logprob(
        velocity,
        timestep=0.8,
        next_timestep=0.6,
        sample=sample,
        noise_level=0.0,
    )
    assert torch.allclose(
        transition.next_sample,
        sample + velocity.detach() * (0.6 - 0.8),
    )
    assert float(transition.log_prob.detach()) == 0.0


def test_sde_replay_logprob_has_velocity_gradient():
    velocity = torch.randn(4, 3, requires_grad=True)
    sample = torch.randn(4, 3)
    rollout = sde_step_with_logprob(
        velocity.detach(),
        timestep=0.8,
        next_timestep=0.7,
        sample=sample,
        noise_level=0.8,
    )
    replay = sde_step_with_logprob(
        velocity,
        timestep=0.8,
        next_timestep=0.7,
        sample=sample,
        next_sample=rollout.next_sample,
        noise_level=0.8,
    )
    (-replay.log_prob).backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()


def test_sde_explicit_noise_is_shareable_and_shape_checked():
    velocity = torch.randn(4, 3)
    sample = torch.randn(4, 3)
    noise = torch.randn_like(sample)
    left = sde_step_with_logprob(
        velocity,
        timestep=0.8,
        next_timestep=0.7,
        sample=sample,
        noise_level=0.8,
        noise=noise,
    )
    right = sde_step_with_logprob(
        velocity,
        timestep=0.8,
        next_timestep=0.7,
        sample=sample,
        noise_level=0.8,
        noise=noise,
    )
    assert torch.equal(left.next_sample, right.next_sample)
    with pytest.raises(ValueError, match="noise shape"):
        sde_step_with_logprob(
            velocity,
            timestep=0.8,
            next_timestep=0.7,
            sample=sample,
            noise_level=0.8,
            noise=torch.randn(2),
        )


def test_paired_advantage_uses_loop_minus_base():
    advantages = paired_group_advantages(
        torch.tensor([3.0, 2.0, 1.0]),
        torch.tensor([1.0, 1.5, 2.0]),
    )
    assert abs(float(advantages.mean())) < 1e-5
    assert advantages[0] > advantages[1] > advantages[2]


def test_grpo_ratio_one_before_update():
    old = torch.tensor([0.2, -0.3])
    advantage = torch.tensor([1.0, -1.0])
    loss = clipped_grpo_loss(old, old, advantage, clip_range=1e-5)
    assert float(loss) == pytest.approx(0.0)


def test_quality_constrained_advantage_penalizes_only_quality_regression():
    result = quality_constrained_advantages(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([2.0, 1.8, 2.2]),
        torch.tensor([2.0, 2.0, 2.0]),
        quality_tolerance=0.05,
        quality_penalty_weight=2.0,
    )
    assert torch.allclose(result.quality_penalty, torch.tensor([0.0, 0.3, 0.0]))
    assert result.advantage[1] < result.advantage[0]
    assert abs(float(result.advantage.mean())) < 1e-5


def test_gaussian_mean_kl_is_zero_for_reference_and_positive_after_shift():
    policy = torch.zeros(2, 3)
    reference = torch.zeros(2, 3)
    assert float(gaussian_mean_kl(policy, reference, 0.5)) == 0.0
    assert float(gaussian_mean_kl(policy + 1.0, reference, 0.5)) == pytest.approx(2.0)
