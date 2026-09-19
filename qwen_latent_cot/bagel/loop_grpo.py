"""Differentiable replay of BAGEL loop-policy rollout transitions.

Rollout runs under ``no_grad`` and stores only selected stochastic transitions.
This module replays those exact states through the current and frozen-reference
loop adapters, so GRPO never needs to keep a 29-step BAGEL graph in memory.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch import nn

from .flow_grpo import clipped_grpo_loss, gaussian_mean_kl, sde_step_with_logprob


@dataclass(frozen=True)
class ReplayResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    ratio: torch.Tensor
    new_log_prob: torch.Tensor
    old_log_prob: torch.Tensor


@dataclass(frozen=True)
class ReplayGroupResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max_deviation: torch.Tensor
    transition_count: int


def _loop_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    if not parameters:
        raise RuntimeError("model contains no loop LoRA parameters")
    return parameters


def clone_loop_adapter_state(
    model: nn.Module,
    *,
    device: torch.device | str | None = None,
) -> Dict[str, torch.Tensor]:
    """Clone the loop adapter only; this is the frozen GRPO reference policy."""

    destination = None if device is None else torch.device(device)
    return {
        name: parameter.detach()
        .clone()
        .to(destination if destination is not None else parameter.device)
        for name, parameter in _loop_parameters(model).items()
    }


def _copy_loop_adapter_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    parameters = _loop_parameters(model)
    if set(parameters) != set(state):
        raise RuntimeError(
            "loop adapter state keys differ from model: "
            f"missing={sorted(set(parameters) - set(state))[:8]}, "
            f"unexpected={sorted(set(state) - set(parameters))[:8]}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"loop adapter shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@contextmanager
def temporary_loop_adapter_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
):
    """Temporarily swap only LoRA tensors without duplicating the BAGEL model."""

    current = clone_loop_adapter_state(model)
    _copy_loop_adapter_state(model, state)
    try:
        yield
    finally:
        _copy_loop_adapter_state(model, current)


def _flow_kwargs(
    replay_context: Mapping[str, Any],
    timestep: torch.Tensor,
    transition: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg_low, cfg_high = replay_context["cfg_interval"]
    scalar_t = float(timestep.detach().float().reshape(-1)[0])
    cfg_active = float(cfg_low) < scalar_t <= float(cfg_high)

    keys = (
        "packed_vae_token_indexes",
        "packed_vae_position_ids",
        "packed_text_ids",
        "packed_text_indexes",
        "packed_boundary_token_indexes",
        "packed_position_ids",
        "packed_indexes",
        "packed_seqlens",
        "key_values_lens",
        "past_key_values",
        "packed_key_value_indexes",
        "cfg_renorm_min",
        "cfg_renorm_type",
        "cfg_text_packed_position_ids",
        "cfg_text_packed_query_indexes",
        "cfg_text_key_values_lens",
        "cfg_text_past_key_values",
        "cfg_text_packed_key_value_indexes",
        "cfg_img_packed_position_ids",
        "cfg_img_packed_query_indexes",
        "cfg_img_key_values_lens",
        "cfg_img_past_key_values",
        "cfg_img_packed_key_value_indexes",
    )
    kwargs = {key: replay_context[key] for key in keys if key in replay_context}
    kwargs.update(
        timestep=timestep,
        cfg_text_scale=(float(replay_context["cfg_text_scale"]) if cfg_active else 1.0),
        cfg_img_scale=(float(replay_context["cfg_img_scale"]) if cfg_active else 1.0),
        cfg_type=replay_context.get("cfg_type", "parallel"),
    )
    loop_idx = replay_context.get("packed_loop_token_indexes")
    if loop_idx is not None and int(getattr(loop_idx, "numel", lambda: 0)()) > 0:
        kwargs.update(
            packed_loop_token_indexes=loop_idx,
            loop_memory=replay_context.get("embed_memory"),
            embed_memory=replay_context.get("embed_memory"),
            recycle_mode=str(replay_context.get("recycle_mode", "same_depth")),
            memory_loop_repeat=int(replay_context.get("memory_loop_repeat", 2)),
            memory_loop_start=replay_context.get("memory_loop_start"),
            memory_loop_end=replay_context.get("memory_loop_end"),
            round0_gen_reads_memory=bool(
                replay_context.get("round0_gen_reads_memory", False)
            ),
        )
        if transition is not None:
            kwargs.update(
                memory_body_in=transition.get("m_in"),
                memory_body_in_text=transition.get("m_in_text"),
                memory_body_in_img=transition.get("m_in_img"),
                loop_memory_text=transition.get("m_in_text"),
                loop_memory_img=transition.get("m_in_img"),
            )
    return kwargs


def _policy_velocity(model: nn.Module, sample: torch.Tensor, kwargs: Dict[str, Any]):
    loop_idx = kwargs.get("packed_loop_token_indexes")
    if loop_idx is not None and int(getattr(loop_idx, "numel", lambda: 0)()) > 0:
        result = model._forward_flow_loop(x_t=sample, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return model._forward_flow(x_t=sample, **kwargs)


def replay_transition(
    model: nn.Module,
    transition: Mapping[str, Any],
    replay_context: Mapping[str, Any],
    reference_adapter_state: Mapping[str, torch.Tensor],
    advantage: torch.Tensor | float,
    *,
    clip_range: float,
    kl_beta: float,
) -> ReplayResult:
    """Replay one sampled transition through current and reference policies."""

    sample = transition["sample"]
    next_sample = transition["next_sample"]
    scalar_t = torch.as_tensor(
        transition["timestep"], device=sample.device, dtype=torch.float32
    ).reshape(())
    timestep = scalar_t.expand(int(sample.shape[0]))
    kwargs = _flow_kwargs(replay_context, timestep, transition)

    # Reference first, then restore the current adapter before constructing the
    # autograd graph. This avoids parameter-version changes after policy forward.
    with temporary_loop_adapter_state(model, reference_adapter_state):
        with torch.no_grad():
            reference_velocity = _policy_velocity(model, sample, kwargs)

    with torch.no_grad():
        reference_transition = sde_step_with_logprob(
            reference_velocity,
            timestep=scalar_t,
            next_timestep=transition["next_timestep"],
            sample=sample,
            next_sample=next_sample,
            sigma_max=transition["sigma_max"],
            noise_level=float(transition["noise_level"]),
        )

    return _replay_transition_with_reference(
        model,
        transition,
        replay_context,
        reference_transition,
        advantage,
        clip_range=clip_range,
        kl_beta=kl_beta,
    )


def _replay_transition_with_reference(
    model: nn.Module,
    transition: Mapping[str, Any],
    replay_context: Mapping[str, Any],
    reference_transition: Any,
    advantage: torch.Tensor | float,
    *,
    clip_range: float,
    kl_beta: float,
) -> ReplayResult:
    """Build the current-policy graph after the reference adapter is restored."""

    sample = transition["sample"]
    scalar_t = torch.as_tensor(
        transition["timestep"], device=sample.device, dtype=torch.float32
    ).reshape(())
    timestep = scalar_t.expand(int(sample.shape[0]))
    kwargs = _flow_kwargs(replay_context, timestep, transition)
    policy_velocity = _policy_velocity(model, sample, kwargs)
    policy_transition = sde_step_with_logprob(
        policy_velocity,
        timestep=scalar_t,
        next_timestep=transition["next_timestep"],
        sample=sample,
        next_sample=transition["next_sample"],
        sigma_max=transition["sigma_max"],
        noise_level=float(transition["noise_level"]),
    )

    old_log_prob = torch.as_tensor(
        transition["old_log_prob"],
        device=policy_transition.log_prob.device,
        dtype=policy_transition.log_prob.dtype,
    )
    advantage_tensor = torch.as_tensor(
        advantage,
        device=policy_transition.log_prob.device,
        dtype=policy_transition.log_prob.dtype,
    )
    policy_loss = clipped_grpo_loss(
        policy_transition.log_prob,
        old_log_prob,
        advantage_tensor,
        clip_range=float(clip_range),
    )
    next_t = torch.as_tensor(
        transition["next_timestep"], device=sample.device, dtype=torch.float32
    )
    transition_std = policy_transition.std * torch.sqrt(scalar_t - next_t)
    kl = gaussian_mean_kl(
        policy_transition.mean,
        reference_transition.mean,
        transition_std,
    )
    ratio = torch.exp(policy_transition.log_prob.detach() - old_log_prob)
    return ReplayResult(
        loss=policy_loss + float(kl_beta) * kl,
        policy_loss=policy_loss,
        kl=kl,
        ratio=ratio,
        new_log_prob=policy_transition.log_prob,
        old_log_prob=old_log_prob,
    )


def replay_group(
    model: nn.Module,
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    replay_contexts: Sequence[Mapping[str, Any]],
    reference_adapter_state: Mapping[str, torch.Tensor],
    advantages: torch.Tensor,
    *,
    clip_range: float,
    kl_beta: float,
) -> ReplayGroupResult:
    """Average GRPO loss over every selected transition in a prompt group."""

    if not (len(trajectories) == len(replay_contexts) == int(advantages.numel())):
        raise ValueError("trajectory, context, and advantage group sizes must match")
    prepared = []
    # Swapping the reference adapter once for the whole group is essential:
    # swapping it between transitions would bump LoRA parameter versions while
    # earlier current-policy graphs are still alive, triggering autograd's
    # in-place modification check at the final backward().
    with temporary_loop_adapter_state(model, reference_adapter_state):
        for trajectory, context, advantage in zip(
            trajectories, replay_contexts, advantages.reshape(-1)
        ):
            if not trajectory:
                raise ValueError(
                    "each rollout must contain at least one SDE transition"
                )
            for transition in trajectory:
                sample = transition["sample"]
                scalar_t = torch.as_tensor(
                    transition["timestep"],
                    device=sample.device,
                    dtype=torch.float32,
                ).reshape(())
                timestep = scalar_t.expand(int(sample.shape[0]))
                kwargs = _flow_kwargs(context, timestep, transition)
                with torch.no_grad():
                    reference_velocity = _policy_velocity(model, sample, kwargs)
                    reference_transition = sde_step_with_logprob(
                        reference_velocity,
                        timestep=scalar_t,
                        next_timestep=transition["next_timestep"],
                        sample=sample,
                        next_sample=transition["next_sample"],
                        sigma_max=transition["sigma_max"],
                        noise_level=float(transition["noise_level"]),
                    )
                prepared.append((transition, context, advantage, reference_transition))

    results = [
        _replay_transition_with_reference(
            model,
            transition,
            context,
            reference_transition,
            advantage,
            clip_range=float(clip_range),
            kl_beta=float(kl_beta),
        )
        for transition, context, advantage, reference_transition in prepared
    ]
    if not results:
        raise ValueError("replay group is empty")
    losses = torch.stack([result.loss for result in results])
    policy_losses = torch.stack([result.policy_loss for result in results])
    kls = torch.stack([result.kl for result in results])
    ratios = torch.stack([result.ratio for result in results])
    return ReplayGroupResult(
        loss=losses.mean(),
        policy_loss=policy_losses.mean(),
        kl=kls.mean(),
        ratio_mean=ratios.mean(),
        ratio_max_deviation=(ratios - 1.0).abs().max(),
        transition_count=len(results),
    )


__all__ = [
    "ReplayGroupResult",
    "ReplayResult",
    "clone_loop_adapter_state",
    "replay_group",
    "replay_transition",
    "temporary_loop_adapter_state",
]
