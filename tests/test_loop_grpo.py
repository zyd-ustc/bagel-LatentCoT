from __future__ import annotations

import torch
from torch import nn

from qwen_latent_cot.bagel.flow_grpo import sde_step_with_logprob
from qwen_latent_cot.bagel.loop_grpo import (
    clone_loop_adapter_state,
    replay_group,
    replay_transition,
    temporary_loop_adapter_state,
)


def test_truncated_final_geneval_row_is_ignored(tmp_path):
    from scripts.train.bagel_loop_grpo_train import _load_prompts

    path = tmp_path / "metadata.jsonl"
    path.write_text('{"prompt":"valid", "tag":"counting"}\n{"prompt":', encoding="utf-8")
    rows = _load_prompts(str(path))
    assert [row["prompt"] for row in rows] == ["valid"]


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.lora_A = nn.Linear(1, 1, bias=False)
        self.block.lora_B = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.block.lora_A.weight.fill_(0.5)
            self.block.lora_B.weight.fill_(0.4)

    def _forward_flow(self, x_t, **_kwargs):
        return self.block.lora_B(self.block.lora_A(x_t))


def _context():
    value = torch.zeros(1, dtype=torch.long)
    return {
        "packed_vae_token_indexes": value,
        "packed_vae_position_ids": value,
        "packed_text_ids": value,
        "packed_text_indexes": value,
        "packed_boundary_token_indexes": value,
        "packed_loop_semantic_token_indexes": value,
        "packed_position_ids": value,
        "packed_indexes": value,
        "packed_seqlens": value,
        "key_values_lens": value,
        "past_key_values": object(),
        "packed_key_value_indexes": value,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
        "cfg_interval": (0.4, 1.0),
        "cfg_text_scale": 4.0,
        "cfg_text_packed_position_ids": value,
        "cfg_text_packed_query_indexes": value,
        "cfg_text_key_values_lens": value,
        "cfg_text_past_key_values": object(),
        "cfg_text_packed_key_value_indexes": value,
        "cfg_img_scale": 1.0,
        "cfg_img_packed_position_ids": value,
        "cfg_img_packed_query_indexes": value,
        "cfg_img_key_values_lens": value,
        "cfg_img_past_key_values": object(),
        "cfg_img_packed_key_value_indexes": value,
        "loop_depth": 2,
        "loop_start_layer": 8,
        "loop_end_layer": 20,
        "loop_timestep_threshold": 0.75,
        "loop_residual_scale": 0.05,
        "loop_state_mode": "boundary",
    }


def _rollout(policy: _TinyPolicy):
    sample = torch.tensor([[0.25], [-0.5]])
    timestep = torch.tensor(0.9)
    next_timestep = torch.tensor(0.8)
    with torch.no_grad():
        velocity = policy._forward_flow(sample)
        generated = sde_step_with_logprob(
            velocity,
            timestep=timestep,
            next_timestep=next_timestep,
            sample=sample,
            sigma_max=0.95,
            noise_level=0.8,
            noise=torch.tensor([[0.3], [-0.7]]),
        )
    return {
        "step_index": 2,
        "sample": sample,
        "next_sample": generated.next_sample,
        "timestep": timestep,
        "next_timestep": next_timestep,
        "old_log_prob": generated.log_prob,
        "sigma_max": torch.tensor(0.95),
        "noise_level": 0.8,
    }


def test_temporary_reference_adapter_restores_policy():
    policy = _TinyPolicy()
    current = clone_loop_adapter_state(policy)
    reference = {name: torch.zeros_like(value) for name, value in current.items()}
    with temporary_loop_adapter_state(policy, reference):
        assert all(torch.count_nonzero(parameter) == 0 for parameter in [
            policy.block.lora_A.weight,
            policy.block.lora_B.weight,
        ])
    restored = clone_loop_adapter_state(policy)
    assert all(torch.equal(current[name], restored[name]) for name in current)


def test_exact_replay_starts_at_ratio_one_and_reaches_adapter_gradient():
    policy = _TinyPolicy()
    reference = clone_loop_adapter_state(policy)
    transition = _rollout(policy)
    result = replay_transition(
        policy,
        transition,
        _context(),
        reference,
        advantage=torch.tensor(1.0),
        clip_range=1e-4,
        kl_beta=0.01,
    )
    assert torch.allclose(result.ratio, torch.ones_like(result.ratio), atol=1e-6)
    assert torch.allclose(result.kl, torch.zeros_like(result.kl), atol=1e-7)
    result.loss.backward()
    assert policy.block.lora_A.weight.grad is not None
    assert policy.block.lora_B.weight.grad is not None
    assert policy.block.lora_A.weight.grad.abs().sum() > 0
    assert policy.block.lora_B.weight.grad.abs().sum() > 0


def test_group_replay_reports_all_transitions():
    policy = _TinyPolicy()
    reference = clone_loop_adapter_state(policy)
    first = _rollout(policy)
    second = dict(first)
    result = replay_group(
        policy,
        trajectories=[(first,), (second,)],
        replay_contexts=[_context(), _context()],
        reference_adapter_state=reference,
        advantages=torch.tensor([1.0, -1.0]),
        clip_range=1e-4,
        kl_beta=0.01,
    )
    assert result.transition_count == 2
    assert torch.allclose(result.ratio_mean, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(result.ratio_max_deviation, torch.tensor(0.0), atol=1e-6)


def test_second_policy_epoch_observes_non_unit_ratio():
    policy = _TinyPolicy()
    reference = clone_loop_adapter_state(policy)
    transition = _rollout(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2, weight_decay=0.0)
    first = replay_transition(
        policy,
        transition,
        _context(),
        reference,
        advantage=torch.tensor(1.0),
        clip_range=1e-3,
        kl_beta=0.0,
    )
    first.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = replay_transition(
        policy,
        transition,
        _context(),
        reference,
        advantage=torch.tensor(1.0),
        clip_range=1e-3,
        kl_beta=0.0,
    )
    assert not torch.allclose(second.ratio, torch.ones_like(second.ratio), atol=1e-7)
