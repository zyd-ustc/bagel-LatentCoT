from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_latent_cot.bagel.flow_grpo import sde_step_with_logprob
from qwen_latent_cot.bagel.loop_grpo import (
    _flow_kwargs,
    clone_loop_adapter_state,
    replay_group,
    replay_transition,
    temporary_loop_adapter_state,
)


def test_grpo_rollout_uses_base_k0_and_loop_k8():
    from scripts.train.bagel_loop_grpo_train import _rollout_num_loop_tokens

    config = {"num_loop_tokens": 8}
    assert _rollout_num_loop_tokens(config, base=True) == 0
    assert _rollout_num_loop_tokens(config, base=False) == 8


def test_grpo_accepts_phase1_v8_adapter_contract(tmp_path):
    import json

    from scripts.train.bagel_loop_grpo_train import _validate_adapter_contract

    adapter = tmp_path / "phase1.safetensors"
    metadata = {
        "schema": "bagel_loop_delta_velocity_adapter_v8",
        "objective": "structured_reflection_delta_velocity_distillation",
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "round0_memory_write_enabled": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "gen_attention_o_lora": False,
        "k_v_lora": False,
    }
    adapter.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    config = {
        "adapter_path": str(adapter),
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "round0_memory_write_enabled": False,
        "lora_rank": 8,
        "lora_alpha": 16,
        "gen_attention_o_lora": False,
        "k_v_lora": False,
    }
    assert _validate_adapter_contract(config) == metadata

    config["num_loop_tokens"] = 4
    with pytest.raises(RuntimeError, match="num_loop_tokens"):
        _validate_adapter_contract(config)


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


def test_replay_uses_memory_loop_when_packed_loop_indexes_are_set():
    class Policy(_TinyPolicy):
        def __init__(self):
            super().__init__()
            self.calls = []

        def _forward_flow(self, x_t, **kwargs):
            self.calls.append("vanilla")
            return super()._forward_flow(x_t, **kwargs)

        def _forward_flow_loop(self, x_t, **kwargs):
            self.calls.append("loop")
            assert kwargs["packed_loop_token_indexes"].numel() > 0
            assert kwargs.get("memory_body_in") is not None
            return super()._forward_flow(x_t, **kwargs)

    policy = Policy()
    reference = clone_loop_adapter_state(policy)
    context = _context()
    context["packed_loop_token_indexes"] = torch.tensor([1, 2], dtype=torch.long)
    context["embed_memory"] = torch.zeros(2, 1)
    context["recycle_mode"] = "same_depth"
    context["memory_loop_repeat"] = 2
    context["memory_loop_start"] = 1
    context["memory_loop_end"] = 3
    transition = _rollout(policy)
    transition["m_in"] = torch.zeros(2, 1)
    policy.calls.clear()
    replay_transition(
        policy,
        transition,
        context,
        reference,
        advantage=torch.tensor(1.0),
        clip_range=1e-4,
        kl_beta=0.0,
    )
    assert "loop" in policy.calls
    assert "vanilla" not in policy.calls


def test_replay_prefers_canonical_round0_write_flag_and_accepts_legacy():
    context = _context()
    context["packed_loop_token_indexes"] = torch.tensor([1], dtype=torch.long)
    context["embed_memory"] = torch.zeros(1, 1)
    context["round0_gen_reads_memory"] = True
    legacy = _flow_kwargs(context, torch.tensor([0.9]))
    assert legacy["round0_memory_write_enabled"] is True

    context["round0_memory_write_enabled"] = False
    canonical = _flow_kwargs(context, torch.tensor([0.9]))
    assert canonical["round0_memory_write_enabled"] is False


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
