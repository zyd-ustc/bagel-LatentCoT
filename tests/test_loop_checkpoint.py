from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_latent_cot.bagel.loop import (
    LoopLoRALinear,
    load_loop_adapter_state_dict,
    loop_adapter_state_dict,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = LoopLoRALinear(nn.Linear(3, 2), rank=1, alpha=2)


class ExpandedTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj_moe_gen = LoopLoRALinear(
            nn.Linear(3, 2), rank=1, alpha=2
        )
        self.attn.k_proj = LoopLoRALinear(nn.Linear(3, 2), rank=1, alpha=2)


def test_loop_adapter_round_trip_is_strict():
    source = Tiny()
    with torch.no_grad():
        source.proj.lora_A.weight.fill_(3.0)
        source.proj.lora_B.weight.fill_(4.0)
    state = loop_adapter_state_dict(source)
    target = Tiny()
    load_loop_adapter_state_dict(target, state)
    assert torch.equal(target.proj.lora_A.weight, source.proj.lora_A.weight)
    assert torch.equal(target.proj.lora_B.weight, source.proj.lora_B.weight)
    with pytest.raises(RuntimeError, match="key mismatch"):
        load_loop_adapter_state_dict(target, {"bad": torch.ones(1)})


def test_gen_only_checkpoint_can_initialize_expanded_text_kv_policy():
    target = ExpandedTiny()
    state = {
        name: tensor
        for name, tensor in loop_adapter_state_dict(target).items()
        if ".q_proj_moe_gen." in name
    }
    missing = load_loop_adapter_state_dict(
        target, state, allow_missing_projections=("k_proj", "v_proj")
    )
    assert len(missing) == 2
    assert all(".k_proj." in name for name in missing)
