from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate"))

from bagel_hard16_checkpoint import (  # noqa: E402
    generate,
    load_benchmark,
    resolve_contract,
    selected_arms,
    set_adapter,
    set_loop,
    validate_adapter_state,
)
from qwen_latent_cot.bagel.loop import LoopLoRALinear, loop_adapter_state_dict  # noqa: E402


def config():
    return {
        "num_loop_tokens": 8, "loop_depth": 2, "loop_recycle_mode": "same_depth",
        "loop_memory_persist": False, "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20, "round0_memory_write_enabled": False,
        "lora_rank": 8, "lora_alpha": 16,
    }


def test_phase11_read_round_is_not_generation_depth():
    contract = resolve_contract(config(), {
        "schema": "bagel_pair_grounded_memory_adapter_v1", "K": 8,
        "R": 1, "body": [12, 20],
    })
    assert contract["loop_depth"] == 2
    assert contract["loop_uncond_memory"] == "m0"
    with pytest.raises(ValueError, match="body"):
        resolve_contract(config(), {"body": [16, 24]})
    with pytest.raises(ValueError, match="R"):
        resolve_contract(config(), {"R": 3})


def test_base_and_both_loop_arms_share_training_body():
    model = SimpleNamespace(config=SimpleNamespace())
    contract = resolve_contract(config(), {})
    set_loop(model, contract, False)
    assert model.num_loop_tokens == 0
    assert model.loop_depth == 1
    set_loop(model, contract, True)
    assert model.num_loop_tokens == 8
    assert model.loop_depth == 2
    assert (model.memory_loop_start_layer, model.memory_loop_end_layer) == (12, 20)
    assert (model.num_read_rounds, model.num_write_rounds) == (1, 1)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = LoopLoRALinear(nn.Linear(3, 2), rank=1, alpha=2)
        self.attn.q_proj_moe_gen = LoopLoRALinear(nn.Linear(3, 2), rank=1, alpha=2)


def test_training_free_zero_residual_then_restore_trained():
    model = Tiny()
    with torch.no_grad():
        model.attn.q_proj.lora_B.weight.fill_(2)
        model.attn.q_proj_moe_gen.lora_B.weight.fill_(3)
    full = loop_adapter_state_dict(model)
    read_only = {key: value.clone() for key, value in full.items() if ".q_proj." in key}
    metadata = {"schema": "bagel_pair_grounded_memory_adapter_v1"}
    assert len(validate_adapter_state(read_only, model, metadata)) == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_adapter_state(read_only, model, {})
    set_adapter(model, None)
    assert torch.count_nonzero(model.attn.q_proj.lora_B.weight) == 0
    assert torch.count_nonzero(model.attn.q_proj_moe_gen.lora_B.weight) == 0
    set_adapter(model, read_only)
    assert torch.all(model.attn.q_proj.lora_B.weight == 2)
    assert torch.count_nonzero(model.attn.q_proj_moe_gen.lora_B.weight) == 0


def test_hard16_is_fixed_and_arms_are_unambiguous():
    path = Path(__file__).resolve().parents[1] / "experiments/data/geneval2_hard_16.jsonl"
    assert len(load_benchmark(path, 16)) == 16
    assert len(load_benchmark(path, 1)) == 1
    with pytest.raises(ValueError, match="subset"):
        selected_arms("base,base")


def test_generation_avoids_unused_npu_svd_diagnostics():
    calls = []

    def inferencer(**kwargs):
        calls.append(kwargs)
        return {"image": "image"}

    args = SimpleNamespace(
        cfg_text_scale=4.0, cfg_img_scale=1.0,
        num_steps=50, timestep_shift=3.0,
    )
    noise = torch.ones(2, 3)
    assert generate(inferencer, "prompt", noise, (512, 512), args) == "image"
    assert calls[0]["return_loop_diagnostics"] is False
    assert calls[0]["init_noise"] is not noise
    assert torch.equal(calls[0]["init_noise"], noise)
