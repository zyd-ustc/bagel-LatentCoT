from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from bagel_single_pass_prompt_kv_t2i import (  # noqa: E402
    MEMORY_UPDATE_END,
    POLICIES,
    generate,
    validate_protocol,
)


def _contract():
    return {
        "num_loop_tokens": 8,
        "loop_recycle_mode": "same_depth",
        "loop_memory_persist": False,
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
    }


def _args():
    return SimpleNamespace(
        max_prompts=4, height=512, width=512,
        num_steps=50, timestep_shift=3.0,
    )


def test_protocol_uses_training_mask_range_and_one_pass():
    validate_protocol(_contract(), _args())
    assert MEMORY_UPDATE_END == 16
    assert POLICIES == ("keep", "mask_body")
    bad = _contract()
    bad["memory_loop_end_layer"] = 16
    with pytest.raises(ValueError, match="overlapping memory update"):
        validate_protocol(bad, _args())


def test_two_policies_use_identical_input_noise_and_only_differ_by_mask():
    calls = []

    def inferencer(**kwargs):
        calls.append(kwargs)
        kwargs["init_noise"].add_(100)
        return {"image": object()}

    noise = torch.arange(4.0)
    for policy in POLICIES:
        generate(inferencer, "test prompt", noise, _args(), mask_body=policy == "mask_body")
    assert torch.equal(noise, torch.arange(4.0))
    assert torch.equal(calls[0]["init_noise"], calls[1]["init_noise"])
    assert calls[0]["single_pass_memory_update_end"] == 16
    assert calls[1]["single_pass_memory_update_end"] == 16
    assert calls[0]["single_pass_mask_prompt_kv"] is False
    assert calls[1]["single_pass_mask_prompt_kv"] is True
    for key in set(calls[0]) - {"init_noise", "single_pass_mask_prompt_kv"}:
        assert calls[0][key] == calls[1][key]
