from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from bagel_write_sensitivity_t2i import (  # noqa: E402
    ARM_SOURCES,
    paired_rows,
    prepare_pair,
    summarize_probe,
    validate_protocol,
)
from qwen_latent_cot.bagel.write_sensitivity import (  # noqa: E402
    append_write_probe,
    select_write_memory,
)
from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel  # noqa: E402


def contract():
    return {
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "loop_recycle_mode": "same_depth",
        "loop_memory_persist": False,
        "round0_memory_write_enabled": False,
        "loop_uncond_memory": "m0",
    }


def args():
    return SimpleNamespace(
        max_prompts=16, height=512, width=512,
        num_steps=50, timestep_shift=3.0,
        cfg_text_scale=4.0, cfg_img_scale=1.0,
    )


def test_protocol_fixes_architecture_and_real_pairs():
    validate_protocol(contract(), args())
    assert paired_rows([{"prompt": str(i)} for i in range(4)]) == [
        [{"prompt": "0"}, {"prompt": "1"}],
        [{"prompt": "2"}, {"prompt": "3"}],
    ]
    assert list(ARM_SOURCES.values()) == ["correct", "shuffle", "m0", "zero"]
    bad = contract()
    bad["loop_depth"] = 1
    with pytest.raises(ValueError, match="strict Read"):
        validate_protocol(bad, args())
    bad_args = args()
    bad_args.max_prompts = 1
    with pytest.raises(ValueError, match="even"):
        validate_protocol(contract(), bad_args)


def test_write_memory_swaps_whole_samples_not_slots():
    read = torch.tensor([[11.0], [12.0], [21.0], [22.0]])  # B=2, K=2
    initial = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert select_write_memory(read, initial, source="correct", batch_size=2) is read
    assert torch.equal(
        select_write_memory(read, initial, source="shuffle", batch_size=2),
        torch.tensor([[21.0], [22.0], [11.0], [12.0]]),
    )
    assert select_write_memory(read, initial, source="m0", batch_size=2) is initial
    assert torch.count_nonzero(select_write_memory(read, initial, source="zero", batch_size=2)) == 0
    assert torch.equal(read, torch.tensor([[11.0], [12.0], [21.0], [22.0]]))


def test_shuffle_is_derangement_and_single_sample_fails():
    read = torch.arange(6.0).reshape(6, 1)
    initial = torch.zeros_like(read)
    shuffled = select_write_memory(read, initial, source="shuffle", batch_size=3)
    assert torch.equal(shuffled.reshape(3, 2, 1)[0], read.reshape(3, 2, 1)[2])
    assert all(not torch.equal(shuffled.reshape(3, 2, 1)[i], read.reshape(3, 2, 1)[i]) for i in range(3))
    with pytest.raises(ValueError, match="batch size"):
        select_write_memory(read, initial, source="shuffle", batch_size=1)
    with pytest.raises(ValueError, match="matching"):
        select_write_memory(read, initial[:2], source="m0", batch_size=3)


def test_probe_records_read_m0_and_used_memory_without_svd():
    read = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    initial = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    used = select_write_memory(read, initial, source="shuffle", batch_size=2)
    rows = []
    append_write_probe(rows, read, initial, used, batch_size=2)
    assert len(rows) == 2
    assert rows[0]["read_l2"] == pytest.approx(1.0)
    assert rows[0]["m0_l2"] == pytest.approx(1.0)
    assert rows[0]["read_m0_cos"] == pytest.approx(0.0)
    assert summarize_probe(rows, 0, 1)["read_l2"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="expected"):
        summarize_probe(rows, 0, 2)


def test_sample_global_cfg_keeps_other_sample_out_of_norm():
    reference = torch.tensor([[1.0], [1.0]])
    text_branch = torch.tensor([[0.0], [2.0]])
    result = Bagel._combine_cfg_velocities(
        None, reference, text_branch, None,
        cfg_text_scale=4.0, cfg_img_scale=1.0,
        cfg_renorm_min=0.0, cfg_renorm_type="sample_global",
        vae_seqlens=torch.tensor([1, 1]),
    )
    assert torch.allclose(result, torch.tensor([[1.0], [-1.0]]))
    changed_other = Bagel._combine_cfg_velocities(
        None, reference, torch.tensor([[0.0], [20.0]]), None,
        cfg_text_scale=4.0, cfg_img_scale=1.0,
        cfg_renorm_min=0.0, cfg_renorm_type="sample_global",
        vae_seqlens=torch.tensor([1, 1]),
    )
    assert result[0] == changed_other[0]


def test_prepare_pair_uses_native_two_sample_packing():
    calls = []

    class FakeModel:
        config = SimpleNamespace(
            llm_config=SimpleNamespace(num_hidden_layers=1), num_loop_tokens=8,
        )

        def prepare_prompts(self, **kwargs):
            calls.append(("prompts", kwargs["prompts"]))
            return {"input": torch.tensor(1)}, [3, 5], [3, 5]

        def forward_cache_update_text(self, cache, **kwargs):
            return cache

        def prepare_vae_latent(self, **kwargs):
            calls.append(("images", kwargs["image_sizes"]))
            return {"packed_init_noises": torch.zeros(4, 3)}

        def prepare_vae_latent_cfg(self, **kwargs):
            return {
                "cfg_packed_position_ids": torch.zeros(1),
                "cfg_packed_query_indexes": torch.zeros(1),
                "cfg_key_values_lens": torch.zeros(2),
                "cfg_packed_key_value_indexes": torch.zeros(1),
            }

    inferencer = SimpleNamespace(
        model=FakeModel(), device=torch.device("cpu"),
        tokenizer=None, new_token_ids={},
    )
    bundle = prepare_pair(
        inferencer,
        [{"prompt": "first"}, {"prompt": "second"}],
        [torch.ones(2, 3), torch.full((2, 3), 2.0)],
        (512, 512),
    )
    assert calls == [
        ("prompts", ["first", "second"]),
        ("images", [(512, 512), (512, 512)]),
    ]
    assert torch.equal(bundle["packed_init_noises"][:2], torch.ones(2, 3))
    assert torch.equal(bundle["packed_init_noises"][2:], torch.full((2, 3), 2.0))
