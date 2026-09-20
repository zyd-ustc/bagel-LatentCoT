from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from bagel_loop_t2i_zeroshot import (  # noqa: E402
    ARMS,
    T2I_HYPER,
    apply_loop_config,
    load_prompts,
    official_t2i,
    select_arms,
    shard_indices,
    summarize_diagnostics,
)


def test_t2i_hyper_matches_official_native_generation():
    assert T2I_HYPER == {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
    }


def test_t2i_arm_matrix_has_strict_read_and_direct_write_controls():
    assert [arm["id"] for arm in ARMS] == [
        "Z0",
        "Z1",
        "Z2",
        "Z3",
        "Z4",
        "Z5",
        "C0",
        "C1",
        "C2",
    ]
    assert all("remove_old_prompt" not in arm for arm in ARMS)
    assert all("source_image" not in arm for arm in ARMS)
    c1 = select_arms("C1")[0]
    assert c1["R"] == 1
    assert c1["round0_memory_write_enabled"] is False
    c2 = select_arms("C2")[0]
    assert c2["R"] == 2
    assert c2["round0_memory_write_enabled"] is True


def test_c1_and_c2_each_change_one_z2_read_write_variable():
    z2 = select_arms("Z2")[0]
    c1 = select_arms("C1")[0]
    c2 = select_arms("C2")[0]
    fields = (
        "K",
        "R",
        "recycle_mode",
        "persist",
        "start_layer",
        "end_layer",
        "round0_memory_write_enabled",
    )
    assert {key for key in fields if z2[key] != c1[key]} == {"R"}
    assert {key for key in fields if z2[key] != c2[key]} == {
        "round0_memory_write_enabled"
    }


def test_official_t2i_uses_prompt_only_and_fixed_noise():
    calls = []
    expected_image = object()

    class Inferencer:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {"image": expected_image}

    noise = torch.zeros(4, 8)
    result = official_t2i(Inferencer(), "three red cubes", noise, (64, 96))
    assert result is expected_image
    assert calls[0]["image"] is None
    assert calls[0]["text"] == "three red cubes"
    assert calls[0]["image_shapes"] == (64, 96)
    assert calls[0]["init_noise"] is noise
    assert calls[0]["cfg_img_scale"] == 1.0
    assert calls[0]["return_loop_diagnostics"] is True


def test_apply_t2i_loop_config_and_vanilla_path():
    model = SimpleNamespace(
        config=SimpleNamespace(),
        loop_memory=torch.nn.Parameter(torch.zeros(8, 4)),
    )
    apply_loop_config(model, select_arms("Z2")[0])
    assert model.num_loop_tokens == 8
    assert model.loop_depth == 2
    assert model.num_read_rounds == 1
    assert model.num_write_rounds == 1
    apply_loop_config(model, select_arms("Z0")[0])
    assert model.num_loop_tokens == 0
    assert model.loop_depth == 1
    apply_loop_config(model, select_arms("C1")[0])
    assert model.num_read_rounds == 1
    assert model.num_write_rounds == 0
    apply_loop_config(model, select_arms("C2")[0])
    assert model.num_read_rounds == 0
    assert model.num_write_rounds == 2


def test_prompt_loading_and_sharding(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("one\n# comment\ntwo\nthree\n", encoding="utf-8")
    assert load_prompts(str(path), 2) == ["one", "two"]
    assert shard_indices(5, 1, 2) == [1, 3]


def test_empty_t2i_diagnostics_are_explicit():
    summary = summarize_diagnostics([])
    assert summary["n_steps"] == 0
    assert summary["mean_delta_m"] is None
    assert summary["mean_delta_g"] is None
    assert summary["mean_delta_v"] is None
