from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from bagel_loop_zeroshot import (  # noqa: E402
    ARMS,
    NOTEBOOK_T2I_HYPER,
    apply_loop_config,
    select_arms,
    shard_indices,
    summarize_diagnostics,
)


def test_arm_table_matches_phase0_protocol():
    by_id = {arm["id"]: arm for arm in ARMS}
    assert list(by_id) == ["A0", "A1", "A2", "A3", "A4", "A5"]
    assert by_id["A0"]["K"] == 0 and by_id["A0"]["R"] == 1
    assert by_id["A1"]["K"] == 8 and by_id["A1"]["R"] == 1
    assert by_id["A2"]["K"] == 8
    assert by_id["A2"]["R"] == 2
    assert by_id["A2"]["recycle_mode"] == "same_depth"
    assert by_id["A2"]["persist"] is True
    assert by_id["A2"]["start_layer"] == 20
    assert by_id["A2"]["end_layer"] == 28
    assert by_id["A3"]["R"] == 3 and by_id["A3"]["recycle_mode"] == "same_depth"
    assert by_id["A4"]["recycle_mode"] == "full_depth" and by_id["A4"]["R"] == 2
    assert by_id["A5"]["persist"] is False and by_id["A5"]["recycle_mode"] == "same_depth"


def test_notebook_hyper_keeps_cfg_open_and_avoids_extra_knobs():
    assert NOTEBOOK_T2I_HYPER["cfg_interval"] == [0.0, 1.0]
    assert NOTEBOOK_T2I_HYPER["cfg_text_scale"] == 4.0
    assert NOTEBOOK_T2I_HYPER["cfg_img_scale"] == 1.0
    assert NOTEBOOK_T2I_HYPER["num_timesteps"] == 50
    assert NOTEBOOK_T2I_HYPER["timestep_shift"] == 3.0
    assert NOTEBOOK_T2I_HYPER["cfg_renorm_type"] == "global"
    for banned in (
        "enable_taylorseer",
        "think",
        "loop_state_scale",
        "sde_noise_level",
        "n_min",
        "n_max",
    ):
        assert banned not in NOTEBOOK_T2I_HYPER


def test_apply_loop_config_writes_bagelconfig_fields():
    memory = torch.nn.Parameter(torch.zeros(8, 4))
    model = SimpleNamespace(
        config=SimpleNamespace(),
        loop_memory=memory,
    )
    apply_loop_config(model, select_arms("A2")[0])
    assert model.config.num_loop_tokens == 8
    assert model.config.loop_depth == 2
    assert model.config.loop_recycle_mode == "same_depth"
    assert model.config.loop_memory_persist is True
    assert model.config.memory_loop_start_layer == 20
    assert model.config.memory_loop_end_layer == 28
    apply_loop_config(model, select_arms("A0")[0])
    assert model.config.num_loop_tokens == 0
    assert model.config.loop_depth == 1
    apply_loop_config(model, select_arms("A5")[0])
    assert model.config.loop_memory_persist is False
    apply_loop_config(model, select_arms("A4")[0])
    assert model.config.loop_recycle_mode == "full_depth"


def test_one_prompt_per_shard_round_robin():
    assert shard_indices(16, 0, 16) == [0]
    assert shard_indices(16, 15, 16) == [15]
    assert shard_indices(16, 3, 8) == [3, 11]


def test_empty_diagnostics_for_vanilla_path():
    summary = summarize_diagnostics([])
    assert summary["n_steps"] == 0
    assert summary["n_inner"] == 0
