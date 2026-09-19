from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from bagel_loop_zeroshot import (  # noqa: E402
    ARMS,
    NOTEBOOK_EDIT_HYPER,
    apply_loop_config,
    load_edit_cases,
    make_k_ablation_arms,
    parse_k_values,
    resolve_arms,
    select_arms,
    shard_indices,
    summarize_diagnostics,
)


def test_arm_table_matches_read_route_write_protocol():
    expected = {
        "Z0": dict(K=0, R=1, recycle_mode="same_depth", persist=False, start_layer=16, end_layer=24, remove_old_prompt=True, round0_memory_write_enabled=False),
        "Z1": dict(K=8, R=2, recycle_mode="same_depth", persist=True, start_layer=20, end_layer=28, remove_old_prompt=False, round0_memory_write_enabled=True),
        "Z2": dict(K=8, R=2, recycle_mode="same_depth", persist=False, start_layer=16, end_layer=24, remove_old_prompt=True, round0_memory_write_enabled=False),
        "Z3": dict(K=8, R=2, recycle_mode="same_depth", persist=False, start_layer=12, end_layer=20, remove_old_prompt=True, round0_memory_write_enabled=False),
        "Z4": dict(K=8, R=2, recycle_mode="same_depth", persist=False, start_layer=20, end_layer=28, remove_old_prompt=True, round0_memory_write_enabled=False),
        "Z5": dict(K=8, R=2, recycle_mode="same_depth", persist=True, start_layer=16, end_layer=24, remove_old_prompt=True, round0_memory_write_enabled=False),
        "C0": dict(K=8, R=2, recycle_mode="full_depth", persist=False, start_layer=16, end_layer=24, remove_old_prompt=True, round0_memory_write_enabled=False),
    }
    by_id = {arm["id"]: arm for arm in ARMS}
    assert list(by_id) == list(expected)
    for arm_id, row in expected.items():
        for key, value in row.items():
            assert by_id[arm_id][key] == value, (arm_id, key)


def test_notebook_edit_hyper_matches_official_editing():
    assert NOTEBOOK_EDIT_HYPER["cfg_interval"] == [0.0, 1.0]
    assert NOTEBOOK_EDIT_HYPER["cfg_text_scale"] == 4.0
    assert NOTEBOOK_EDIT_HYPER["cfg_img_scale"] == 2.0
    assert NOTEBOOK_EDIT_HYPER["num_timesteps"] == 50
    assert NOTEBOOK_EDIT_HYPER["timestep_shift"] == 3.0
    assert NOTEBOOK_EDIT_HYPER["cfg_renorm_type"] == "text_channel"
    for banned in (
        "enable_taylorseer",
        "think",
        "loop_state_scale",
        "sde_noise_level",
        "n_min",
        "n_max",
    ):
        assert banned not in NOTEBOOK_EDIT_HYPER


def test_apply_loop_config_writes_bagelconfig_fields():
    memory = torch.nn.Parameter(torch.zeros(8, 4))
    model = SimpleNamespace(
        config=SimpleNamespace(),
        loop_memory=memory,
    )
    apply_loop_config(model, select_arms("Z2")[0])
    assert model.config.num_loop_tokens == 8
    assert model.config.loop_depth == 2
    assert model.config.loop_recycle_mode == "same_depth"
    assert model.config.loop_memory_persist is False
    assert model.config.memory_loop_start_layer == 16
    assert model.config.memory_loop_end_layer == 24
    assert model.config.round0_memory_write_enabled is False
    assert model.config.num_read_rounds == 1
    assert model.config.num_write_rounds == 1
    apply_loop_config(model, select_arms("Z0")[0])
    assert model.config.num_loop_tokens == 0
    assert model.config.loop_depth == 1
    apply_loop_config(model, select_arms("Z1")[0])
    assert model.config.loop_memory_persist is True
    assert model.config.round0_memory_write_enabled is True
    assert model.config.num_read_rounds == 0
    assert model.config.num_write_rounds == 2
    apply_loop_config(model, select_arms("C0")[0])
    assert model.config.loop_recycle_mode == "full_depth"


def test_one_prompt_per_shard_round_robin():
    assert shard_indices(16, 0, 16) == [0]
    assert shard_indices(16, 15, 16) == [15]
    assert shard_indices(16, 3, 8) == [3, 11]


def test_k_ablation_is_separate_and_uses_strict_mid_body():
    assert parse_k_values("1,4,8,4") == [1, 4, 8]
    arms = make_k_ablation_arms([1, 4, 8])
    assert [arm["id"] for arm in arms] == ["K1", "K4", "K8"]
    assert [arm["K"] for arm in arms] == [1, 4, 8]
    assert all(arm["start_layer"] == 16 for arm in arms)
    assert all(arm["end_layer"] == 24 for arm in arms)
    assert all(arm["round0_memory_write_enabled"] is False for arm in arms)
    assert [arm["id"] for arm in resolve_arms("", "1,4,8")] == ["K1", "K4", "K8"]
    try:
        resolve_arms("Z2", "1")
    except ValueError as exc:
        assert "separate protocols" in str(exc)
    else:
        raise AssertionError("mixed main arms and K ablation must fail")


def test_load_paired_edit_cases_resolves_images_and_validates_schema(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"E1","source_image":"images/a.png","source_prompt":"a red apple",'
        '"instruction":"Make it green.","target":"a green apple",'
        '"preserve":["apple shape"]}\n',
        encoding="utf-8",
    )
    cases = load_edit_cases(str(path))
    assert cases[0]["source_image"] == str((tmp_path / "images/a.png").resolve())
    assert cases[0]["instruction"] == "Make it green."
    assert cases[0]["preserve"] == ["apple shape"]


def test_removed_loop_state_kwargs_are_gone_from_public_generate():
    import inspect
    from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel

    for banned in ("loop_state_scale", "loop_state_mode"):
        assert banned not in inspect.signature(Bagel.generate_image).parameters
        assert banned not in inspect.signature(InterleaveInferencer.gen_image).parameters


def test_empty_diagnostics_for_vanilla_path():
    summary = summarize_diagnostics([])
    assert summary["n_steps"] == 0
    assert summary["n_inner"] == 0
    assert summary["mean_delta_m"] is None
