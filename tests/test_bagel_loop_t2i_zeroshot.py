from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluate"))

from bagel_loop_t2i_zeroshot import (  # noqa: E402
    ARMS,
    T2I_HYPER,
    aggregate_mechanism_rows,
    apply_loop_config,
    load_prompts,
    make_k_ablation_arms,
    official_t2i,
    parse_k_values,
    resolve_arms,
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


def test_t2i_arm_matrix_keeps_selected_body_and_persistence_ablations():
    assert [arm["id"] for arm in ARMS] == [
        "Z0",
        "Z2",
        "Z3",
        "Z4",
        "Z6",
    ]
    assert all("remove_old_prompt" not in arm for arm in ARMS)
    assert all("source_image" not in arm for arm in ARMS)
    assert all(select_arms(arm_id)[0]["persist"] is False for arm_id in ("Z2", "Z3", "Z4"))
    z3 = select_arms("Z3")[0]
    z6 = select_arms("Z6")[0]
    assert z6["persist"] is True
    assert {
        key
        for key in z3
        if key not in {"id", "slug", "title"} and z3[key] != z6[key]
    } == {"persist"}


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
    apply_loop_config(model, select_arms("Z6")[0])
    assert model.num_read_rounds == 1
    assert model.num_write_rounds == 1
    assert model.loop_memory_persist is True


def test_prompt_loading_and_sharding(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("one\n# comment\ntwo\nthree\n", encoding="utf-8")
    assert load_prompts(str(path), 2) == ["one", "two"]
    assert shard_indices(5, 1, 2) == [1, 3]


def test_k_ablation_is_z0_plus_strict_mid_body_scaling():
    assert parse_k_values("1,4,8,4") == [1, 4, 8]
    arms = make_k_ablation_arms([1, 4, 8])
    assert [arm["id"] for arm in arms] == ["Z0", "K1", "K4", "K8"]
    assert [arm["K"] for arm in arms] == [0, 1, 4, 8]
    assert all(arm["R"] in (1, 2) for arm in arms)
    assert all(arm["start_layer"] == 16 for arm in arms)
    assert all(arm["end_layer"] == 24 for arm in arms)
    assert all(arm["round0_memory_write_enabled"] is False for arm in arms)
    assert [arm["id"] for arm in resolve_arms("", "1,4,8")] == [
        "Z0",
        "K1",
        "K4",
        "K8",
    ]


def test_mechanism_aggregation_keeps_deltas_separate_from_mae():
    arms = select_arms("Z0,Z2")
    rows = [
        {
            "pixel_mae_vs_Z0": {"Z2": 12.0},
            "arms": [
                {"id": "Z0", "diagnostics": {}},
                {
                    "id": "Z2",
                    "diagnostics": {
                        "mean_delta_m": 0.2,
                        "mean_delta_g": 0.1,
                        "mean_delta_v": 0.05,
                        "mean_effective_rank": 3.0,
                    },
                },
            ],
        },
        {
            "pixel_mae_vs_Z0": {"Z2": 16.0},
            "arms": [
                {"id": "Z0", "diagnostics": {}},
                {
                    "id": "Z2",
                    "diagnostics": {
                        "mean_delta_m": 0.4,
                        "mean_delta_g": 0.2,
                        "mean_delta_v": 0.15,
                        "mean_effective_rank": 5.0,
                    },
                },
            ],
        },
    ]
    summary = aggregate_mechanism_rows(rows, arms)
    by_id = {row["id"]: row for row in summary["arms"]}
    assert by_id["Z0"]["mean_pixel_mae_vs_Z0"] == 0.0
    assert by_id["Z2"]["mean_pixel_mae_vs_Z0"] == 14.0
    assert by_id["Z2"]["mean_delta_m"] == pytest.approx(0.3)
    assert by_id["Z2"]["mean_delta_v"] == pytest.approx(0.1)


def test_empty_t2i_diagnostics_are_explicit():
    summary = summarize_diagnostics([])
    assert summary["n_steps"] == 0
    assert summary["mean_delta_m"] is None
    assert summary["mean_delta_g"] is None
    assert summary["mean_delta_v"] is None
