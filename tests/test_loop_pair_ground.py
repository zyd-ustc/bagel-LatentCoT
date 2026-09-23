from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from PIL import Image

from qwen_latent_cot.bagel.loop_pair_ground import (
    pair_memory_loss,
    prepare_flow_training_state,
    sample_weighted_timestep,
)
from qwen_latent_cot.bagel.loop import (
    LoopLoRALinear,
    UND_Q_PROJECTIONS,
    configure_loop_trainable_routes,
)
from qwen_latent_cot.data.phase1_pairs import (
    STAGE_A_EDIT_TYPES,
    build_phase1_sampling_order,
    load_phase1_pairs,
)


def test_prepare_flow_training_state_matches_bagel_convention():
    clean = torch.tensor([[1.0, 3.0]])
    noise = torch.tensor([[5.0, 7.0]])
    x_t, velocity = prepare_flow_training_state(clean, 0.25, noise)
    assert torch.equal(x_t, torch.tensor([[2.0, 4.0]]))
    assert torch.equal(velocity, torch.tensor([[4.0, 4.0]]))


def test_pair_memory_direction_is_per_slot_and_differentiable():
    source = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    student = target.clone().requires_grad_(True)
    result = pair_memory_loss(
        student_memory=student,
        source_reference=source,
        target_reference=target,
        is_noop=False,
    )
    assert result.direction_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert result.relative_error.item() == pytest.approx(0.0, abs=1e-6)
    result.loss.backward()
    assert student.grad is not None


def test_noop_memory_loss_only_penalizes_student_delta():
    source = torch.zeros(2, 3)
    student = torch.ones(2, 3, requires_grad=True)
    result = pair_memory_loss(
        student_memory=student,
        source_reference=source,
        target_reference=source,
        is_noop=True,
    )
    assert result.loss.item() == pytest.approx(1.0)
    assert result.direction_loss.item() == 0.0
    result.loss.backward()
    assert float(student.grad.abs().sum()) > 0.0


def test_weighted_timestep_is_deterministic_and_bounded():
    first = sample_weighted_timestep(torch.Generator().manual_seed(7))
    second = sample_weighted_timestep(torch.Generator().manual_seed(7))
    assert first == second
    assert 0.0 <= first <= 1.0


def test_phase11_route_policy_trains_only_und_q_lora():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoopLoRALinear(nn.Linear(3, 3), rank=2, alpha=2)
            self.q_proj_moe_gen = LoopLoRALinear(
                nn.Linear(3, 3), rank=2, alpha=2, read_enabled=False
            )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    model = Model()
    names = configure_loop_trainable_routes(model, UND_Q_PROJECTIONS)
    assert names
    assert all(".q_proj." in name for name in names)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".q_proj_moe_gen." in name
    )


def test_pair_loader_ignores_teacher_gate_and_maps_noop_target(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (8, 8), "red").save(source)
    Image.new("RGB", (8, 8), "blue").save(target)
    rows = [
        {
            "id": "edit",
            "source_image": source.name,
            "target_image": target.name,
            "instruction": "Move the cube left.",
            "edit_type": ["move"],
            "is_noop": False,
            "teacher_valid": False,
        },
        {
            "id": "noop",
            "source_image": source.name,
            "instruction": "Keep the image unchanged.",
            "edit_type": ["noop"],
            "is_noop": True,
            "teacher_valid": False,
        },
    ]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    loaded = load_phase1_pairs(str(manifest))
    assert len(loaded) == 2
    assert loaded[1]["target_image"] == loaded[1]["source_image"]


def test_pair_loader_requires_target_for_edit(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "red").save(source)
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "bad",
                "source_image": source.name,
                "instruction": "Add a cube.",
                "edit_type": ["addition"],
                "is_noop": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_image"):
        load_phase1_pairs(str(manifest))


def test_stage_a_filter_excludes_mixed_later_stage_pair(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (8, 8), "red").save(source)
    Image.new("RGB", (8, 8), "blue").save(target)
    rows = [
        {
            "id": "stage-a",
            "source_image": source.name,
            "target_image": target.name,
            "instruction": "Move the cube.",
            "edit_type": ["move"],
            "is_noop": False,
        },
        {
            "id": "mixed",
            "source_image": source.name,
            "target_image": target.name,
            "instruction": "Move and replace the cube.",
            "edit_type": ["move", "replacement"],
            "is_noop": False,
        },
    ]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    loaded = load_phase1_pairs(
        str(manifest), allowed_edit_types=STAGE_A_EDIT_TYPES
    )
    assert [row["id"] for row in loaded] == ["stage-a"]


def test_sampling_order_enforces_twenty_percent_noop():
    records = [
        {"id": f"edit-{index}", "is_noop": False} for index in range(80)
    ] + [{"id": f"noop-{index}", "is_noop": True} for index in range(80)]
    ordered = build_phase1_sampling_order(records, noop_fraction=0.20, seed=7)
    assert len(ordered) == 100
    assert sum(bool(row["is_noop"]) for row in ordered) == 20
    assert [row["id"] for row in ordered] == [
        row["id"]
        for row in build_phase1_sampling_order(records, noop_fraction=0.20, seed=7)
    ]
