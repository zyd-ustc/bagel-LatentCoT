from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from scripts.train.bagel_loop_delta_v_distill import (
    _backward_scaled_replay_loss,
    load_phase1_records,
)


def _row(source_image):
    return {
        "id": "edit_000001",
        "source_image": source_image,
        "instruction": "Add one red cube.",
        "reflection": (
            "EDIT PLAN\n"
            "Target changes:\n"
            "- Increase the red cube count to three.\n\n"
            "Preserve:\n"
            "- Keep the existing objects and background unchanged."
        ),
        "edit_type": ["addition", "count"],
        "target_constraints": ["red_cube_count=3"],
        "preserve_constraints": ["background"],
        "is_noop": False,
        "difficulty": 2,
        "teacher_valid": True,
        "teacher_semantic_delta": 0.15,
        "teacher_preserve_delta": -0.01,
    }


def test_phase1_jsonl_resolves_source_relative_to_manifest(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16)).save(images / "source.png")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(_row("images/source.png")) + "\n", encoding="utf-8"
    )
    records = load_phase1_records(str(manifest))
    assert records[0]["source_image"] == str((images / "source.png").resolve())


def test_phase1_jsonl_rejects_freeform_or_overlong_reflection(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "source.png")
    row = _row("source.png")
    row["reflection"] = "First I should inspect and think about the image."
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Target changes"):
        load_phase1_records(str(manifest))

    row = _row("source.png")
    row["reflection"] = (
        "EDIT PLAN\nTarget changes:\n- "
        + "word " * 121
        + "\nPreserve:\n- Keep everything else."
    )
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="120 English tokens"):
        load_phase1_records(str(manifest))


def test_phase1_jsonl_requires_edit_plan_header(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "source.png")
    row = _row("source.png")
    row["reflection"] = row["reflection"].replace("EDIT PLAN\n", "")
    manifest = tmp_path / "bad_header.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start with EDIT PLAN"):
        load_phase1_records(str(manifest))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("edit_type", "edit_type must be non-empty"),
        ("target_constraints", "non-noop target_constraints must be non-empty"),
        ("preserve_constraints", "preserve_constraints must be non-empty"),
    ],
)
def test_non_noop_requires_semantic_contract_lists(tmp_path, field, message):
    Image.new("RGB", (16, 16)).save(tmp_path / "source.png")
    row = _row("source.png")
    row[field] = []
    manifest = tmp_path / f"bad_{field}.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_phase1_records(str(manifest))


def test_noop_requires_explicit_no_change_language(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "source.png")
    row = _row("source.png")
    row["is_noop"] = True
    row["target_constraints"] = []
    manifest = tmp_path / "noop.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="noop reflection"):
        load_phase1_records(str(manifest))

    row["reflection"] = row["reflection"].replace(
        "Increase the red cube count to three.",
        "No structural change is required.",
    )
    row["teacher_semantic_delta"] = 0.0
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_phase1_records(str(manifest))[0]["is_noop"] is True


def test_teacher_valid_filter_and_score_contract(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "source.png")
    valid = _row("source.png")
    invalid = _row("source.png")
    invalid["id"] = "edit_invalid_teacher"
    invalid["teacher_valid"] = False
    invalid["teacher_semantic_delta"] = -0.2
    manifest = tmp_path / "teachers.jsonl"
    manifest.write_text(
        json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )
    assert [row["id"] for row in load_phase1_records(str(manifest))] == [
        "edit_000001"
    ]
    assert len(load_phase1_records(str(manifest), teacher_valid_only=False)) == 2

    invalid["teacher_valid"] = True
    manifest.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with teacher score"):
        load_phase1_records(str(manifest))


def test_sequential_state_backward_matches_mean_loss_gradient():
    coefficients = (0.5, 1.5, -0.75, 2.0)
    reference = torch.tensor(0.4, requires_grad=True)
    reference_losses = [
        (reference * coefficient - 0.3).square()
        for coefficient in coefficients
    ]
    reference_mean = torch.stack(reference_losses).mean()
    reference_mean.backward()

    sequential = torch.tensor(0.4, requires_grad=True)
    detached_losses = []
    for coefficient in coefficients:
        loss = (sequential * coefficient - 0.3).square()
        detached_losses.append(
            _backward_scaled_replay_loss(loss, state_count=len(coefficients))
        )
    assert torch.allclose(sequential.grad, reference.grad, atol=1e-7)
    assert torch.allclose(torch.stack(detached_losses).mean(), reference_mean.detach())
