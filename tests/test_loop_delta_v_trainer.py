from __future__ import annotations

import json

import pytest
from PIL import Image

from scripts.train.bagel_loop_delta_v_distill import load_phase1_records


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
