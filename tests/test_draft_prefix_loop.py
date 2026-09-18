from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluate"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from draft_prefix_loop import (  # noqa: E402
    compose_gen_text,
    is_usable_reflection,
    parse_edit_stop,
    parse_gen_text_modes,
    says_already_correct,
)
from score_draft_prefix_geneval2 import _maps  # noqa: E402


def test_rejects_template_echo_from_v4():
    echo = (
        "Mismatches:\n- (one specific error, e.g. five pigs instead of four)\n"
        "INSTRUCTION: (one imperative sentence that fixes that error)"
    )
    assert is_usable_reflection(echo) is False
    assert is_usable_reflection("INSTRUCTION: <no changes needed>") is False
    assert is_usable_reflection("No fix needed.") is False


def test_accepts_plain_official_style_answer():
    text = (
        "There are four pigs, but the target asks for four pigs and three bagels. "
        "The bagels are missing. Add three white bagels behind the pigs."
    )
    assert is_usable_reflection(text) is True
    assert compose_gen_text("a_only", "target prompt", text) == text


def test_p002_style_already_correct_stops():
    text = (
        "The image shows five wooden rabbits in front of three pink croissants, "
        "and two purple zebras. The object counts and positions are correct "
        "compared to the target description. There are no issues with the "
        "object counts or positions."
    )
    assert says_already_correct(text) is True
    assert is_usable_reflection(text) is False


def test_p_plus_a_keeps_prompt_and_instruction():
    instruction = "Add three white bagels behind the pigs."
    assert compose_gen_text("p_plus_a", "four pigs", instruction) == (
        "four pigs\n\nEdit instruction: Add three white bagels behind the pigs."
    )
    assert compose_gen_text("a_only", "four pigs", instruction) == instruction


def test_edit_stop_defaults_to_prefix():
    assert parse_edit_stop("prefix") == "prefix"
    assert parse_edit_stop("full") == "full"
    with pytest.raises(ValueError, match="edit-stop"):
        parse_edit_stop("t_A")


def test_default_mode_is_a_only():
    assert parse_gen_text_modes("a_only,p_plus_a") == ["a_only", "p_plus_a"]
    with pytest.raises(ValueError, match="invalid"):
        parse_gen_text_modes("same")


def test_score_maps_cover_r0_r1_r3_and_both_modes(tmp_path: Path):
    prompts = ["pigs", "rabbits"]
    snapshots = _maps(tmp_path, prompts, "t0.8", ["a_only", "p_plus_a"])
    assert set(snapshots) == {
        "baseline",
        "r0",
        "a_only_r1",
        "a_only_r3",
        "a_only_final",
        "p_plus_a_r1",
        "p_plus_a_r3",
        "p_plus_a_final",
    }
    assert snapshots["r0"]["pigs"] == tmp_path / "p000" / "draft_t0.8_r0.png"
    assert snapshots["a_only_r1"]["rabbits"] == tmp_path / "p001" / "draft_t0.8_a_only_r1.png"
    assert snapshots["p_plus_a_r3"]["pigs"] == tmp_path / "p000" / "draft_t0.8_p_plus_a_r3.png"


def test_hard_16_jsonl_matches_prompt_file():
    root = Path(__file__).resolve().parents[1] / "experiments" / "data"
    prompts = [
        line.strip()
        for line in (root / "geneval2_hard_16.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in (root / "geneval2_hard_16.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(prompts) == 16
    assert [row["prompt"] for row in rows] == prompts
    assert all(row.get("vqa_list") for row in rows)
