from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluate"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from edit_loop_zero_shot import (  # noqa: E402
    GOLD_TEXT,
    build_edit_kwargs,
    parse_instruction,
    select_edit_text,
)


def test_parse_instruction_extracts_first_line_after_marker():
    text = "MISMATCHES:\n- three bagels, not two\nINSTRUCTION: Add one more white bagel.\nIgnore this."
    assert parse_instruction(text) == "Add one more white bagel."


def test_parse_instruction_falls_back_to_full_text():
    assert parse_instruction("just make it darker") == "just make it darker"
    assert parse_instruction("INSTRUCTION:\n") == "INSTRUCTION:"


def test_select_edit_text_routes_controls():
    reflection = "MISMATCHES:\n- count\nINSTRUCTION: Add two pigs."
    assert select_edit_text("none", reflection, fixed_text="watercolor", gold_text="cars") == (
        "Add two pigs."
    )
    assert select_edit_text("no_text", reflection, fixed_text="watercolor", gold_text="cars") is None
    assert (
        select_edit_text("fixed", reflection, fixed_text="watercolor", gold_text="cars")
        == "watercolor"
    )
    assert select_edit_text("gold", reflection, fixed_text="watercolor", gold_text="cars") == "cars"


def test_select_edit_text_rejects_unknown_control():
    with pytest.raises(ValueError, match="unknown control"):
        select_edit_text("best_of_n", "x", fixed_text="a", gold_text="b")


def test_fresh_edit_omits_t2i_noise():
    t2i_noise = object()
    kwargs = build_edit_kwargs(
        image_shape=(512, 512),
        edit_text="Add one bagel.",
        t2i_noise=t2i_noise,
        edit_noise="fresh",
    )
    assert "init_noise" not in kwargs
    assert kwargs["text"] == "Add one bagel."
    assert kwargs["return_latent"] is True
    assert kwargs["cfg_renorm_type"] == "text_channel"
    assert kwargs["cfg_img_scale"] == 2.0


def test_same_edit_pins_t2i_noise():
    t2i_noise = object()
    kwargs = build_edit_kwargs(
        image_shape=(512, 512),
        edit_text=None,
        t2i_noise=t2i_noise,
        edit_noise="same",
    )
    assert kwargs["init_noise"] is t2i_noise
    assert "text" not in kwargs


def test_gold_default_is_a_strong_semantic_instruction():
    assert "zero animals" in GOLD_TEXT
    assert "red" in GOLD_TEXT
