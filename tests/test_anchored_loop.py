"""Anchored dual-state contract. The heavy checks live in the package self-check."""
import os

import pytest

from qwen_latent_cot.anchored.cli import main
from qwen_latent_cot.anchored.self_check import run_self_check


def test_anchored_contract():
    names = run_self_check()
    assert "exact_bypass_and_cache" in names
    assert "cfg_matches_bagel" in names
    assert "gradients_and_frozen_native" in names
    assert "trained_observation_is_an_independent_copy" in names
    assert "observation_source_mode_follows_the_training_config" in names
    assert "anchored_observation_reuses_native_pre_kv" in names
    assert "image_tokens_address_memory_slots" in names
    assert "inference_reads_the_trained_gate" in names
    assert "written_residual_respects_the_rms_cap" in names
    assert "observation_query_follows_the_verified_writer" in names
    assert "gen_only_does_not_carry_memory" in names
    assert "round_velocities_and_supervision" in names


def test_show_config():
    main(["show-config", "--config", "configs/anchored_dual_state_loop.json"])


def test_legacy_training_entries_are_locked(monkeypatch):
    monkeypatch.delenv("BAGEL_ALLOW_LEGACY_MEMORY_SFT", raising=False)
    from qwen_latent_cot.diagnostics.visual_cli import main as visual_main
    from qwen_latent_cot.interaction.cli import main as interaction_main
    with pytest.raises(RuntimeError, match="retired"):
        visual_main(["train", "--model", "unused", "--output", "unused"])
    with pytest.raises(RuntimeError, match="retired"):
        interaction_main(["sft", "--model", "unused", "--parent", "unused", "--train", "a", "--val", "b", "--output", "c"])
    monkeypatch.setenv("BAGEL_ALLOW_LEGACY_MEMORY_SFT", "1")
    from qwen_latent_cot.anchored.legacy import refuse_legacy_memory_sft
    refuse_legacy_memory_sft("visual-train")
    os.environ.pop("BAGEL_ALLOW_LEGACY_MEMORY_SFT", None)
