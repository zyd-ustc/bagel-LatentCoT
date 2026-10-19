"""Two-stage curriculum. The 64-card launcher is not started here."""
from pathlib import Path

import pytest
import torch

from qwen_latent_cot.anchored.anchored_train import (
    TRAINER,
    AnchoredTrainConfig,
    apply_stage_parameters,
    correction_rounds_for,
    load_checkpoint,
    require_stage2_reader,
    stage1_val_flags,
    stage2_val_flags,
    _resume_cursor,
)
from qwen_latent_cot.anchored.cli import main
from qwen_latent_cot.anchored.config import LoopConfig
from qwen_latent_cot.anchored.gen_only_train import GenOnlyTrainConfig, load_checkpoint as load_gen_only
from qwen_latent_cot.anchored.self_check import make_loop


ROOT = Path(__file__).resolve().parents[1]


def test_curriculum_config_is_two_stages_on_the_gen_only_window():
    train = AnchoredTrainConfig.from_json(ROOT / "configs/anchored_curriculum_train.json")
    train.validate()
    loop = LoopConfig.from_json(ROOT / "configs/anchored_gen_only.json")
    old = GenOnlyTrainConfig.from_json(ROOT / "configs/anchored_gen_only_train.json")
    assert old.steps == 500 and old.learning_rate == 1e-5
    assert [stage.name for stage in train.stages] == ["gen_only", "dual"]
    assert train.stages[0].steps == 400
    assert train.stages[1].steps == 600
    assert train.stages[0].learning_rate == pytest.approx(1e-4)
    assert train.stages[0].output_learning_rate == pytest.approx(2e-5)
    assert train.stages[1].output_learning_rate == pytest.approx(2e-5)
    assert train.lambda_improve == pytest.approx(1.0)
    assert train.improvement_margin == pytest.approx(1e-4)
    assert train.val_images == 128
    assert stage1_val_flags(1.0, 0.9) == {"v1_below_v0": True}
    assert stage1_val_flags(1.0, 1.0) == {"v1_below_v0": False}
    assert stage2_val_flags(0.4, 0.8, 0.9, 0.7, shuffled_measured=True) == {
        "dual_below_gen_only": True,
        "zero_rose": True,
        "shuffled_rose": True,
        "shuffled_measured": True,
    }
    assert stage2_val_flags(0.4, 0.8, 0.4, 0.7, shuffled_measured=False)["shuffled_rose"] is False
    assert correction_rounds_for(train.stages[0], 400) == 1
    assert correction_rounds_for(train.stages[1], 300) == 1
    assert correction_rounds_for(train.stages[1], 301) == 2
    assert _resume_cursor({"stage": "gen_only", "step": 400}, train.stages) == (1, 0)
    assert _resume_cursor({"stage": "dual", "step": 300}, train.stages) == (1, 300)
    assert loop.variant == "gen_only"
    assert loop.schedule_start == 0.15 and loop.schedule_end == 0.65
    script = (ROOT / "scripts/anchored_train_64.sh").read_text()
    assert "WORLD" in script and "-ne 64" in script
    assert "train-anchored" in script
    assert "OBSERVATION_CHECKPOINT" in script
    assert "anchored_gen_only.json" in script
    assert "BAGEL_ALLOW_LEGACY_MEMORY_SFT" not in script
    assert "train-gen-only" in (ROOT / "scripts/anchored_gen_only_64.sh").read_text()


def test_stage_parameters_keep_the_residual_and_add_slot_attention():
    loop = make_loop(variant="gen_only", max_delta_rms_ratio=1.0)
    first = apply_stage_parameters(loop, AnchoredTrainConfig.from_json(
        ROOT / "configs/anchored_curriculum_train.json",
    ).stages[0])
    names = {name for name, parameter in loop.named_parameters() if parameter.requires_grad}
    assert names
    assert any(name.startswith("gate.") for name in names)
    assert any(name.startswith("adapter.up.") for name in names)
    assert not any(name.startswith(("updater.", "adapter.q_proj.", "adapter.mem_out.")) for name in names)
    groups = {group["name"]: group for group in first}
    assert any(parameter is loop.adapter.up.weight for parameter in groups["output"]["params"])
    assert any(parameter is loop.gate.raw for parameter in groups["base"]["params"])
    assert groups["output"]["base_lr"] == pytest.approx(2e-5)
    assert groups["base"]["base_lr"] == pytest.approx(1e-4)
    second = apply_stage_parameters(loop, AnchoredTrainConfig.from_json(
        ROOT / "configs/anchored_curriculum_train.json",
    ).stages[1])
    live = {name for name, parameter in loop.named_parameters() if parameter.requires_grad}
    for prefix in (
        "updater.down.", "updater.up.", "updater.norm.",
        "adapter.q_proj.", "adapter.k_proj.", "adapter.v_proj.", "adapter.mem_out.",
    ):
        assert any(name.startswith(prefix) for name in live)
    assert not any(name.startswith((
        "gate.", "adapter.up.", "adapter.norm_h.", "adapter.h_down.",
        "adapter.q_norm.", "adapter.k_norm.", "adapter.norm_m.",
    )) for name in live)
    grouped = {group["name"]: group for group in second}

    def holds(group, parameter):
        return any(item is parameter for item in grouped[group]["params"])

    assert holds("output", loop.adapter.mem_out.weight)
    assert not holds("output", loop.adapter.up.weight)
    assert holds("base", loop.adapter.q_proj.weight)
    assert holds("base", loop.updater.up.weight)
    assert not loop.gate.raw.requires_grad
    assert loop.config.variant == "dual"
    apply_stage_parameters(loop, AnchoredTrainConfig.from_json(
        ROOT / "configs/anchored_curriculum_train.json",
    ).stages[0])
    assert not loop.adapter.q_proj.weight.requires_grad
    assert loop.gate.raw.requires_grad
    assert loop.config.variant == "gen_only"


def test_stage2_reader_path_is_required_and_stays_gen_full_query(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="missing"):
        require_stage2_reader(str(missing), "gen_full_query")
    present = tmp_path / "reader.pt"
    present.write_bytes(b"not a checkpoint")
    assert require_stage2_reader(str(present), "gen_full_query") == present
    with pytest.raises(ValueError, match="gen_full_query"):
        require_stage2_reader(str(present), "prompt_and_gen")


def test_curriculum_checkpoints_do_not_resume_the_gen_only_entry(tmp_path):
    path = tmp_path / "gen_only_step_000400.pt"
    torch.save({
        "experiment": "anchored_dual_state_loop",
        "trainer": TRAINER,
        "variant": "gen_only",
        "model": {},
    }, path)
    with pytest.raises(ValueError, match="two-stage"):
        load_gen_only(path, None, None, None)
    other = tmp_path / "plain.pt"
    torch.save({
        "experiment": "anchored_dual_state_loop",
        "variant": "gen_only",
        "model": {},
    }, other)
    with pytest.raises(ValueError, match="anchored curriculum"):
        load_checkpoint(other, None, None)


def test_train_anchored_refuses_a_single_process_launch():
    with pytest.raises(SystemExit, match="anchored_train_64"):
        main([
            "train-anchored", "--model", "unused", "--train", "unused",
            "--output", "unused", "--observation-checkpoint", "unused.pt",
        ])
