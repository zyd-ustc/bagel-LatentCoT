"""GEN-only training entry. The 64-card launcher is not started here."""
from pathlib import Path

import pytest
import torch

from qwen_latent_cot.anchored.cli import main
from qwen_latent_cot.anchored.config import LoopConfig
from qwen_latent_cot.anchored.gen_only_train import (
    DEVICES,
    GenOnlyTrainConfig,
    correction_loss,
    require_gen_only_loop,
    require_sixty_four,
    scheduled_flow_times,
    select_gen_only_parameters,
)
from qwen_latent_cot.anchored.legacy import refuse_legacy_checkpoint
from qwen_latent_cot.anchored.schedule import in_loop_schedule
from qwen_latent_cot.anchored.self_check import make_loop, pack_case, tiny_language_model
from qwen_latent_cot.diagnostics.schedule import native_times


ROOT = Path(__file__).resolve().parents[1]


def test_gen_only_configs_match_the_sixty_four_card_entry():
    loop = LoopConfig.from_json(ROOT / "configs/anchored_gen_only.json")
    train = GenOnlyTrainConfig.from_json(ROOT / "configs/anchored_gen_only_train.json")
    require_gen_only_loop(loop)
    assert loop.correction_rounds == 1
    assert loop.round_loss_weights is None
    assert loop.observation_checkpoint is None
    train.validate()
    require_sixty_four(DEVICES, train.global_batch)
    script = (ROOT / "scripts/anchored_gen_only_64.sh").read_text()
    assert "WORLD" in script and "-ne 64" in script
    assert "train-gen-only" in script
    assert "GEN_ONLY_VAL" not in script and "--val" not in script
    assert "BAGEL_ALLOW_LEGACY_MEMORY_SFT" not in script
    with pytest.raises(ValueError, match="gen_only"):
        require_gen_only_loop(LoopConfig.from_json(ROOT / "configs/anchored_dual_state_loop.json"))
    with pytest.raises(ValueError, match="64"):
        require_sixty_four(16, 64)


def test_scheduled_times_stay_inside_the_loop_window():
    config = LoopConfig.from_json(ROOT / "configs/anchored_gen_only.json")
    selected = scheduled_flow_times(config, 50, 3.0)
    grid = native_times({"sampler_steps": 50, "timestep_shift": 3.0})
    assert config.schedule == "interval"
    assert config.schedule_start == 0.15 and config.schedule_end == 0.65
    assert selected
    assert all(in_loop_schedule(config, flow_time=value) for value in selected)
    assert all(0.35 - 1e-6 <= float(value) <= 0.85 + 1e-6 for value in selected)
    assert any(float(value) > 0.85 for value in grid)
    assert not in_loop_schedule(config, flow_time=0.95)
    assert in_loop_schedule(config, flow_time=0.6)
    assert any(not in_loop_schedule(config, flow_time=float(value)) for value in grid)


def test_gen_only_step_trains_the_gate_and_leaves_memory_out():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=19, flow_time=0.9)
    loop = make_loop(language_model, variant="gen_only", max_delta_rms_ratio=1.0)
    parameters = select_gen_only_parameters(loop)
    names = {name for name, parameter in loop.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("gate.") for name in names)
    assert any(name.startswith("adapter.up.") for name in names)
    assert not any(name.startswith((
        "adapter.mem_out.", "adapter.q_proj.", "adapter.q_norm.", "adapter.k_norm.", "updater.",
    )) for name in names)
    loop.train()
    output = loop.forward_branch(hidden, branch, flow_time=0.9, train=True)
    assert torch.equal(output.delta, torch.zeros_like(output.delta))
    target = torch.randn_like(output.velocity)
    loss, parts = correction_loss(loop, output, target, branch.packed_vae_token_indexes)
    assert "error_fell_1" in parts
    (loss / 64).backward()
    missing = [name for name, parameter in loop.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert not missing
    with torch.no_grad():
        assert float(loop.gate.raw.grad.abs().sum()) > 0
        assert float(loop.adapter.up.weight.grad.abs().sum()) > 0
    assert loop.adapter.mem_out.weight.grad is None
    assert loop.adapter.q_norm.weight.grad is None
    assert loop.adapter.k_norm.weight.grad is None
    assert loop.updater.up.weight.grad is None
    blob = loop.export_state()
    blob["variant"] = "gen_only"
    refuse_legacy_checkpoint(blob)
    assert all(not key.startswith("observer") for key in blob["model"])
    assert parameters


def test_train_gen_only_refuses_a_single_process_launch():
    with pytest.raises(SystemExit, match="anchored_gen_only_64"):
        main([
            "train-gen-only", "--model", "unused", "--train", "unused",
            "--output", "unused",
        ])
