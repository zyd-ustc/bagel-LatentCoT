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
    sample_timesteps,
    schedule_time_window,
    scheduled_flow_times,
    shift_unit_into_window,
    gen_only_parameter_groups,
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
    t_min, t_max = schedule_time_window(config)
    assert t_min == pytest.approx(0.35) and t_max == pytest.approx(0.85)
    ends = shift_unit_into_window(torch.tensor([0.0, 1.0]), 3.0, t_min, t_max)
    assert ends[0].item() == pytest.approx(t_min)
    assert ends[1].item() == pytest.approx(t_max)
    generator = torch.Generator().manual_seed(17)
    drawn = sample_timesteps(
        2048, timestep_shift=3.0, device=torch.device("cpu"), generator=generator,
        t_min=t_min, t_max=t_max,
    )
    assert drawn.shape == (2048,)
    assert float(drawn.min()) >= t_min - 1e-6
    assert float(drawn.max()) <= t_max + 1e-6
    assert all(in_loop_schedule(config, flow_time=float(value)) for value in drawn[::128])


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


def test_gen_only_gate_keeps_its_own_learning_rate():
    loop = make_loop(variant="gen_only", max_delta_rms_ratio=1.0)
    train = GenOnlyTrainConfig.from_json(ROOT / "configs/anchored_gen_only_train.json")
    groups = {
        group["name"]: group
        for group in gen_only_parameter_groups(loop, train.learning_rate, train.weight_decay)
    }
    assert any(item is loop.gate.raw for item in groups["gate"]["params"])
    assert not any(item is loop.gate.raw for item in groups["base"]["params"])
    assert any(item is loop.adapter.up.weight for item in groups["base"]["params"])
    assert groups["base"]["base_lr"] == pytest.approx(1e-5)
    assert groups["gate"]["base_lr"] == pytest.approx(5e-4)
    assert groups["gate"]["weight_decay"] == 0.0
    optimizer = torch.optim.AdamW(
        [groups["base"], groups["gate"]], weight_decay=train.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * (1.0 / 20.0)
    warmed = {group["name"]: group for group in optimizer.param_groups}
    assert warmed["gate"]["lr"] == pytest.approx(5e-4 / 20.0)
    assert warmed["base"]["lr"] == pytest.approx(1e-5 / 20.0)
    assert warmed["gate"]["weight_decay"] == 0.0
    assert float(loop.gate.raw.detach()) == 0.0


def test_train_gen_only_refuses_a_single_process_launch():
    with pytest.raises(SystemExit, match="anchored_gen_only_64"):
        main([
            "train-gen-only", "--model", "unused", "--train", "unused",
            "--output", "unused",
        ])
