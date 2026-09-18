from __future__ import annotations

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from qwen_latent_cot.bagel.checkpoint_io import load_finetuned_weights
from qwen_latent_cot.bagel.checkpoint_loading import project_complete_state_dict


def test_project_complete_state_ignores_disabled_checkpoint_branches() -> None:
    module = nn.Linear(3, 2)
    source = {
        **module.state_dict(),
        "generation_expert.weight": torch.randn(2, 3),
    }

    selected, ignored = project_complete_state_dict(
        module,
        source,
        source_name="checkpoint",
    )

    assert set(selected) == set(module.state_dict())
    assert ignored == 1
    module.load_state_dict(selected, strict=True)


def test_project_complete_state_rejects_missing_understanding_tensor() -> None:
    module = nn.Linear(3, 2)
    source = {"weight": module.weight.detach().clone()}

    with pytest.raises(RuntimeError, match="missing=.*bias"):
        project_complete_state_dict(
            module,
            source,
            source_name="checkpoint",
        )


def test_strict_finetuned_loader_rejects_missing_active_tensor(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = nn.Linear(3, 2)
    save_file({"weight": model.weight.detach().clone()}, checkpoint / "model.safetensors")

    with pytest.raises(RuntimeError, match="incomplete.*count=1"):
        load_finetuned_weights(model, checkpoint, require_complete=True)


def test_finetuned_loader_preserves_wrapper_and_bagel_namespaces(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = nn.Module()
    model.bagel = nn.Linear(3, 2, bias=False)
    model.dino_to_hidden = nn.Linear(4, 3, bias=False)
    bagel_weight = torch.full_like(model.bagel.weight, 2.0)
    adapter_weight = torch.full_like(model.dino_to_hidden.weight, 3.0)
    save_file(
        {
            "bagel.weight": bagel_weight,
            "dino_to_hidden.weight": adapter_weight,
        },
        checkpoint / "model.safetensors",
    )

    load_finetuned_weights(model, checkpoint)

    assert torch.equal(model.bagel.weight, bagel_weight)
    assert torch.equal(model.dino_to_hidden.weight, adapter_weight)


def test_finetuned_loader_selects_joint_component_prefix(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = nn.Linear(3, 2, bias=False)
    producer_weight = torch.full_like(model.weight, 2.0)
    consumer_weight = torch.full_like(model.weight, 7.0)
    save_file(
        {
            "producer.weight": producer_weight,
            "consumer.weight": consumer_weight,
        },
        checkpoint / "model.safetensors",
    )

    load_finetuned_weights(model, checkpoint, source_prefix="producer")

    assert torch.equal(model.weight, producer_weight)


class _ToyConditionLift(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 3, bias=False)
        self.register_buffer("output_scale", torch.tensor(2.5))


def test_finetuned_loader_migrates_legacy_condition_lift_and_scale(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = nn.Module()
    model.dino_to_hidden = _ToyConditionLift()
    legacy_weight = torch.full_like(model.dino_to_hidden.net.weight, 7.0)
    save_file(
        {"condition_lift.net.weight": legacy_weight},
        checkpoint / "model.safetensors",
    )

    load_finetuned_weights(
        model,
        checkpoint,
        required_keys=("dino_to_hidden.net.weight",),
    )

    assert torch.equal(model.dino_to_hidden.net.weight, legacy_weight)
    assert model.dino_to_hidden.output_scale.item() == 1.0


def test_finetuned_loader_rejects_missing_required_interface(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = nn.Module()
    model.dino_to_hidden = _ToyConditionLift()
    save_file(
        {"unrelated": torch.ones(1)},
        checkpoint / "model.safetensors",
    )

    with pytest.raises(RuntimeError, match="missing required interface tensors"):
        load_finetuned_weights(
            model,
            checkpoint,
            required_keys=("dino_to_hidden.net.weight",),
        )
