"""CPU process fixture: exercise real Stage-1 CLI or packed-token DDP math."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import distributed as dist, nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from qwen_latent_cot.bagel.anchored_loop import LoopConfig, LoopResult
from qwen_latent_cot.bagel.stage1_distributed import (
    initialize_stage1_distributed,
    stage1_objective,
    token_weighted_backward_loss,
)
from test_t2i_loop_entrypoints import install_tiny_backbone, script_module
from qwen_latent_cot.bagel.backbone import BagelBackbone


def digest(values):
    value = hashlib.sha256()
    for name, tensor in sorted(values):
        value.update(name.encode())
        value.update(tensor.detach().float().cpu().numpy().tobytes())
    return value.hexdigest()


def real_cli(config_path):
    monkeypatch = pytest.MonkeyPatch()
    install_tiny_backbone(monkeypatch)
    load = BagelBackbone.load
    tracked = {}

    def capture(backbone):
        load(backbone)
        tracked["model"] = backbone.bagel
        tracked["native_before"] = digest(
            (name, value) for name, value in backbone.bagel.named_parameters()
            if not name.startswith("t2i_loop.")
        )
        return backbone

    monkeypatch.setattr(BagelBackbone, "load", capture)
    module = script_module(ROOT / "scripts/train/train_t2i_loop.py")
    sys.argv = ["train_t2i_loop.py", "--config", config_path]
    module.main()
    import os
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    rank = int(os.environ["RANK"])
    model = tracked["model"]
    state = {
        "loop_sha256": digest(model.t2i_loop.named_parameters()),
        "native_before": tracked["native_before"],
        "native_after": digest(
            (name, value) for name, value in model.named_parameters()
            if not name.startswith("t2i_loop.")
        ),
    }
    (Path(cfg["output_dir"]) / f"rank_{rank}_state.json").write_text(json.dumps(state))


class ToyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.t2i_loop = nn.Linear(1, 1, bias=False)
        self.t2i_loop.weight.data.fill_(0.5)
        self.frozen_native = nn.Parameter(torch.ones(3), requires_grad=False)

    def forward_t2i_loop(self, x_t, **kwargs):
        value = self.t2i_loop(x_t)
        return LoopResult(value, torch.zeros_like(value), [value], [])


def token_mean(output):
    device, rank, world_size = initialize_stage1_distributed("cpu")
    model = ToyNative()
    objective = stage1_objective(model, device, world_size)
    assert all(name.startswith("loop.") for name, _ in objective.module.named_parameters())
    tokens = 1 if rank == 0 else 3
    config = LoopConfig()
    loss, _ = objective(
        target=torch.full((tokens, 1), float(rank)),
        x_t=torch.ones(tokens, 1), loop_config=config,
    )
    token_weighted_backward_loss(loss, tokens, device, world_size).backward()
    gradient = model.t2i_loop.weight.grad.clone()
    assert torch.allclose(gradient, torch.tensor([[-0.5]]))
    weights = [torch.empty_like(gradient) for _ in range(world_size)]
    dist.all_gather(weights, gradient)
    assert all(torch.equal(value, gradient) for value in weights)
    if rank == 0:
        Path(output).write_text(json.dumps({"gradient": gradient.item(), "ranks": world_size}))
    dist.destroy_process_group()


if __name__ == "__main__":
    {"cli": real_cli, "token_mean": token_mean}[sys.argv[1]](sys.argv[2])
