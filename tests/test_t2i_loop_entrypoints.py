"""Run the real training/generation entry points with a tiny native MoT/VAE."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from test_anchored_t2i_loop import tiny_model
from torch import nn

from qwen_latent_cot.bagel.anchored_loop import LoopConfig, LoopModules
from qwen_latent_cot.bagel.backbone import BagelBackbone
from qwen_latent_cot.bagel.inferencer import InterleaveInferencer


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return [3, 4, 5 + len(text) % 20]


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode(self, pixels):
        return F.avg_pool2d(pixels[:, :2], 2)

    def decode(self, latent):
        value = latent.mean(1, keepdim=True).expand(-1, 3, -1, -1)
        return F.interpolate(value, scale_factor=2).tanh()


def install_tiny_backbone(monkeypatch):
    def load(backbone):
        model, _ = tiny_model()
        config = LoopConfig(**backbone.cfg["t2i_loop"])
        model.t2i_loop = LoopModules(32, config).to(torch.bfloat16)
        backbone.bagel = backbone.raw_model = model
        backbone.vae_model = TinyVAE()
        backbone.tokenizer = TinyTokenizer()
        backbone.token_ids = {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "start_of_image": 2,
            "end_of_image": 3,
        }
        return backbone

    monkeypatch.setattr(BagelBackbone, "load", load)


def script_module(path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_alpha_has_identical_final_pixels_and_resets_workspace_each_step():
    model, config = tiny_model(2)
    inferencer = InterleaveInferencer(
        model,
        TinyVAE(),
        TinyTokenizer(),
        {"bos_token_id": 0, "eos_token_id": 1, "start_of_image": 2, "end_of_image": 3},
    )
    images = inferencer.generate(
        ["two cubes"],
        image_shape=(16, 16),
        seed=42,
        num_timesteps=4,
        cfg_text_scale=4,
        cfg_img_scale=1,
    )
    logs = list(model.last_loop_diagnostics)
    model.t2i_loop.config = replace(config, enable_t2i_loop=False)
    base = inferencer.generate(
        ["two cubes"],
        image_shape=(16, 16),
        seed=42,
        num_timesteps=4,
        cfg_text_scale=4,
        cfg_img_scale=1,
    )
    assert images[0].tobytes() == base[0].tobytes()
    assert {item["step"] for item in logs} == {0, 1, 2}
    assert all(item["velocity_delta_ratio"] == 0 for item in logs)


@pytest.mark.parametrize("mode", ["gen_memory_anchored", "gen_only"])
def test_stage1_cli_executes_optimizer_and_saves_loop_only_checkpoint(
    tmp_path, monkeypatch, mode
):
    install_tiny_backbone(monkeypatch)
    Image.new("RGB", (16, 16), "red").save(tmp_path / "image.png")
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps(
            {"prompt": "two red cubes", "image": "image.png", "bucket": "structural"}
        )
        + "\n"
    )
    config = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=1,
        loop_end_layer=3,
        runtime_loop_depth=2,
        loop_mode=mode,
        memory_slots=0 if mode == "gen_only" else 2,
        loop_output_alpha_init=0.01 if mode == "gen_only" else 0.0,
    )
    cfg = {
        "model_path": str(tmp_path / "base"),
        "data_path": str(data),
        "output_dir": str(tmp_path / "train"),
        "device": "cpu",
        "image_size": 16,
        "steps": 2,
        "save_every": 1,
        "learning_rate": 0.01,
        "depth_curriculum": [1, 2],
        "loop": config.to_dict(),
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    module = script_module(
        Path(__file__).resolve().parents[1] / "scripts/train/train_t2i_loop.py"
    )
    monkeypatch.setattr("sys.argv", ["train_t2i_loop.py", "--config", str(path)])
    module.main()
    records = [
        json.loads(line)
        for line in (tmp_path / "train/metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert any(value != 0 for value in records[-1]["alpha"])
    checkpoint = tmp_path / "train/step_000002"
    assert (checkpoint / "loop.safetensors").is_file()
    names = json.loads((tmp_path / "train/trainable.json").read_text())
    assert all(name.startswith("t2i_loop.") for name in names)
    assert all(record["grad_norm"] > 0 for record in records)
    if mode == "gen_only":
        from safetensors.torch import load_file

        assert (
            load_file(str(checkpoint / "loop.safetensors"))["reentry.up.bias"]
            .abs()
            .sum()
            > 0
        )


def test_topology_cli_exports_matched_arms_and_all_timestep_bins(tmp_path, monkeypatch):
    install_tiny_backbone(monkeypatch)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps({"prompt": prompt}) for prompt in ["two cubes", "a sphere"]
        )
    )
    module = script_module(
        Path(__file__).resolve().parents[1] / "scripts/evaluate/t2i_loop_matrix.py"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "t2i_loop_matrix.py",
            "--model-path",
            str(tmp_path / "base"),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(tmp_path / "eval"),
            "--device",
            "cpu",
            "--image-size",
            "16",
            "--num-timesteps",
            "4",
            "--start-layer",
            "1",
            "--end-layer",
            "3",
            "--memory-slots",
            "2",
            "--depths",
            "0,1,2",
            "--batch-size",
            "2",
            "--save-readouts",
        ],
    )
    module.main()
    manifest = json.loads((tmp_path / "eval/manifest.json").read_text())
    assert len(manifest["images"]) == 22  # base + 5 modes * 2 depths, two samples
    assert all(Path(row["path"]).is_file() for row in manifest["images"])
    arm = tmp_path / "eval/gen_memory_anchored_R2_K2_correct"
    logs = json.loads((arm / "batch_0000_loop_logs.json").read_text())
    assert {row["bin"] for row in logs} == {"early", "middle", "late"}
    assert len(list(arm.glob("*_velocities.pt"))) == 3
    assert len(list(arm.glob("*_r2.png"))) == 6
