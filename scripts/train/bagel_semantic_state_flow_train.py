#!/usr/bin/env python3
"""Warm up BAGEL semantic recurrent state with cross-step local flow loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_latent_cot.bagel.backbone import BagelBackbone
from qwen_latent_cot.bagel.loop import (
    BagelCrossStepFlowModule,
    loop_adapter_state_dict,
    loop_trainable_names,
)
from qwen_latent_cot.bagel.loop_data import (
    LoopFlowCollator,
    LoopFlowCollatorConfig,
    LoopFlowDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/semantic_state_flow.yaml"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--state-mode", choices=("semantic_token", "kv_prefix"))
    parser.add_argument("--state-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = dict(yaml.safe_load(handle) or {})
    if args.max_steps is not None:
        config["max_steps"] = int(args.max_steps)
    if args.state_mode is not None:
        config["loop_state_mode"] = args.state_mode
    if args.state_tokens is not None:
        config["loop_state_tokens"] = int(args.state_tokens)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    required = ("model_path", "data_paths", "output_dir")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"missing configuration values: {missing}")
    if config.get("loop_state_mode", "semantic_token") not in {
        "semantic_token",
        "kv_prefix",
    }:
        raise ValueError("loop_state_mode must be semantic_token or kv_prefix")
    if int(config.get("loop_state_tokens", 16)) < 1:
        raise ValueError("loop_state_tokens must be at least 1")
    return config


def save_adapter(model, output_dir: Path, step: int, config: dict) -> None:
    stem = f"semantic_state_flow_step_{int(step):07d}"
    save_file(loop_adapter_state_dict(model), str(output_dir / f"{stem}.safetensors"))
    metadata = {
        "schema": "bagel_semantic_state_flow_adapter_v7",
        "objective": "cross_step_local_flow",
        "step": int(step),
        "loop_start_layer": int(config["loop_start_layer"]),
        "loop_end_layer": int(config["loop_end_layer"]),
        "loop_state_scale": float(config["loop_state_scale"]),
        "loop_draft_state_scale": float(config.get("loop_draft_state_scale", 0.2)),
        "loop_state_mode": str(config["loop_state_mode"]),
        "loop_state_tokens": int(config["loop_state_tokens"]),
        "rollout_steps": int(config.get("rollout_steps", 4)),
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_alpha": int(config.get("lora_alpha", 16)),
        "include_text_kv_lora": bool(config.get("include_text_kv_lora", True)),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args)
    device = torch.device(str(config.get("device", "cuda:0")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("semantic-state flow warm-up requires CUDA")
    torch.manual_seed(int(config.get("seed", 42)))
    torch.cuda.manual_seed_all(int(config.get("seed", 42)))

    backbone = BagelBackbone(
        {
            "model_path": str(config["model_path"]),
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "num_image_tokens": int(config.get("num_image_tokens", 1024)),
        }
    ).load()
    model, vae = backbone.bagel, backbone.vae_model
    assert model is not None and vae is not None and backbone.token_ids is not None
    start, end = int(config["loop_start_layer"]), int(config["loop_end_layer"])
    backbone.apply_loop_trainable_policy(
        start_layer=start,
        end_layer=end,
        rank=int(config.get("lora_rank", 8)),
        alpha=int(config.get("lora_alpha", 16)),
        dropout=float(config.get("lora_dropout", 0.0)),
        include_text_kv=bool(config.get("include_text_kv_lora", True)),
    )
    model.to(device)
    vae.to(device).eval()
    model.language_model.model.gradient_checkpointing_enable()

    dataset = LoopFlowDataset(
        [str(path) for path in config["data_paths"]],
        include_open_trajectories=bool(config.get("include_open_trajectories", False)),
        sample_size=config.get("sample_size"),
        seed=int(config.get("seed", 42)),
    )
    collator = LoopFlowCollator(
        LoopFlowCollatorConfig(
            vae_image_size=int(config.get("image_size", 512)),
            vae_min_image_size=int(config.get("image_size", 512)),
            latent_downsample=int(model.latent_downsample),
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    module = BagelCrossStepFlowModule(
        model,
        tokenizer=backbone.tokenizer,
        token_ids=backbone.token_ids,
        vae_model=vae,
        loop_start_layer=start,
        loop_end_layer=end,
        loop_state_scale=float(config.get("loop_state_scale", 0.2)),
        loop_state_mode=str(config.get("loop_state_mode", "semantic_token")),
        loop_state_tokens=int(config.get("loop_state_tokens", 16)),
        loop_draft_state_scale=float(config.get("loop_draft_state_scale", 0.2)),
        rollout_steps=int(config.get("rollout_steps", 4)),
        timestep_shift=float(config.get("timestep_shift", 3.0)),
        loop_state_timestep_threshold=config.get("loop_state_timestep_threshold"),
        loop_residual_alpha=float(config.get("loop_residual_alpha", 1.0)),
    ).train()
    vae.eval()
    names = loop_trainable_names(
        model,
        start_layer=start,
        end_layer=end,
        include_text_kv=bool(config.get("include_text_kv_lora", True)),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 1e-5)),
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps({**config, "trainable_names": names}, indent=2) + "\n",
        encoding="utf-8",
    )

    iterator = iter(loader)
    max_steps = int(config.get("max_steps", 100))
    save_steps = int(config.get("save_steps", 50))
    log_path = output_dir / "metrics.jsonl"
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = module(batch)
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(config.get("max_grad_norm", 1.0))
        )
        optimizer.step()
        row = {
            "step": step,
            "loss": float(result.loss.detach()),
            "state_rms": float(result.state_rms),
            "grad_norm": float(grad_norm),
            "per_step_flow_losses": [
                float(value.detach()) for value in result.per_step_flow_losses
            ],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if step % save_steps == 0 or step == max_steps:
            save_adapter(model, output_dir, step, config)


if __name__ == "__main__":
    main()
