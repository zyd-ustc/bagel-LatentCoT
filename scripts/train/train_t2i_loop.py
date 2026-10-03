#!/usr/bin/env python3
"""Stage 1 only. Later workspace/body stages require measured experiment gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
from dataclasses import replace

import torch
import yaml
from torch.utils.data import DataLoader

from qwen_latent_cot.bagel import BagelBackbone, LoopConfig, direct_flow_loss
from qwen_latent_cot.bagel.accelerator import (
    autocast_for,
    manual_seed_all,
    resolve_device,
)
from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
from qwen_latent_cot.bagel.loop_checkpoint import save_loop_checkpoint
from qwen_latent_cot.data.t2i import T2IDataset, patchify_latents, sample_flow_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    config = LoopConfig(**cfg["loop"])
    if (
        not config.enable_t2i_loop
        or config.loop_depth == 0
        or config.reentry_adapter_type != "low_rank"
    ):
        raise ValueError(
            "Stage 1 requires enabled extra loops with a trainable low_rank adapter"
        )
    if config.loop_mode == "direct_native" or config.memory_control != "correct":
        raise ValueError(
            "Stage 1 trains anchored states with correct workspace; use evaluation for negative controls"
        )
    if config.loop_gate_init == 0:
        raise ValueError(
            "Stage 1 needs positive loop_gate_init so output alpha can learn"
        )
    if int(cfg.get("save_every", 100)) < 1 or int(cfg.get("batch_size", 1)) < 1:
        raise ValueError("save_every and batch_size must be positive")
    depths = cfg.get("depth_curriculum", [1, 2])
    if not depths or min(depths) < 1 or max(depths) > config.loop_depth:
        raise ValueError("depth curriculum must lie within allocated loop_depth")
    seed = int(cfg.get("seed", 0))
    manual_seed_all(seed)
    device = resolve_device(cfg.get("device", "auto"))
    backbone = BagelBackbone(
        {
            "model_path": cfg["model_path"],
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "t2i_loop": config.to_dict(),
        }
    ).load()
    model, vae = backbone.bagel.to(device), backbone.vae_model.to(device)
    # Packed inference retains autograd. Native weights and dropout stay frozen.
    model.eval()
    trainable_names = backbone.apply_stage1_policy()
    model.language_model.model.gradient_checkpointing = bool(
        cfg.get("gradient_checkpointing", True)
    )
    inferencer = InterleaveInferencer(
        model, vae, backbone.tokenizer, backbone.token_ids
    )
    size = int(cfg.get("image_size", 512))
    if size % model.latent_downsample:
        raise ValueError("image_size must be divisible by native latent_downsample")
    dataset = T2IDataset(cfg["data_path"], size)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "trainable.json").write_text(json.dumps(trainable_names, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    step, total = 0, int(cfg.get("steps", 1000))
    if total < 1:
        raise ValueError("steps must be positive")
    with (out / "metrics.jsonl").open("w") as log:
        while step < total:
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), autocast_for(device):
                    clean = patchify_latents(
                        vae.encode(batch["pixels"].to(device)), model.latent_patch_size
                    ).float()
                    condition = inferencer.prepare_condition(
                        batch["prompt"], (size, size)
                    )
                    # Prompt caches are anchors, not data for an autograd graph.
                    inputs = {
                        key: value
                        for key, value in condition.inputs.items()
                        if not key.startswith("cfg_")
                    }
                depth = int(depths[int(torch.randint(len(depths), ()).item())])
                runtime = replace(config, loop_depth=depth)
                t = torch.rand((), device=device)
                noise = torch.randn_like(clean)
                x_t, target = sample_flow_state(clean, t, noise)
                with autocast_for(device):
                    result = model.forward_t2i_loop(
                        x_t=x_t,
                        timestep=t.expand(clean.shape[0]),
                        loop_config=runtime,
                        **inputs,
                    )
                    loss = direct_flow_loss(result, target, runtime)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite flow loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(cfg.get("max_grad_norm", 1.0))
                )
                if not torch.isfinite(norm):
                    raise RuntimeError(f"nonfinite loop gradients at step {step}")
                optimizer.step()
                step += 1
                record = {
                    "step": step,
                    "depth": depth,
                    "flow_loss": float(loss.detach()),
                    "grad_norm": float(norm),
                    "alpha": model.t2i_loop.output_alpha.detach()
                    .float()
                    .cpu()
                    .tolist(),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if step % int(cfg.get("save_every", 100)) == 0 or step == total:
                    save_loop_checkpoint(
                        model,
                        out / f"step_{step:06d}",
                        step=step,
                        model_path=cfg["model_path"],
                    )
                if step >= total:
                    break


if __name__ == "__main__":
    main()
