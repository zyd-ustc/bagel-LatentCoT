#!/usr/bin/env python3
"""Phase 0.5 anchored dynamic-prompt V-residual probe and T2I runner."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from bagel_common import (
    add_native_model_args,
    autocast_for,
    load_native_bagel,
    make_noise,
    parse_csv_floats,
    relative_l2,
    stable_noise_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--mode", choices=("probe", "generate"), default="probe")
    parser.add_argument(
        "--prompt-file", default="experiments/data/geneval2_hard_16.txt"
    )
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--alphas", default="0,0.1,-0.1,0.2")
    parser.add_argument("--body-start", type=int, default=12)
    parser.add_argument("--body-end", type=int, default=20)
    parser.add_argument("--step-fraction", type=float, default=0.35)
    parser.add_argument("--probe-timestep", type=float, default=0.8)
    return parser.parse_args()


def load_prompts(path: str, limit: int) -> List[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts[: int(limit)] if int(limit) > 0 else prompts


def alpha_slug(alpha: float) -> str:
    text = f"{float(alpha):+.4g}".replace("+", "p").replace("-", "m")
    return "alpha_" + text.replace(".", "p")


def make_bundle(inferencer, prompt: str, image_shape, init_noise):
    empty = inferencer.init_gen_context()
    full = inferencer.update_context_text(prompt, deepcopy(empty))
    image_removed = inferencer.update_context_text(prompt, deepcopy(empty))
    return inferencer.prepare_velocity_bundle(
        name="dynamic_prompt",
        contexts={
            "full": full,
            "text_removed": deepcopy(empty),
            "image_removed": image_removed,
            "has_visual_condition": False,
        },
        image_shape=tuple(image_shape),
        init_noise=init_noise,
        num_loop_tokens=0,
    )


def prompt_noise(model, args, prompt: str) -> torch.Tensor:
    seed = stable_noise_seed(
        int(args.seed), prompt, schema="bagel-dynamic-prompt-phase05-v1"
    )
    return make_noise(model, (int(args.height), int(args.width)), seed)


def probe_prompt(inferencer, model, args, prompt: str, alphas: List[float]):
    image_shape = (int(args.height), int(args.width))
    noise = prompt_noise(model, args, prompt)
    bundle = make_bundle(inferencer, prompt, image_shape, noise)
    x_t = bundle.flow_input["packed_init_noises"].to(inferencer.device)
    velocities: Dict[float, torch.Tensor] = {}
    diagnostics: Dict[str, Any] = {}
    with autocast_for(inferencer.device):
        native_velocity = inferencer.predict_image_velocity(
            x_t=x_t,
            timestep=float(args.probe_timestep),
            condition=bundle,
            cfg_text_scale=float(args.cfg_text_scale),
            cfg_img_scale=float(args.cfg_img_scale),
            cfg_interval=(
                float(args.cfg_interval_min),
                float(args.cfg_interval_max),
            ),
            cfg_renorm_min=float(args.cfg_renorm_min),
            cfg_renorm_type=str(args.cfg_renorm_type),
        ).detach().float().cpu()
        for alpha in alphas:
            velocity, diag = inferencer.predict_dynamic_prompt_velocity(
                x_t=x_t,
                timestep=float(args.probe_timestep),
                condition=bundle,
                alpha=float(alpha),
                body_start=int(args.body_start),
                body_end=int(args.body_end),
                cfg_text_scale=float(args.cfg_text_scale),
                cfg_img_scale=float(args.cfg_img_scale),
                cfg_interval=(
                    float(args.cfg_interval_min),
                    float(args.cfg_interval_max),
                ),
                cfg_renorm_min=float(args.cfg_renorm_min),
                cfg_renorm_type=str(args.cfg_renorm_type),
                return_diagnostics=True,
            )
            velocities[float(alpha)] = velocity.detach().float().cpu()
            diagnostics[str(alpha)] = diag

    if 0.0 not in velocities:
        raise ValueError("--alphas must contain 0 for native parity/reference")
    native = native_velocity
    metrics = {
        str(alpha): {"relative_l2_vs_native": relative_l2(value, native)}
        for alpha, value in velocities.items()
    }
    metrics["0.0"]["max_abs_vs_native"] = float(
        (velocities[0.0] - native).abs().max()
    )
    positive = sorted(alpha for alpha in velocities if alpha > 0)
    directionality = None
    if positive and -positive[0] in velocities:
        plus = (velocities[positive[0]] - native).reshape(-1)
        minus = (velocities[-positive[0]] - native).reshape(-1)
        directionality = float(
            F.cosine_similarity(plus, -minus, dim=0, eps=1e-12)
        )
    scaling = None
    if len(positive) >= 2:
        a, b = positive[:2]
        denom = (velocities[a] - native).norm().clamp_min(1e-12)
        scaling = {
            "alpha_ratio": b / a,
            "delta_norm_ratio": float((velocities[b] - native).norm() / denom),
        }
    return {
        "prompt": prompt,
        "timestep": float(args.probe_timestep),
        "body": [int(args.body_start), int(args.body_end)],
        "metrics": metrics,
        "cos_delta_plus_vs_neg_delta_minus": directionality,
        "scaling": scaling,
        "diagnostics": diagnostics,
    }


def generate_prompt(inferencer, model, args, prompt: str, alphas: List[float], out):
    image_shape = (int(args.height), int(args.width))
    noise = prompt_noise(model, args, prompt)
    rows = []
    for alpha in alphas:
        result = inferencer(
            image=None,
            text=prompt,
            image_shapes=image_shape,
            init_noise=noise.clone(),
            cfg_text_scale=float(args.cfg_text_scale),
            cfg_img_scale=float(args.cfg_img_scale),
            cfg_interval=(
                float(args.cfg_interval_min),
                float(args.cfg_interval_max),
            ),
            timestep_shift=float(args.timestep_shift),
            num_timesteps=int(args.num_steps),
            cfg_renorm_min=float(args.cfg_renorm_min),
            cfg_renorm_type=str(args.cfg_renorm_type),
            dynamic_prompt_alpha=float(alpha),
            dynamic_prompt_body_start=int(args.body_start),
            dynamic_prompt_body_end=int(args.body_end),
            dynamic_prompt_step_fraction=float(args.step_fraction),
            dynamic_prompt_delta_mode="dynamic",
        )
        path = out / f"{alpha_slug(alpha)}.png"
        result["image"].save(path)
        rows.append(
            {
                "alpha": float(alpha),
                "image": path.name,
                "diagnostics": model.last_dynamic_prompt_diagnostics,
            }
        )
    return {"prompt": prompt, "arms": rows}


def main() -> None:
    args = parse_args()
    alphas = parse_csv_floats(args.alphas)
    if not alphas:
        raise ValueError("--alphas cannot be empty")
    prompts = load_prompts(args.prompt_file, args.max_prompts)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, inferencer = load_native_bagel(args)
    model = inferencer.model
    rows = []
    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}] {prompt}", flush=True)
        if args.mode == "probe":
            rows.append(probe_prompt(inferencer, model, args, prompt, alphas))
        else:
            prompt_dir = output_dir / f"p{index:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            rows.append(
                generate_prompt(
                    inferencer, model, args, prompt, alphas, prompt_dir
                )
            )
    manifest = {
        "schema": "bagel_dynamic_prompt_phase05_v1",
        "mode": args.mode,
        "alphas": alphas,
        "body": [int(args.body_start), int(args.body_end)],
        "step_fraction": float(args.step_fraction),
        "rows": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
