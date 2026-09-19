"""Shared zero-shot helpers for frozen BAGEL evaluation scripts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image, ImageChops, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_latent_cot.bagel import accelerator


def parse_csv_floats(value: str) -> List[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def shifted_schedule(num_steps: int, shift: float):
    if int(num_steps) < 2:
        raise ValueError("num_steps must be at least 2")
    if float(shift) <= 0.0:
        raise ValueError("timestep_shift must be positive")
    raw = torch.linspace(1.0, 0.0, int(num_steps), dtype=torch.float64)
    values = float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)
    return values[:-1].tolist(), (values[:-1] - values[1:]).tolist()


def stable_noise_seed(
    seed: int, prompt: str, schema: str = "bagel-zero-shot-v1"
) -> int:
    payload = f"{schema}:{int(seed)}:{prompt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def make_noise(model, image_shape: Tuple[int, int], seed: int) -> torch.Tensor:
    height, width = image_shape
    rows = (height // int(model.latent_downsample)) * (
        width // int(model.latent_downsample)
    )
    width_per_token = int(model.latent_channel) * int(model.latent_patch_size) ** 2
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        (rows, width_per_token), generator=generator, dtype=torch.float32
    )


def autocast_for(device: torch.device):
    return accelerator.autocast_for(device)


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    delta = candidate.detach().float().reshape(-1) - reference.detach().float().reshape(
        -1
    )
    ref_norm = torch.linalg.vector_norm(reference.detach().float().reshape(-1))
    return float(torch.linalg.vector_norm(delta) / torch.clamp(ref_norm, min=1e-12))


def pixel_mae(candidate: Image.Image, reference: Image.Image) -> float:
    difference = ImageChops.difference(
        candidate.convert("RGB"), reference.convert("RGB")
    )
    stats = ImageStat.Stat(difference)
    return float(sum(stats.mean) / len(stats.mean))


def add_native_model_args(parser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda:0 / npu:0 / auto (picks CUDA, else Ascend NPU).",
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval-min", type=float, default=0.4)
    parser.add_argument("--cfg-interval-max", type=float, default=1.0)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="global")
    parser.add_argument("--vit-max-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--vit-image-stride", type=int, default=14)


def load_native_bagel(args):
    from qwen_latent_cot.bagel.backbone import BagelBackbone
    from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
    from qwen_latent_cot.bagel.modeling._bagel_utils import ImageTransform

    patches = int(args.vit_max_image_size) // int(args.vit_image_stride)
    backbone = BagelBackbone(
        cfg={
            "model_path": str(args.model_path),
            "freeze_vit": True,
            "freeze_llm": True,
            "train_gen_expert": False,
            "train_gen_output": False,
            "disable_gen_expert": False,
            "disable_visual_gen": False,
            "num_image_tokens": patches * patches,
            "num_loop_tokens": int(getattr(args, "num_loop_tokens", 8) or 0),
            "loop_depth": int(getattr(args, "loop_depth", 2) or 1),
            "loop_recycle_mode": str(
                getattr(args, "loop_recycle_mode", "same_depth")
            ),
            "loop_memory_persist": bool(
                getattr(args, "loop_memory_persist", False)
            ),
            "memory_loop_start_layer": int(
                getattr(args, "memory_loop_start_layer", 16)
            ),
            "memory_loop_end_layer": int(
                getattr(args, "memory_loop_end_layer", 24)
            ),
            "round0_gen_reads_memory": bool(
                getattr(args, "round0_gen_reads_memory", False)
            ),
        }
    ).load()
    device = accelerator.resolve_device(args.device)
    if not accelerator.is_accelerator(device):
        raise RuntimeError("frozen BAGEL evaluation requires CUDA or Ascend NPU")
    accelerator.manual_seed_all(int(getattr(args, "seed", 0)))
    assert backbone.bagel is not None and backbone.vae_model is not None
    backbone.bagel.to(device).eval()
    backbone.vae_model.to(device).eval()
    for module in (backbone.bagel, backbone.vae_model):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    token_ids = backbone.token_ids
    new_token_ids = {
        "bos_token_id": int(token_ids.im_start),
        "eos_token_id": int(token_ids.im_end),
        "start_of_image": int(token_ids.vision_start),
        "end_of_image": int(token_ids.vision_end),
    }
    inferencer = InterleaveInferencer(
        model=backbone.bagel,
        vae_model=backbone.vae_model,
        tokenizer=backbone.tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(
            int(args.vit_max_image_size),
            int(args.vit_min_image_size),
            int(args.vit_image_stride),
        ),
        new_token_ids=new_token_ids,
    )
    return backbone, inferencer


def predict_velocity(
    inferencer,
    x_t,
    timestep,
    bundle,
    args,
    *,
    cfg=None,
    within_step_loop=None,
):
    cfg_kwargs = {
        "cfg_text_scale": float(args.cfg_text_scale),
        "cfg_img_scale": float(args.cfg_img_scale),
        "cfg_interval": (float(args.cfg_interval_min), float(args.cfg_interval_max)),
        "cfg_renorm_min": float(args.cfg_renorm_min),
        "cfg_renorm_type": str(args.cfg_renorm_type),
    }
    for key, value in (cfg or {}).items():
        if value is not None:
            cfg_kwargs[key] = value
    for key, value in (within_step_loop or {}).items():
        if value is not None:
            cfg_kwargs[key] = value
    return inferencer.predict_image_velocity(
        x_t=x_t,
        timestep=float(timestep),
        condition=bundle,
        **cfg_kwargs,
    )


def run_full_trajectory(
    inferencer, model, bundle, init_noise, timesteps, dts, args, *, cfg=None,
    within_step_loop=None,
):
    """Native full Euler t: 1 -> 0 from a fixed initial noise."""

    x_t = init_noise.clone()
    with autocast_for(accelerator.resolve_device(args.device)):
        for index, timestep in enumerate(timesteps):
            velocity = predict_velocity(
                inferencer, x_t, timestep, bundle, args, cfg=cfg,
                within_step_loop=within_step_loop,
            )
            x_t = model.image_euler_step(x_t, velocity, float(dts[index]))
    return x_t


def run_truncated_draft(
    inferencer, model, bundle, init_noise, timesteps, dts, args, t_stop, *, cfg=None
):
    """Run Euler until t <= t_stop, then return (x0_hat, actual_t)."""

    x_t = init_noise.clone()
    with autocast_for(accelerator.resolve_device(args.device)):
        for index, timestep in enumerate(timesteps):
            if float(timestep) <= float(t_stop):
                velocity = predict_velocity(
                    inferencer, x_t, timestep, bundle, args, cfg=cfg
                )
                x0_hat = x_t - float(timestep) * velocity
                return x0_hat, float(timestep)
            velocity = predict_velocity(
                inferencer, x_t, timestep, bundle, args, cfg=cfg
            )
            x_t = model.image_euler_step(x_t, velocity, float(dts[index]))
    raise ValueError("t_stop was below the final scheduled timestep")
