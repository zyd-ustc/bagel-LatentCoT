#!/usr/bin/env python3
"""Phase 1.1: ground BAGEL latent memory from paired source/target images."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from PIL import Image
from safetensors.torch import save_file

from qwen_latent_cot.bagel import accelerator
from qwen_latent_cot.bagel.loop import (
    UND_Q_PROJECTIONS,
    configure_loop_trainable_routes,
    loop_adapter_route_state_dict,
)
from qwen_latent_cot.bagel.loop_pair_ground import (
    pair_memory_loss,
    prepare_flow_training_state,
    sample_weighted_timestep,
)
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb
from qwen_latent_cot.data.phase1_pairs import (
    STAGE_A_EDIT_TYPES,
    build_phase1_sampling_order,
    load_phase1_pairs,
)


LOGGER = logging.getLogger("bagel.pair_memory.train")
SCHEMA = "bagel_pair_grounded_memory_adapter_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/loop_pair_memory_early.yaml")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = dict(yaml.safe_load(handle) or {})
    for key, value in {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "device": args.device,
        "max_steps": args.max_steps,
    }.items():
        if value not in (None, ""):
            config[key] = value
    missing = [key for key in ("model_path", "data_path", "output_dir") if not config.get(key)]
    if missing:
        raise ValueError(f"missing Phase-1.1 configuration: {missing}")
    locked = {
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "loop_recycle_mode": "same_depth",
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
        "round0_memory_write_enabled": False,
        "gen_attention_o_lora": False,
        "k_v_lora": False,
    }
    mismatches = {key: (config.get(key, expected), expected) for key, expected in locked.items() if config.get(key, expected) != expected}
    if mismatches:
        raise ValueError(f"Phase 1.1 architecture is locked; mismatches={mismatches}")
    return config


def _set_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)


def _scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(round(total_steps * warmup_ratio)))

    def scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _load_image(inferencer, path: str) -> Image.Image:
    with Image.open(path) as handle:
        image = pil_img2rgb(handle.convert("RGB"))
    return inferencer.vae_transform.resize_transform(image)


def build_visual_reference_context(inferencer, image: Image.Image):
    return inferencer.update_context_image(
        image, inferencer.init_gen_context(), vae=True, vit=True
    )


def build_student_edit_context(inferencer, source_context, instruction: str):
    # Reuse the exact source-image KV realization used by M0_ref.  BAGEL's
    # VAE encoder samples its posterior, so re-encoding the same PIL image here
    # would add an unintended source-latent difference to D_S.
    return inferencer.update_context_text(
        str(instruction), deepcopy(source_context)
    )


def _condition(inferencer, name: str, context, image_shape):
    return inferencer.prepare_memory_read_bundle(
        name=name,
        context=context,
        image_shape=image_shape,
        num_loop_tokens=8,
    )


def _read_kwargs(inferencer, condition, x_t, timestep: float) -> dict[str, Any]:
    return inferencer.build_memory_read_kwargs(
        x_t=x_t,
        timestep=timestep,
        condition=condition,
        memory_loop_start=12,
        memory_loop_end=20,
    )


def _save(model, output_dir: Path, step: int, config: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_memory_adapter_step_{step:07d}"
    save_file(
        loop_adapter_route_state_dict(model, UND_Q_PROJECTIONS),
        str(output_dir / f"{stem}.safetensors"),
    )
    metadata = {
        "schema": SCHEMA,
        "objective": "pair_grounded_memory_delta",
        "step": step,
        "K": 8,
        "R": 1,
        "body": [12, 20],
        "adapter_routes": list(UND_Q_PROJECTIONS),
        "m0_trainable": False,
        "weights": {
            key: float(config[key])
            for key in ("lambda_mem_dir", "lambda_mem_mag", "lambda_mem_reg", "lambda_noop_mem")
        },
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _arguments()
    config = _load_config(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    allowed = STAGE_A_EDIT_TYPES if bool(config.get("stage_a_only", True)) else None
    records = load_phase1_pairs(str(config["data_path"]), allowed_edit_types=allowed)
    if args.validate_only:
        LOGGER.info("validated %d pair-grounded records", len(records))
        return

    model_path = Path(str(config["model_path"])).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    device = accelerator.resolve_device(config.get("device", "auto"))
    if not accelerator.is_accelerator(device):
        raise RuntimeError("Phase-1.1 requires CUDA or Ascend NPU")
    _set_device(device)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    accelerator.manual_seed_all(seed)
    training_records = build_phase1_sampling_order(
        records,
        noop_fraction=float(config.get("noop_fraction", 0.20)),
        seed=seed,
    )

    from qwen_latent_cot.bagel.backbone import BagelBackbone
    from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
    from qwen_latent_cot.bagel.modeling._bagel_utils import ImageTransform

    backbone = BagelBackbone({
        "model_path": str(model_path),
        "disable_visual_gen": False,
        "disable_gen_expert": False,
        "num_image_tokens": int(config.get("num_image_tokens", 4900)),
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "loop_recycle_mode": "same_depth",
        "loop_memory_persist": False,
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
        "round0_memory_write_enabled": False,
    }).load()
    model, vae = backbone.bagel, backbone.vae_model
    assert model is not None and vae is not None
    backbone.apply_loop_trainable_policy(
        start_layer=12,
        end_layer=20,
        rank=int(config.get("lora_rank", 8)),
        alpha=int(config.get("lora_alpha", 16)),
        dropout=float(config.get("lora_dropout", 0.0)),
        gen_attention_o_lora=False,
        k_v_lora=False,
    )
    trainable_names = configure_loop_trainable_routes(model, UND_Q_PROJECTIONS)
    model.to(device).eval()
    vae.to(device).eval()
    if model.loop_memory is None or model.loop_memory.requires_grad:
        raise RuntimeError("Phase-1.1 requires frozen boundary-initialized m0")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ids = backbone.token_ids
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae,
        tokenizer=backbone.tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(
            int(config.get("vit_max_image_size", 980)),
            int(config.get("vit_min_image_size", 224)),
            int(config.get("vit_image_stride", 14)),
        ),
        new_token_ids={
            "bos_token_id": int(ids.im_start),
            "eos_token_id": int(ids.im_end),
            "start_of_image": int(ids.vision_start),
            "end_of_image": int(ids.vision_end),
        },
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 5.0e-6)),
        betas=tuple(float(value) for value in config.get("betas", (0.9, 0.95))),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    max_steps = int(config.get("max_steps", 1))
    scheduler = _scheduler(optimizer, max_steps, float(config.get("warmup_ratio", 0.03)))
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps({**config, "schema": SCHEMA, "records": len(records), "sampling_epoch_records": len(training_records), "trainable_tensors": trainable_names}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"

    for step in range(1, max_steps + 1):
        record = training_records[(step - 1) % len(training_records)]
        source = _load_image(inferencer, record["source_image"])
        target = source.copy() if record["target_image"] == record["source_image"] else _load_image(inferencer, record["target_image"])
        image_shape = tuple(target.size[::-1])
        with torch.no_grad():
            clean_latent = inferencer.encode_image(target, image_shape)
        generator = torch.Generator(device="cpu").manual_seed(seed + step * 1_000_003)
        noise = torch.randn(clean_latent.shape, generator=generator, dtype=torch.float32).to(device=device, dtype=clean_latent.dtype)
        timestep = sample_weighted_timestep(
            generator,
            bucket_weights=tuple(
                float(value)
                for value in config.get("timestep_bucket_weights", (0.5, 0.3, 0.2))
            ),
        )
        x_t, _ = prepare_flow_training_state(clean_latent, timestep, noise)

        source_context = build_visual_reference_context(inferencer, source)
        source_condition = _condition(inferencer, "source_ref", source_context, image_shape)
        target_condition = (
            source_condition
            if bool(record["is_noop"])
            else _condition(
                inferencer,
                "target_ref",
                build_visual_reference_context(inferencer, target),
                image_shape,
            )
        )
        student_condition = _condition(inferencer, "student", build_student_edit_context(inferencer, source_context, record["instruction"]), image_shape)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast_for(device):
            with torch.no_grad():
                source_ref = model.forward_memory_read(**_read_kwargs(inferencer, source_condition, x_t, timestep), adapter_mode="off").memory_read
                target_ref = model.forward_memory_read(**_read_kwargs(inferencer, target_condition, x_t, timestep), adapter_mode="off").memory_read
            student = model.forward_memory_read(**_read_kwargs(inferencer, student_condition, x_t, timestep), adapter_mode="read").memory_read
            result = pair_memory_loss(
                student_memory=student,
                source_reference=source_ref,
                target_reference=target_ref,
                is_noop=bool(record["is_noop"]),
                lambda_mem_dir=float(config.get("lambda_mem_dir", 1.0)),
                lambda_mem_mag=float(config.get("lambda_mem_mag", 0.1)),
                lambda_mem_reg=float(config.get("lambda_mem_reg", 0.25)),
                lambda_noop_mem=float(config.get("lambda_noop_mem", 1.0)),
                normalization_floor=float(config.get("normalization_floor", 1.0e-4)),
            )
        result.loss.backward()
        missing_grad = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        if missing_grad:
            raise RuntimeError(f"UND-Q tensors without gradient: {missing_grad[:8]}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config.get("max_grad_norm", 1.0)))
        optimizer.step()
        scheduler.step()
        stats = model.memory_slot_stats(student)
        is_noop = bool(record["is_noop"])
        row = {
            "step": step,
            "sample_id": str(record["id"]),
            "is_noop": is_noop,
            "timestep": timestep,
            "loss": float(result.loss.detach()),
            # The target delta is exactly zero for no-op; cosine and relative
            # error are undefined there.  Use no-op RMS for that group.
            "cosine": None if is_noop else float(result.cosine.detach()),
            "relative_error": None if is_noop else float(result.relative_error.detach()),
            "direction_loss": float(result.direction_loss.detach()),
            "magnitude_loss": float(result.magnitude_loss.detach()),
            "regression_loss": float(result.regression_loss.detach()),
            "noop_loss": float(result.noop_loss.detach()),
            "student_rms": float(result.student_rms.detach()),
            "target_rms": float(result.target_rms.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().float()),
            "effective_rank": float(stats["effective_rank"]),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step % int(config.get("logging_steps", 1)) == 0:
            if is_noop:
                LOGGER.info("step=%d id=%s loss=%.6f noop_rms=%.4f grad=%.4f", step, record["id"], row["loss"], row["student_rms"], row["grad_norm"])
            else:
                LOGGER.info("step=%d id=%s loss=%.6f cos=%.4f rel=%.4f grad=%.4f", step, record["id"], row["loss"], row["cosine"], row["relative_error"], row["grad_norm"])
        if step % int(config.get("save_steps", 100)) == 0 or step == max_steps:
            _save(model, output_dir, step, config)


if __name__ == "__main__":
    main()
