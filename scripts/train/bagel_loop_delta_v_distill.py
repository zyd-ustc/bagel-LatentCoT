#!/usr/bin/env python3
"""Phase-1 structured-reflection delta-velocity distillation for BAGEL."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from PIL import Image
from safetensors.torch import save_file

from qwen_latent_cot.bagel import accelerator
from qwen_latent_cot.bagel.loop import (
    loop_adapter_state_dict,
    loop_trainable_names,
)
from qwen_latent_cot.bagel.loop_distill import (
    delta_velocity_distillation_loss,
    replay_velocity,
    sample_replay_step_indices,
)
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb


LOGGER = logging.getLogger("bagel.loop_delta_v.train")
ADAPTER_SCHEMA = "bagel_loop_delta_velocity_adapter_v8"
OBJECTIVE = "structured_reflection_delta_velocity_distillation"
REQUIRED_FIELDS = (
    "id",
    "source_image",
    "instruction",
    "reflection",
    "edit_type",
    "target_constraints",
    "preserve_constraints",
    "is_noop",
    "difficulty",
    "teacher_valid",
    "teacher_semantic_delta",
    "teacher_preserve_delta",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/loop_delta_v_early_fresh.yaml"
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
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
    missing = [
        key
        for key in ("model_path", "data_path", "output_dir")
        if not config.get(key)
    ]
    if missing:
        raise ValueError(f"missing required Phase-1 configuration: {missing}")
    expected = {
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "loop_recycle_mode": "same_depth",
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
        "round0_memory_write_enabled": False,
        "gen_attention_o_lora": False,
        "k_v_lora": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key, value) != value
    }
    if mismatches:
        raise ValueError(
            "Phase 1.1 is locked to K8/R2 strict-read early-body Q-only; "
            f"mismatches={mismatches}"
        )
    replay_count = int(config.get("replay_states_per_sample", 3))
    if not 2 <= replay_count <= 4:
        raise ValueError("replay_states_per_sample must be in [2, 4]")
    if float(config.get("teacher_preserve_epsilon", 0.02)) < 0.0:
        raise ValueError("teacher_preserve_epsilon must be non-negative")
    if float(config.get("direction_active_threshold", 1e-3)) < 0.0:
        raise ValueError("direction_active_threshold must be non-negative")
    return config


def _english_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _reflection_bullet_counts(reflection: str) -> Tuple[int, int]:
    target = preserve = 0
    section = None
    for raw_line in reflection.splitlines():
        line = raw_line.strip()
        lowered = line.lower().rstrip(":")
        if lowered == "target changes":
            section = "target"
        elif lowered == "preserve":
            section = "preserve"
        elif line.startswith("-"):
            if section == "target":
                target += 1
            elif section == "preserve":
                preserve += 1
    return target, preserve


def validate_phase1_record(
    row: Mapping[str, Any],
    *,
    source_root: Path,
    location: str,
    teacher_min_semantic_delta: float = 0.0,
    teacher_preserve_epsilon: float = 0.02,
) -> Dict[str, Any]:
    missing = [key for key in REQUIRED_FIELDS if key not in row]
    if missing:
        raise ValueError(f"{location}: missing fields {missing}")
    normalized = dict(row)
    for key in ("id", "source_image", "instruction", "reflection"):
        if not str(normalized[key]).strip():
            raise ValueError(f"{location}: {key} must be non-empty")
    reflection = str(normalized["reflection"])
    target_bullets, preserve_bullets = _reflection_bullet_counts(reflection)
    if "target changes:" not in reflection.lower() or "preserve:" not in reflection.lower():
        raise ValueError(
            f"{location}: reflection requires Target changes and Preserve sections"
        )
    first_line = next(
        (line.strip() for line in reflection.splitlines() if line.strip()), ""
    )
    if first_line.upper() != "EDIT PLAN":
        raise ValueError(f"{location}: reflection must start with EDIT PLAN")
    if not 1 <= target_bullets <= 4 or not 1 <= preserve_bullets <= 4:
        raise ValueError(
            f"{location}: reflection bullets must be 1-4 per section, got "
            f"target={target_bullets}, preserve={preserve_bullets}"
        )
    if _english_token_count(reflection) > 120:
        raise ValueError(f"{location}: reflection exceeds 120 English tokens")
    for key in ("edit_type", "target_constraints", "preserve_constraints"):
        if not isinstance(normalized[key], list):
            raise ValueError(f"{location}: {key} must be a JSON list")
    if not normalized["edit_type"]:
        raise ValueError(f"{location}: edit_type must be non-empty")
    if not normalized["preserve_constraints"]:
        raise ValueError(f"{location}: preserve_constraints must be non-empty")
    if not isinstance(normalized["is_noop"], bool):
        raise ValueError(f"{location}: is_noop must be a JSON boolean")
    if normalized["is_noop"]:
        noop_phrases = ("no structural change", "no change required")
        if not any(phrase in reflection.lower() for phrase in noop_phrases):
            raise ValueError(
                f"{location}: noop reflection must explicitly state "
                "No structural change or No change required"
            )
    elif not normalized["target_constraints"]:
        raise ValueError(
            f"{location}: non-noop target_constraints must be non-empty"
        )
    if not isinstance(normalized["difficulty"], (int, float)):
        raise ValueError(f"{location}: difficulty must be numeric")
    if not isinstance(normalized["teacher_valid"], bool):
        raise ValueError(f"{location}: teacher_valid must be a JSON boolean")
    for key in ("teacher_semantic_delta", "teacher_preserve_delta"):
        if not isinstance(normalized[key], (int, float)) or not math.isfinite(
            float(normalized[key])
        ):
            raise ValueError(f"{location}: {key} must be a finite number")
    semantic_delta = float(normalized["teacher_semantic_delta"])
    semantic_valid = (
        semantic_delta >= float(teacher_min_semantic_delta)
        if normalized["is_noop"]
        else semantic_delta > float(teacher_min_semantic_delta)
    )
    score_valid = semantic_valid and float(
        normalized["teacher_preserve_delta"]
    ) >= -float(teacher_preserve_epsilon)
    if normalized["teacher_valid"] and not score_valid:
        raise ValueError(
            f"{location}: teacher_valid conflicts with teacher score deltas"
        )
    source = Path(str(normalized["source_image"])).expanduser()
    if not source.is_absolute():
        source = source_root / source
    if not source.is_file():
        raise FileNotFoundError(f"{location}: source image not found: {source}")
    normalized["source_image"] = str(source.resolve())
    return normalized


def load_phase1_records(
    path: str,
    *,
    teacher_valid_only: bool = True,
    teacher_min_semantic_delta: float = 0.0,
    teacher_preserve_epsilon: float = 0.02,
) -> list[Dict[str, Any]]:
    data_path = Path(path).expanduser().resolve()
    rows = []
    for line_number, line in enumerate(
        data_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        normalized = validate_phase1_record(
            row,
            source_root=data_path.parent,
            location=f"{data_path}:{line_number}",
            teacher_min_semantic_delta=float(teacher_min_semantic_delta),
            teacher_preserve_epsilon=float(teacher_preserve_epsilon),
        )
        if not teacher_valid_only or normalized["teacher_valid"]:
            rows.append(normalized)
    if not rows:
        suffix = " teacher-valid" if teacher_valid_only else ""
        raise ValueError(f"no{suffix} Phase-1 records found in {data_path}")
    return rows


def _initial_noise(model, image_shape: Tuple[int, int], seed: int) -> torch.Tensor:
    height, width = image_shape
    rows = (height // int(model.latent_downsample)) * (
        width // int(model.latent_downsample)
    )
    channels = int(model.latent_channel) * int(model.latent_patch_size) ** 2
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((rows, channels), generator=generator, dtype=torch.float32)


def _append_texts(inferencer, context, texts: Sequence[str]):
    for text in texts:
        context = inferencer.update_context_text(str(text), context)
    return context


def build_edit_contexts(
    inferencer, source_image: Image.Image, instruction: str, reflection: str = ""
):
    source = inferencer.update_context_image(
        source_image,
        inferencer.init_gen_context(),
        vae=True,
        vit=True,
    )
    texts = [str(instruction)] + ([str(reflection)] if reflection else [])
    full = _append_texts(inferencer, deepcopy(source), texts)
    image_removed = _append_texts(inferencer, inferencer.init_gen_context(), texts)
    return {
        "full": full,
        "text_removed": deepcopy(source),
        "image_removed": image_removed,
        "has_visual_condition": True,
    }


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
):
    warmup_steps = max(1, int(round(int(total_steps) * float(warmup_ratio))))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _save_adapter(model, output_dir: Path, step: int, config: Mapping[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"loop_delta_v_adapter_step_{step:07d}"
    save_file(loop_adapter_state_dict(model), str(output_dir / f"{stem}.safetensors"))
    metadata = {
        "schema": ADAPTER_SCHEMA,
        "objective": OBJECTIVE,
        "step": int(step),
        "num_loop_tokens": 8,
        "loop_depth": 2,
        "num_read_rounds": 1,
        "num_write_rounds": 1,
        "loop_recycle_mode": "same_depth",
        "loop_memory_persist": bool(config.get("loop_memory_persist", False)),
        "memory_loop_start_layer": 12,
        "memory_loop_end_layer": 20,
        "round0_memory_write_enabled": False,
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_alpha": int(config.get("lora_alpha", 16)),
        "gen_attention_o_lora": False,
        "k_v_lora": False,
        "loop_memory_trainable": False,
        "teacher": "source+instruction+structured_reflection,K=0",
        "base": "source+instruction,K=0",
        "student": "source+instruction,K=8",
        "teacher_valid_only": bool(config.get("teacher_valid_only", True)),
        "teacher_min_semantic_delta": float(
            config.get("teacher_min_semantic_delta", 0.0)
        ),
        "teacher_preserve_epsilon": float(
            config.get("teacher_preserve_epsilon", 0.02)
        ),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _set_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean())
    return float(value)


def _backward_scaled_replay_loss(
    loss: torch.Tensor, *, state_count: int
) -> torch.Tensor:
    """Backward one replay state and return a graph-free metric value."""

    if int(state_count) < 1:
        raise ValueError("state_count must be positive")
    if not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("non-finite replay-state loss")
    (loss / int(state_count)).backward()
    return loss.detach()


def main() -> None:
    args = _arguments()
    config = _load_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    records = load_phase1_records(
        str(config["data_path"]),
        teacher_valid_only=bool(config.get("teacher_valid_only", True)),
        teacher_min_semantic_delta=float(
            config.get("teacher_min_semantic_delta", 0.0)
        ),
        teacher_preserve_epsilon=float(
            config.get("teacher_preserve_epsilon", 0.02)
        ),
    )
    if args.validate_only:
        LOGGER.info("validated %d Phase-1 records", len(records))
        return

    model_path = Path(str(config["model_path"])).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    device = accelerator.resolve_device(config.get("device", "auto"))
    if not accelerator.is_accelerator(device):
        raise RuntimeError("Phase-1 training requires CUDA or Ascend NPU")
    _set_device(device)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    accelerator.manual_seed_all(seed)

    from qwen_latent_cot.bagel.backbone import BagelBackbone
    from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
    from qwen_latent_cot.bagel.modeling._bagel_utils import ImageTransform

    backbone = BagelBackbone(
        {
            "model_path": str(model_path),
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "num_image_tokens": int(config.get("num_image_tokens", 4900)),
            # Allocate m0 for Student. Base/Teacher still explicitly pass K=0.
            "num_loop_tokens": 8,
            "loop_depth": 2,
            "loop_recycle_mode": "same_depth",
            "loop_memory_persist": bool(config.get("loop_memory_persist", False)),
            "memory_loop_start_layer": 12,
            "memory_loop_end_layer": 20,
            "round0_memory_write_enabled": False,
        }
    ).load()
    model = backbone.bagel
    vae = backbone.vae_model
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
    model.to(device).eval()
    vae.to(device).eval()
    trainable_names = loop_trainable_names(model, start_layer=12, end_layer=20)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if model.loop_memory is None or model.loop_memory.requires_grad:
        raise RuntimeError("Phase-1 requires initialized frozen m0")

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
    scheduler = _cosine_warmup_scheduler(
        optimizer, max_steps, float(config.get("warmup_ratio", 0.03))
    )
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **config,
                "schema": ADAPTER_SCHEMA,
                "objective": OBJECTIVE,
                "device_resolved": str(device),
                "records": len(records),
                "trainable_tensors": trainable_names,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "loaded records=%d device=%s Q-only tensors=%d params=%d persist=%s",
        len(records),
        device,
        len(trainable_names),
        sum(parameter.numel() for parameter in trainable),
        bool(config.get("loop_memory_persist", False)),
    )

    generation = {
        "cfg_text_scale": float(config.get("cfg_text_scale", 4.0)),
        "cfg_img_scale": float(config.get("cfg_img_scale", 2.0)),
        "cfg_interval": tuple(config.get("cfg_interval", (0.0, 1.0))),
        "cfg_renorm_min": float(config.get("cfg_renorm_min", 0.0)),
        "cfg_renorm_type": str(config.get("cfg_renorm_type", "text_channel")),
        "num_timesteps": int(config.get("num_steps", 30)),
        "timestep_shift": float(config.get("timestep_shift", 3.0)),
    }
    num_flow_steps = generation["num_timesteps"] - 1
    replay_count = int(config.get("replay_states_per_sample", 3))
    metrics_path = output_dir / "metrics.jsonl"
    persist = bool(config.get("loop_memory_persist", False))

    for step in range(1, max_steps + 1):
        record_index = (step - 1) % len(records)
        record = records[record_index]
        with Image.open(record["source_image"]) as handle:
            source = inferencer.vae_transform.resize_transform(
                pil_img2rgb(handle.convert("RGB"))
            )
        image_shape = tuple(source.size[::-1])
        rollout_seed = seed + step * 1_000_003
        initial_noise = _initial_noise(model, image_shape, rollout_seed)
        sample_generator = torch.Generator(device="cpu").manual_seed(rollout_seed)
        selected_steps = sample_replay_step_indices(
            num_flow_steps,
            replay_count,
            generator=sample_generator,
        )

        base_contexts = build_edit_contexts(
            inferencer, source, str(record["instruction"])
        )
        teacher_contexts = build_edit_contexts(
            inferencer,
            source,
            str(record["instruction"]),
            str(record["reflection"]),
        )
        base_bundle = inferencer.prepare_velocity_bundle(
            name="base",
            contexts=base_contexts,
            image_shape=image_shape,
            num_loop_tokens=0,
        )
        teacher_bundle = inferencer.prepare_velocity_bundle(
            name="teacher",
            contexts=teacher_contexts,
            image_shape=image_shape,
            num_loop_tokens=0,
        )
        student_bundle = inferencer.prepare_velocity_bundle(
            name="student",
            contexts=base_contexts,
            image_shape=image_shape,
            num_loop_tokens=8,
        )
        if (
            base_bundle.flow_input["packed_loop_token_indexes"].numel() != 0
            or teacher_bundle.flow_input["packed_loop_token_indexes"].numel() != 0
            or student_bundle.flow_input["packed_loop_token_indexes"].numel() != 8
        ):
            raise RuntimeError("Base/Teacher/Student K contract was violated")

        with accelerator.autocast_for(device):
            rollout = inferencer.gen_image(
                image_shape,
                base_contexts["full"],
                cfg_text_precontext=base_contexts["text_removed"],
                cfg_img_precontext=base_contexts["image_removed"],
                init_noise=initial_noise,
                return_trajectory=True,
                capture_step_indices=selected_steps,
                decode_output=False,
                num_loop_tokens=8,
                loop_depth=2,
                loop_uncond_memory=str(config.get("loop_uncond_memory", "m0")),
                loop_recycle_mode="same_depth",
                loop_memory_persist=persist,
                memory_loop_start=12,
                memory_loop_end=20,
                round0_memory_write_enabled=False,
                **generation,
            )
        states = tuple(rollout["trajectory"])
        if tuple(state["step_index"] for state in states) != selected_steps:
            raise RuntimeError("rollout did not return the requested replay states")

        optimizer.zero_grad(set_to_none=True)
        per_state = []
        replay_common = {
            "cfg_text_scale": generation["cfg_text_scale"],
            "cfg_img_scale": generation["cfg_img_scale"],
            "cfg_interval": generation["cfg_interval"],
            "cfg_renorm_min": generation["cfg_renorm_min"],
            "cfg_renorm_type": generation["cfg_renorm_type"],
            "loop_depth": 2,
            "loop_recycle_mode": "same_depth",
            "memory_loop_start": 12,
            "memory_loop_end": 20,
            "round0_memory_write_enabled": False,
            "loop_uncond_memory": str(config.get("loop_uncond_memory", "m0")),
        }
        for state in states:
            with accelerator.autocast_for(device):
                with torch.no_grad():
                    base_velocity = replay_velocity(
                        model, inferencer, base_bundle, state, **replay_common
                    ).velocity
                    teacher_velocity = replay_velocity(
                        model, inferencer, teacher_bundle, state, **replay_common
                    ).velocity
                student_velocity = replay_velocity(
                    model, inferencer, student_bundle, state, **replay_common
                ).velocity
                result = delta_velocity_distillation_loss(
                    student_velocity=student_velocity,
                    teacher_velocity=teacher_velocity,
                    base_velocity=base_velocity,
                    is_noop=bool(record["is_noop"]),
                    normalization_floor=float(config.get("normalization_floor", 1e-4)),
                    direction_active_threshold=float(
                        config.get("direction_active_threshold", 1e-3)
                    ),
                    lambda_delta_v=float(config.get("lambda_delta_v", 1.0)),
                    lambda_dir=float(config.get("lambda_dir", 0.1)),
                    lambda_over=float(config.get("lambda_over", 0.05)),
                    lambda_noop=float(config.get("lambda_noop", 1.0)),
                    overshoot_gamma=float(config.get("overshoot_gamma", 1.5)),
                )
            detached_loss = _backward_scaled_replay_loss(
                result.loss, state_count=len(states)
            )
            per_state.append(
                {
                    "loss": detached_loss,
                    "delta_v_loss": result.delta_v_loss,
                    "direction_loss": result.direction_loss,
                    "overshoot_loss": result.overshoot_loss,
                    "noop_loss": result.noop_loss,
                    "cosine": result.cosine,
                    "relative_error": result.relative_error,
                    "student_rms": result.student_rms,
                    "teacher_rms": result.teacher_rms,
                    "direction_active": result.direction_active,
                }
            )
            del result, student_velocity, base_velocity, teacher_velocity
        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing_gradients:
            raise RuntimeError(
                f"trainable loop tensors without gradient: {missing_gradients[:8]}"
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config.get("max_grad_norm", 1.0))
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()

        def mean_metric(name: str) -> float:
            return sum(_scalar(item[name]) for item in per_state) / len(per_state)

        metrics = {
            "step": step,
            "sample_id": str(record["id"]),
            "record_index": record_index,
            "is_noop": bool(record["is_noop"]),
            "teacher_semantic_delta": float(record["teacher_semantic_delta"]),
            "teacher_preserve_delta": float(record["teacher_preserve_delta"]),
            "selected_step_indices": list(selected_steps),
            "loss": mean_metric("loss"),
            "delta_v_loss": mean_metric("delta_v_loss"),
            "direction_loss": mean_metric("direction_loss"),
            "overshoot_loss": mean_metric("overshoot_loss"),
            "noop_loss": mean_metric("noop_loss"),
            "cosine": mean_metric("cosine"),
            "relative_error": mean_metric("relative_error"),
            "student_rms": mean_metric("student_rms"),
            "teacher_rms": mean_metric("teacher_rms"),
            "direction_active_fraction": sum(
                float(item["direction_active"]) for item in per_state
            )
            / len(per_state),
            "grad_norm": _scalar(grad_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if step == 1 or step % int(config.get("logging_steps", 1)) == 0:
            LOGGER.info("%s", json.dumps(metrics, ensure_ascii=False))
        if step % int(config.get("save_steps", 100)) == 0 or step == max_steps:
            _save_adapter(model, output_dir, step, config)


if __name__ == "__main__":
    main()
