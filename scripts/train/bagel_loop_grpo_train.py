#!/usr/bin/env python3
"""Quality-constrained GRPO for BAGEL semantic recurrent-state LoRA."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file, save_file

from qwen_latent_cot.bagel.flow_grpo import quality_constrained_advantages
from qwen_latent_cot.bagel.loop import (
    TEXT_ATTENTION_PROJECTIONS,
    load_loop_adapter_state_dict,
    loop_adapter_state_dict,
    loop_trainable_names,
)
from qwen_latent_cot.bagel.loop_grpo import clone_loop_adapter_state, replay_group


LOGGER = logging.getLogger("bagel.loop_grpo.train")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/loop_grpo.yaml")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = dict(yaml.safe_load(handle) or {})
    for key, value in {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "group_size": args.group_size,
    }.items():
        if value not in (None, ""):
            config[key] = value
    required = (
        "model_path",
        "adapter_path",
        "data_path",
        "output_dir",
        "geneval_url",
        "diffusion_rm_repo",
        "flux_rm_config",
        "flux_rm_checkpoint",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"missing required GRPO configuration: {missing}")
    if int(config.get("group_size", 4)) < 2:
        raise ValueError("GRPO group_size must be at least 2")
    if not config.get("sde_step_indices"):
        raise ValueError("GRPO requires at least one selected stochastic SDE step")
    if int(config.get("policy_epochs", 1)) < 1:
        raise ValueError("policy_epochs must be at least 1")
    if str(config.get("loop_state_mode", "semantic_token")) not in {
        "semantic_token",
        "kv_prefix",
    }:
        raise ValueError("loop_state_mode must be semantic_token or kv_prefix")
    if int(config.get("loop_state_tokens", 16)) < 1:
        raise ValueError("loop_state_tokens must be at least 1")
    if config.get("semantic_reward_type") != "geneval2_soft_tifa_log_gm":
        raise ValueError(
            "RL requires semantic_reward_type=geneval2_soft_tifa_log_gm"
        )
    return config


def _load_prompts(path: str) -> list[Dict[str, Any]]:
    rows = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # The distributed GenEval dump has a known truncated final row.
            # Ignore only that final non-empty row; fail closed elsewhere.
            if line_number == len(lines):
                LOGGER.warning(
                    "ignoring truncated final metadata row at %s:%d", path, line_number
                )
                continue
            raise
        if not str(row.get("prompt", "")).strip():
            raise ValueError(f"empty prompt at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no GenEval rows found in {path}")
    return rows


def _validate_adapter_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    adapter = Path(str(config["adapter_path"]))
    metadata_path = adapter.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"loop adapter metadata is required for contract validation: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") not in {
        "bagel_native_loop_format_adapter_v6",
        "bagel_native_loop_grpo_adapter_v6",
        "bagel_semantic_state_flow_adapter_v7",
        "bagel_semantic_state_grpo_adapter_v7",
    }:
        raise RuntimeError(
            "GRPO requires a compatible v6 initializer or v7 semantic-state adapter; "
            f"got schema={metadata.get('schema')!r}"
        )
    expected = {
        "loop_start_layer": int(config["loop_start_layer"]),
        "loop_end_layer": int(config["loop_end_layer"]),
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_alpha": int(config.get("lora_alpha", 16)),
        "include_text_kv_lora": bool(config.get("include_text_kv_lora", False)),
    }
    if metadata.get("schema") in {
        "bagel_semantic_state_flow_adapter_v7",
        "bagel_semantic_state_grpo_adapter_v7",
    }:
        expected.update(
            loop_state_mode=str(config.get("loop_state_mode", "semantic_token")),
            loop_state_tokens=int(config.get("loop_state_tokens", 16)),
            loop_draft_state_scale=float(
                config.get("loop_draft_state_scale", 0.2)
            ),
        )
    mismatches = [
        f"{key}: adapter={metadata.get(key)!r}, GRPO={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if metadata.get("objective") not in {
        "depth1_velocity_format_distillation",
        "cross_step_local_flow",
        "quality_constrained_grpo",
    }:
        mismatches.append(
            "objective: adapter must be a v6 format-only or GRPO adapter"
        )
    if mismatches:
        raise RuntimeError(
            "SFT and GRPO loop contracts differ; refusing to train: "
            + "; ".join(mismatches)
        )
    return metadata


def _initial_noise(model, image_shape: tuple[int, int], seed: int) -> torch.Tensor:
    height, width = image_shape
    rows = (height // int(model.latent_downsample)) * (
        width // int(model.latent_downsample)
    )
    channels = int(model.latent_channel) * int(model.latent_patch_size) ** 2
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((rows, channels), generator=generator, dtype=torch.float32)


def _contexts(inferencer, prompt: str):
    full = inferencer.init_gen_context()
    visual_only = deepcopy(full)
    full = inferencer.update_context_text(prompt, full)
    text_only = deepcopy(full)
    return full, visual_only, text_only


def _save_adapter(model, output_dir: Path, step: int, config: Mapping[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"loop_grpo_adapter_step_{step:07d}"
    state = loop_adapter_state_dict(model)
    save_file(state, str(output_dir / f"{stem}.safetensors"))
    metadata = {
        "schema": "bagel_semantic_state_grpo_adapter_v7",
        "objective": "quality_constrained_grpo",
        "step": int(step),
        "base_adapter": str(config["adapter_path"]),
        "loop_start_layer": int(config["loop_start_layer"]),
        "loop_end_layer": int(config["loop_end_layer"]),
        "loop_state_scale": float(config["loop_state_scale"]),
        "loop_draft_state_scale": float(config.get("loop_draft_state_scale", 0.2)),
        "loop_state_mode": str(config.get("loop_state_mode", "semantic_token")),
        "loop_state_tokens": int(config.get("loop_state_tokens", 16)),
        "loop_draft_state_scale": float(
            config.get("loop_draft_state_scale", 0.2)
        ),
        "loop_state_timestep_threshold": float(
            config.get("loop_state_timestep_threshold", 0.0)
        ),
        "lora_rank": int(config["lora_rank"]),
        "lora_alpha": int(config["lora_alpha"]),
        "include_text_kv_lora": bool(config.get("include_text_kv_lora", False)),
        "policy_epochs": int(config.get("policy_epochs", 1)),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _average_gradients(parameters, world_size: int) -> None:
    if world_size <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("a loop LoRA parameter has no gradient before all-reduce")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _broadcast_parameters(parameters, distributed: bool) -> None:
    """Give every manually distributed rank identical adapter initialization."""

    if not distributed:
        return
    for parameter in parameters:
        dist.broadcast(parameter.data, src=0)


def _scalar_metrics(values: Mapping[str, torch.Tensor | float]) -> Dict[str, float]:
    return {
        key: float(value.detach().float().mean())
        if isinstance(value, torch.Tensor)
        else float(value)
        for key, value in values.items()
    }


def main() -> None:
    args = _arguments()
    config = _load_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("BAGEL loop GRPO requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
    else:
        rank = 0
    device = torch.device("cuda", local_rank)
    seed = int(config.get("seed", 42))
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    prompts = _load_prompts(str(config["data_path"]))
    min_atom_count = int(config.get("min_atom_count", 7))
    invalid_rows = [
        index
        for index, row in enumerate(prompts)
        if not row.get("vqa_list")
        or len(row.get("skills", [])) != len(row["vqa_list"])
        or int(row.get("atom_count", 0)) < min_atom_count
    ]
    if invalid_rows:
        raise ValueError(
            "GRPO data must be GenEval2 per-atom metadata with atomicity >= "
            f"{min_atom_count}; invalid rows={invalid_rows[:8]}"
        )

    # Heavy dependencies stay below argument/config validation so bad paths fail early.
    from qwen_latent_cot.bagel.backbone import BagelBackbone
    from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
    from qwen_latent_cot.bagel.rewards import (
        FluxLatentReward,
        GenEvalRewardClient,
        audit_bagel_flux_vae_contract,
    )

    for required_path in (
        "model_path",
        "adapter_path",
        "diffusion_rm_repo",
        "flux_rm_config",
        "flux_rm_checkpoint",
    ):
        if not Path(str(config[required_path])).exists():
            raise FileNotFoundError(
                f"configured {required_path} does not exist: {config[required_path]}"
            )
    adapter_metadata = _validate_adapter_contract(config)
    semantic_reward = GenEvalRewardClient(
        str(config["geneval_url"]),
        timeout_seconds=float(config.get("geneval_timeout_seconds", 120)),
    )
    semantic_reward.check_available()

    backbone = BagelBackbone(
        {
            "model_path": str(config["model_path"]),
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "num_image_tokens": int(config.get("num_image_tokens", 4900)),
        }
    ).load()
    model = backbone.bagel
    vae = backbone.vae_model
    assert model is not None and vae is not None
    vae_contract = audit_bagel_flux_vae_contract(vae)
    start_layer = int(config["loop_start_layer"])
    end_layer = int(config["loop_end_layer"])
    backbone.apply_loop_trainable_policy(
        start_layer=start_layer,
        end_layer=end_layer,
        rank=int(config.get("lora_rank", 8)),
        alpha=int(config.get("lora_alpha", 16)),
        dropout=float(config.get("lora_dropout", 0.0)),
        include_text_kv=bool(config.get("include_text_kv_lora", False)),
    )
    missing_adapter_keys = load_loop_adapter_state_dict(
        model,
        load_file(str(config["adapter_path"]), device="cpu"),
        allow_missing_projections=(
            TEXT_ATTENTION_PROJECTIONS
            if bool(config.get("include_text_kv_lora", False))
            else ()
        ),
    )
    model.to(device).eval()
    vae.to(device).eval()
    names = loop_trainable_names(
        model,
        start_layer=start_layer,
        end_layer=end_layer,
        include_text_kv=bool(config.get("include_text_kv_lora", False)),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _broadcast_parameters(trainable, distributed)
    reference_adapter = clone_loop_adapter_state(model, device=device)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 1e-6)),
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    ids = backbone.token_ids
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae,
        tokenizer=backbone.tokenizer,
        vae_transform=None,
        vit_transform=None,
        new_token_ids={
            "bos_token_id": int(ids.im_start),
            "eos_token_id": int(ids.im_end),
            "start_of_image": int(ids.vision_start),
            "end_of_image": int(ids.vision_end),
        },
    )
    quality_reward = FluxLatentReward(
        diffusion_rm_repo=str(config["diffusion_rm_repo"]),
        config_path=str(config["flux_rm_config"]),
        checkpoint_path=str(config["flux_rm_checkpoint"]),
        device=device,
        noise_level=float(config.get("flux_rm_noise_level", 0.05)),
        noise_levels=config.get("flux_rm_noise_levels"),
        noise_seed=int(config.get("flux_rm_noise_seed", 17)),
    )

    output_dir = Path(str(config["output_dir"]))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    **config,
                    "world_size": world_size,
                    "vae_contract": vae_contract,
                    "adapter_metadata": adapter_metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info(
            "policy=loop attention LoRA layers=[%d,%d) tensors=%d params=%d "
            "text_kv=%s initialized_keys=%d",
            start_layer,
            end_layer,
            len(names),
            sum(parameter.numel() for parameter in trainable),
            bool(config.get("include_text_kv_lora", False)),
            len(missing_adapter_keys),
        )
    if distributed:
        dist.barrier()

    image_shape = (
        int(config.get("image_height", 512)),
        int(config.get("image_width", 512)),
    )
    group_size = int(config.get("group_size", 4))
    max_steps = int(config.get("max_steps", 1))
    policy_epochs = int(config.get("policy_epochs", 1))
    sde_steps = tuple(int(index) for index in config["sde_step_indices"])
    generation_common = {
        "cfg_text_scale": float(config.get("cfg_text_scale", 4.0)),
        "cfg_img_scale": 1.0,
        "cfg_interval": tuple(config.get("cfg_interval", (0.4, 1.0))),
        "num_timesteps": int(config.get("num_steps", 30)),
        "timestep_shift": float(config.get("timestep_shift", 3.0)),
        "sde_step_indices": sde_steps,
        "sde_noise_level": float(config.get("sde_noise_level", 0.8)),
        "loop_state_mode": str(config.get("loop_state_mode", "semantic_token")),
        "loop_state_tokens": int(config.get("loop_state_tokens", 16)),
    }
    log_path = output_dir / f"metrics_rank{rank:02d}.jsonl"
    for step in range(1, max_steps + 1):
        prompt_index = ((step - 1) * world_size + rank) % len(prompts)
        metadata = prompts[prompt_index]
        prompt = str(metadata["prompt"])
        base_images, loop_images = [], []
        base_latents, loop_latents = [], []
        trajectories, replay_contexts = [], []
        rollout_seed_base = seed + step * 100_000 + rank * 10_000

        for group_index in range(group_size):
            rollout_seed = rollout_seed_base + group_index
            init_noise = _initial_noise(model, image_shape, rollout_seed)
            base_context, base_visual_only, base_text_only = _contexts(inferencer, prompt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base_image, base_latent = inferencer.gen_image(
                    image_shape,
                    base_context,
                    cfg_text_precontext=base_visual_only,
                    cfg_img_precontext=base_text_only,
                    init_noise=init_noise,
                    return_latent=True,
                    sde_seed=rollout_seed,
                    **generation_common,
                )
            loop_context, loop_visual_only, loop_text_only = _contexts(inferencer, prompt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rollout = inferencer.gen_image(
                    image_shape,
                    loop_context,
                    cfg_text_precontext=loop_visual_only,
                    cfg_img_precontext=loop_text_only,
                    init_noise=init_noise,
                    return_trajectory=True,
                    loop_start_layer=start_layer,
                    loop_end_layer=end_layer,
                    loop_state_scale=float(config.get("loop_state_scale", 0.2)),
                    loop_state_timestep_threshold=float(
                        config.get("loop_state_timestep_threshold", 0.0)
                    ),
                    sde_seed=rollout_seed,
                    **generation_common,
                )
            base_images.append(base_image)
            loop_images.append(rollout["image"])
            base_latents.append(base_latent.detach())
            loop_latents.append(rollout["latent"].detach())
            trajectories.append(rollout["trajectory"])
            replay_contexts.append(rollout["replay_context"])

        reward_metadata = [metadata] * (2 * group_size)
        semantic = semantic_reward.score(
            base_images + loop_images, reward_metadata, only_strict=True
        ).to(device)
        quality = quality_reward.score(
            base_latents + loop_latents,
            [prompt] * (2 * group_size),
            image_shape=image_shape,
        ).to(device)
        rewards = quality_constrained_advantages(
            semantic[group_size:],
            semantic[:group_size],
            quality[group_size:],
            quality[:group_size],
            quality_tolerance=float(config.get("quality_tolerance", 0.0)),
            quality_penalty_weight=float(config.get("quality_penalty_weight", 1.0)),
        )

        replay = None
        grad_norm = None
        for policy_epoch in range(policy_epochs):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay = replay_group(
                    model,
                    trajectories,
                    replay_contexts,
                    reference_adapter,
                    rewards.advantage.to(device),
                    clip_range=float(config.get("clip_range", 1e-4)),
                    kl_beta=float(config.get("kl_beta", 0.01)),
                )
            ratio_deviation = float(replay.ratio_max_deviation.detach())
            if step == 1 and policy_epoch == 0 and ratio_deviation > float(
                config.get("ratio_gate_tolerance", 5e-4)
            ):
                raise RuntimeError(
                    "exact-replay gate failed before the first optimizer step: "
                    f"max |ratio-1|={ratio_deviation:.6g}"
                )
            replay.loss.backward()
            _average_gradients(trainable, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(config.get("max_grad_norm", 1.0))
            )
            optimizer.step()
        assert replay is not None and grad_norm is not None

        metrics = _scalar_metrics(
            {
                "loss": replay.loss,
                "policy_loss": replay.policy_loss,
                "kl": replay.kl,
                "ratio_mean": replay.ratio_mean,
                "ratio_max_deviation": replay.ratio_max_deviation,
                "semantic_base": semantic[:group_size].mean(),
                "semantic_loop": semantic[group_size:].mean(),
                "semantic_delta": rewards.semantic_delta.mean(),
                "quality_base": quality[:group_size].mean(),
                "quality_loop": quality[group_size:].mean(),
                "quality_delta": rewards.quality_delta.mean(),
                "quality_penalty": rewards.quality_penalty.mean(),
                "objective_std": rewards.objective.std(unbiased=False),
                "grad_norm": grad_norm,
                "policy_epochs": float(policy_epochs),
            }
        )
        record = {
            "step": step,
            "rank": rank,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "tag": metadata.get("tag"),
            **metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step == 1 or step % int(config.get("logging_steps", 1)) == 0:
            LOGGER.info("%s", json.dumps(record, ensure_ascii=False))
        if rank == 0 and (
            step % int(config.get("save_steps", 1)) == 0 or step == max_steps
        ):
            _save_adapter(model, output_dir, step, config)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
