"""Unchanged-schema targets for BAGEL semantic-flow loop post-training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from qwen_latent_cot.data.cort_rows import load_canonical_cort_rows
from qwen_latent_cot.utils import load_image


class LoopFlowDataset(Dataset):
    """Original prompt paired with the terminal target image only."""

    def __init__(
        self,
        data_paths: Sequence[str],
        *,
        include_open_trajectories: bool = False,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        items: List[Dict[str, Any]] = []
        for row in load_canonical_cort_rows(list(data_paths)):
            if not bool(row.get("is_terminal_chain", False)) and not include_open_trajectories:
                continue
            images = [str(path) for path in row["images"]]
            if not images or not Path(images[-1]).is_file():
                continue
            prompt = str(row["prompt"]).strip()
            if not prompt:
                continue
            fixes = [
                str(review.get("fix", "") or "").strip()
                for review in row.get("reviews", [])
            ]
            edit_instruction = ". ".join(value for value in fixes if value)
            items.append(
                {
                    "sample_id": f"{row['sample_id']}__loop_t2i",
                    "prompt": prompt,
                    # Keep the canonical field but never substitute the prompt:
                    # this value is a training-only semantic label.
                    "edit_instruction": edit_instruction,
                    "target_image_path": images[-1],
                    "trajectory_id": str(row.get("trajectory_id", row["sample_id"])),
                }
            )
        if sample_size is not None:
            size = int(sample_size)
            if not 0 < size <= len(items):
                raise ValueError(
                    f"sample_size={size} must be in [1, {len(items)}]"
                )
            items = random.Random(int(seed)).sample(items, size)
        if not items:
            raise RuntimeError("no valid terminal text-to-image targets were found")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


@dataclass(frozen=True)
class LoopFlowCollatorConfig:
    vae_image_size: int = 1024
    vae_min_image_size: int = 512
    vae_image_stride: int = 16
    latent_downsample: int = 16
    image_mean: Sequence[float] = (0.5, 0.5, 0.5)
    image_std: Sequence[float] = (0.5, 0.5, 0.5)
    fixed_timestep: Optional[float] = None


class LoopFlowCollator:
    def __init__(self, cfg: LoopFlowCollatorConfig) -> None:
        from .modeling._bagel_utils import ImageTransform

        if cfg.fixed_timestep is not None and not 0.0 <= float(cfg.fixed_timestep) <= 1.0:
            raise ValueError("fixed_timestep must be in [0, 1]")
        self.cfg = cfg
        self.transform = ImageTransform(
            max_image_size=int(cfg.vae_image_size),
            min_image_size=int(cfg.vae_min_image_size),
            image_stride=int(cfg.vae_image_stride),
            max_pixels=int(cfg.vae_image_size) ** 2,
            image_mean=list(cfg.image_mean),
            image_std=list(cfg.image_std),
        )

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not examples:
            raise ValueError("loop-flow collator received an empty batch")
        tensors = [
            self.transform(load_image(str(example["target_image_path"])))
            for example in examples
        ]
        stride = int(self.cfg.latent_downsample)
        shapes = [
            (int(tensor.shape[-2]) // stride, int(tensor.shape[-1]) // stride)
            for tensor in tensors
        ]
        max_height = max(int(tensor.shape[-2]) for tensor in tensors)
        max_width = max(int(tensor.shape[-1]) for tensor in tensors)
        padded = tensors[0].new_zeros((len(tensors), 3, max_height, max_width))
        for index, tensor in enumerate(tensors):
            padded[index, :, : tensor.shape[-2], : tensor.shape[-1]] = tensor
        result: Dict[str, Any] = {
            "prompts": [str(example["prompt"]) for example in examples],
            "semantic_targets": [
                str(example.get("edit_instruction") or "")
                for example in examples
            ],
            "padded_vae_images": padded,
            "patchified_vae_latent_shapes": shapes,
            "metadata": [dict(example) for example in examples],
        }
        if self.cfg.fixed_timestep is not None:
            result["timesteps"] = torch.full(
                (len(examples),), float(self.cfg.fixed_timestep), dtype=torch.float32
            )
        return result


__all__ = ["LoopFlowCollator", "LoopFlowCollatorConfig", "LoopFlowDataset"]
