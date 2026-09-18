# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous utilities for BAGEL inference.

This module consolidates essential image processing functions from BAGEL's
data_utils and transforms modules.
"""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image
import torch
from torch.nn.attention.flex_attention import or_masks, and_masks
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


# =============================================================================
# Sparse Mask (from data_utils.py)
# =============================================================================


def create_sparse_mask(
    document_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    device,
):
    """Create sparse attention mask for BAGEL model."""

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (
            full_and_noise_seq_id[q_idx] >= 0
        )

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~(
            (noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])
        )

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []
    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        value = i if mode in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if mode == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)
    document_id = torch.cat(
        [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
    ).to(device)

    return and_masks(
        or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask
    )


# =============================================================================
# Image Conversion Utilities (from data_utils.py)
# =============================================================================


def pil_img2rgb(image: Image.Image) -> Image.Image:
    """Convert PIL image to RGB format, handling transparency."""
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")
    return image


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape image tensor into patches."""
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(
    img_h: int, img_w: int, patch_size: int, max_num_patches_per_side: int
) -> torch.Tensor:
    """Get flattened position IDs using extrapolation."""
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(
    img_h: int, img_w: int, patch_size: int, max_num_patches_per_side: int
) -> torch.Tensor:
    """Get flattened position IDs using interpolation."""
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(
        1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side
    )
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (
        bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w
    ).flatten()
    return pos_ids


def add_special_tokens(tokenizer):
    """Add special tokens to tokenizer for BAGEL."""
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []
    for token in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]:
        if token not in all_special_tokens:
            new_tokens.append(token)

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        start_of_image=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        end_of_image=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    return tokenizer, new_token_ids, num_new_tokens


# =============================================================================
# Image Transform Utilities (from transforms.py)
# =============================================================================


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize image to fit within size constraints while keeping divisibility."""

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias: bool = True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value: int, stride: int) -> int:
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width: int, height: int, scale: float) -> Tuple[int, int]:
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img: Image.Image, img_num: int = 1) -> Image.Image:
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(
            img, (new_height, new_width), self.interpolation, antialias=self.antialias
        )


class ImageTransform:
    """Standard image transform for BAGEL inference."""

    def __init__(
        self,
        max_image_size: int,
        min_image_size: int,
        image_stride: int,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list = [0.5, 0.5, 0.5],
        image_std: list = [0.5, 0.5, 0.5],
    ):
        self.stride = image_stride
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=image_mean, std=image_std, inplace=True
        )

    def __call__(self, img: Image.Image, img_num: int = 1) -> torch.Tensor:
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


# =============================================================================
# Attention Mask Utilities (from data_utils.py)
# =============================================================================


def prepare_attention_mask_per_sample(
    split_lens: list,
    attn_modes: list,
    device: str = "cpu",
) -> torch.Tensor:
    """Prepare attention mask for a sample with mixed attention modes."""
    sample_len = sum(split_lens)
    attention_mask = torch.zeros(
        (sample_len, sample_len), dtype=torch.bool, device=device
    )

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise"]
        if attn_mode == "causal":
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones(
                (s, s), device=device
            ).tril()
            attention_mask[csum : csum + s, :csum] = 1
        else:
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
            attention_mask[csum : csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )
    return attention_mask


__all__ = [
    "create_sparse_mask",
    "pil_img2rgb",
    "patchify",
    "get_flattened_position_ids_extrapolate",
    "get_flattened_position_ids_interpolate",
    "add_special_tokens",
    "prepare_attention_mask_per_sample",
    "MaxLongEdgeMinShortEdgeResize",
    "ImageTransform",
]
