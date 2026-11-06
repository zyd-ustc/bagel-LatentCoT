"""One shared fit rectangle for every paired image and deterministic PNG output."""

import numpy as np
from PIL import Image
import torch


def _fit(size, width, height):
    source_width, source_height = size
    scale = min(width / source_width, height / source_height)
    w, h = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    left, top = (width - w) // 2, (height - h) // 2
    return left, top, left + w, top + h


def valid_mask(rect, width, height, device="cpu", dtype=torch.float32):
    left, top, right, bottom = rect
    mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    mask[:, :, top:bottom, left:right] = 1
    return mask


def _canvas(image, width, height, device, rect):
    left, top, right, bottom = rect
    canvas = Image.new("RGB", (width, height))
    canvas.paste(
        image.convert("RGB").resize((right - left, bottom - top), Image.Resampling.LANCZOS),
        (left, top),
    )
    return (
        torch.from_numpy(np.array(canvas)).permute(2, 0, 1)[None].to(
            device=device, dtype=torch.float32
        )
        / 127.5
        - 1
    )


def load_pair(row, width, height, device):
    """Apply one common scale and centering to an actual source/target pair.

    Paired images must share their pixel dimensions. Independent crops would
    turn the paired state label into a different spatial transform.
    """
    with Image.open(row["target_image"]) as target, Image.open(row["source_image"]) as source:
        source_ratio = source.width / source.height
        target_ratio = target.width / target.height
        if abs(source_ratio - target_ratio) > 1e-3:
            raise ValueError("Paired images must share an aspect ratio")
        if source.size == target.size:
            rect = _fit(target.size, width, height)
        else:
            scale = min(
                width / source.width, height / source.height,
                width / target.width, height / target.height,
            )
            w, h = max(1, round(source.width * scale)), max(1, round(source.height * scale))
            left, top = (width - w) // 2, (height - h) // 2
            rect = (left, top, left + w, top + h)
        images = [
            _canvas(image, width, height, device, rect)
            for image in (source, target)
        ]
    return (*images, valid_mask(rect, width, height, device, images[0].dtype), rect)


def load_canvas(path, width, height, device):
    with Image.open(path) as image:
        rect = _fit(image.size, width, height)
        canvas = _canvas(image, width, height, device, rect)
    return canvas, valid_mask(rect, width, height, device, canvas.dtype), rect


def save_image(path, tensor):
    array = (
        ((tensor.detach().float().cpu()[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5)
        .round()
        .numpy()
        .astype(np.uint8)
    )
    Image.fromarray(array).save(path)
