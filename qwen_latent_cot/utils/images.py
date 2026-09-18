"""Image helpers."""

from __future__ import annotations

import tarfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image


@lru_cache(maxsize=16)
def _open_tar(path: str) -> tarfile.TarFile:
    return tarfile.open(path, "r:*")


def load_image(path: str | Path | bytes | dict | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        img = path
    elif isinstance(path, (bytes, bytearray)):
        img = Image.open(BytesIO(path))
    elif isinstance(path, dict):
        raw = path.get("bytes")
        if raw is not None:
            img = Image.open(BytesIO(raw))
        else:
            tar_path = str(path.get("tar_path", "") or "").strip()
            member = str(path.get("member", "") or "").strip()
            if not tar_path or not member:
                raise ValueError("Image dict must contain 'bytes' or ('tar_path', 'member').")
            handle = _open_tar(tar_path).extractfile(member)
            if handle is None:
                raise ValueError(f"Missing tar member: {member} in {tar_path}")
            img = Image.open(BytesIO(handle.read()))
    else:
        img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def save_image(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def resize_by_token_budget(
    images: list[Image.Image],
    global_max_pixels: int = 2000 * 28 * 28,
    per_img_max_pixels: int = 1280 * 28 * 28,
    divisor: int = 28,
) -> tuple[list[Image.Image], list[tuple[int, int]] | None]:
    import math

    if not images:
        return images, None

    total = sum(img.width * img.height for img in images)
    ratio = math.sqrt(global_max_pixels / total) if total > global_max_pixels else 1.0

    processed: list[Image.Image] = []
    new_sizes: list[tuple[int, int]] = []
    changed = False

    for img in images:
        w = int(img.width * ratio)
        h = int(img.height * ratio)
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)

        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, (int(w * r) // divisor) * divisor)
            h = max(divisor, (int(h * r) // divisor) * divisor)

        if w != img.width or h != img.height:
            changed = True
            processed.append(img.resize((w, h), Image.BICUBIC))
            new_sizes.append((w, h))
        else:
            processed.append(img)
            new_sizes.append((img.width, img.height))

    if not changed:
        return images, None
    return processed, new_sizes
