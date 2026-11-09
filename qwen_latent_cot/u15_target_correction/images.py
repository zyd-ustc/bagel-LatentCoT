"""One shared scene extent for every paired image and deterministic PNG output."""

import numpy as np
from PIL import Image
import torch

# Paired files are the same scene only when their aspect ratios agree. A pair
# beyond this tolerance needs declared crops; guessing would rescale the subject.
RATIO_TOLERANCE = 0.01


def _rect_for_ratio(ratio, width, height):
    """The largest centred rect with this aspect ratio inside the canvas."""
    if ratio <= width / height:
        w, h = max(1, min(width, round(height * ratio))), height
    else:
        w, h = width, max(1, min(height, round(width * ratio)))
    left, top = (width - w) // 2, (height - h) // 2
    return left, top, left + w, top + h


def valid_mask(rect, width, height, device="cpu", dtype=torch.float32):
    left, top, right, bottom = rect
    mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    mask[:, :, top:bottom, left:right] = 1
    return mask


def _canvas(image, width, height, device, rect, crop=None):
    left, top, right, bottom = rect
    region = image.convert("RGB")
    if crop is not None:
        region = region.crop(crop)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(
        region.resize((right - left, bottom - top), Image.Resampling.LANCZOS),
        (left, top),
    )
    return (
        torch.from_numpy(np.array(canvas)).permute(2, 0, 1)[None].to(
            device=device, dtype=torch.float32
        )
        / 127.5
        - 1
    )


def _declared_crop(value, size, side):
    """A crop in pixels of the original file, or ``None`` for the whole file."""
    if value in (None, "full"):
        return None
    try:
        left, top, right, bottom = [int(round(float(v))) for v in value]
    except (TypeError, ValueError) as error:
        raise ValueError("geometry." + side + " must be four pixel coordinates") from error
    if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]):
        raise ValueError("geometry." + side + " lies outside the image")
    return left, top, right, bottom


def _scene_crops(row, source_size, target_size):
    """The crop of each file that covers the same scene extent.

    A pair either covers the whole scene in both files (same aspect ratio) or
    declares that extent through ``geometry``. Anything else is rejected: a
    shared scale on different shapes would shrink one subject inside the canvas
    and turn a local edit label into "also rescale the subject".
    """
    declared = row.get("geometry")
    if declared is not None:
        if not isinstance(declared, dict):
            raise ValueError("geometry must map source and target crops")
        source_crop = _declared_crop(declared.get("source"), source_size, "source")
        target_crop = _declared_crop(declared.get("target"), target_size, "target")
        boxes = (
            source_crop or (0, 0, *source_size),
            target_crop or (0, 0, *target_size),
        )
        ratios = [(box[2] - box[0]) / (box[3] - box[1]) for box in boxes]
        if abs(ratios[0] - ratios[1]) > RATIO_TOLERANCE:
            raise ValueError("declared geometry crops do not share an aspect ratio")
        return source_crop, target_crop

    source_ratio = source_size[0] / source_size[1]
    target_ratio = target_size[0] / target_size[1]
    if abs(source_ratio - target_ratio) > RATIO_TOLERANCE:
        raise ValueError(
            "Paired images cover different scene extents (%.4f vs %.4f); "
            "declare geometry crops or drop the pair" % (source_ratio, target_ratio)
        )
    return None, None


def pair_geometry_error(row, source_size, target_size):
    """Why a pair cannot share one scene extent, or ``None`` when it can."""
    try:
        _scene_crops(row, source_size, target_size)
    except ValueError as error:
        return str(error)
    return None


def load_pair(row, width, height, device):
    """Map the shared scene extent of a source/target pair onto one canvas.

    Both files are resized into the same rect, so a source file at half the
    target resolution still covers the same subject at the same position.
    """
    with Image.open(row["target_image"]) as target, Image.open(row["source_image"]) as source:
        source_crop, target_crop = _scene_crops(row, source.size, target.size)
        native = source_crop or (0, 0, *source.size)
        rect = _rect_for_ratio(
            (native[2] - native[0]) / (native[3] - native[1]), width, height
        )
        images = [
            _canvas(source, width, height, device, rect, source_crop),
            _canvas(target, width, height, device, rect, target_crop),
        ]
        valid = valid_mask(rect, width, height, device, images[0].dtype)
    return (*images, valid, rect)


def load_canvas(path, width, height, device, crop=None):
    """One image on the protocol canvas, optionally through a declared crop."""
    with Image.open(path) as image:
        region = _declared_crop(crop, image.size, "source") if crop is not None else None
        box = region or (0, 0, *image.size)
        rect = _rect_for_ratio((box[2] - box[0]) / (box[3] - box[1]), width, height)
        canvas = _canvas(image, width, height, device, rect, region)
    return canvas, valid_mask(rect, width, height, device, canvas.dtype), rect


def save_image(path, tensor):
    array = (
        ((tensor.detach().float().cpu()[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5)
        .round()
        .numpy()
        .astype(np.uint8)
    )
    Image.fromarray(array).save(path)
