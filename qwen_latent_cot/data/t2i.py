"""JSONL prompt/image records; no pair teacher or semantic memory targets."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class T2IDataset(Dataset):
    def __init__(self, path, image_size=512):
        self.path = Path(path)
        self.image_size = image_size
        self.rows = []
        for number, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row.get("prompt"), str)
                or not row["prompt"].strip()
                or not row.get("image")
            ):
                raise ValueError(f"line {number}: prompt and image are required")
            if row.get("bucket", "ordinary") not in {
                "ordinary",
                "structural",
                "easy",
                "noop",
            }:
                raise ValueError(
                    f"line {number}: Stage 1 accepts ordinary/structural/easy/noop buckets"
                )
            self.rows.append(row)
        if not self.rows:
            raise ValueError("empty T2I dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["image"])
        if not path.is_absolute():
            path = self.path.parent / path
        with Image.open(path) as image:
            image = ImageOps.fit(
                image.convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.LANCZOS,
            )
            pixels = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1
        return {
            "prompt": row["prompt"],
            "pixels": torch.from_numpy(pixels).permute(2, 0, 1),
        }


def patchify_latents(latent, patch_size):
    batch, channels, height, width = latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError("VAE latent dimensions must be divisible by patch_size")
    value = latent.reshape(
        batch,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return torch.einsum("nchpwq->nhwpqc", value).reshape(-1, channels * patch_size**2)


def sample_flow_state(clean, timestep, noise):
    """x_t=(1-t)*x1+t*epsilon; velocity points from data to noise."""
    return (1 - timestep) * clean + timestep * noise, noise - clean
