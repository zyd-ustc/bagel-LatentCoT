"""Aspect-preserving T2I records and bucket-aware batches."""

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

from qwen_latent_cot.bagel.modeling._bagel_utils import ImageTransform

BUCKET_WEIGHTS = {"ordinary": 0.4, "structural": 0.3, "easy": 0.2, "noop": 0.1}


class T2IDataset(Dataset):
    def __init__(
        self, path, image_size=512, *, stride=16, min_image_size=None, max_pixels=None
    ):
        self.path = Path(path)
        self.rows = []
        self.transform = ImageTransform(
            image_size,
            min_image_size or max(stride, image_size // 2),
            stride,
            max_pixels=max_pixels or image_size**2,
        )
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
            row.setdefault("bucket", "ordinary")
            if row["bucket"] not in BUCKET_WEIGHTS:
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
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_shape = (image.height, image.width)
            # Native BAGEL resize retains the entire frame; no center crop.
            pixels = self.transform(image)
        return {
            "prompt": row["prompt"],
            "pixels": pixels,
            "bucket": row["bucket"],
            "image_shape": tuple(pixels.shape[-2:]),
            "original_shape": original_shape,
            "index": index,
        }


class BucketBatchSampler(Sampler):
    """One semantic bucket per batch; shapes can differ within that bucket.

    Explicit requested weights require all positive buckets to exist. With no
    explicit weights, canonical proportions normalize over available buckets.
    Draws use replacement, so small buckets retain their configured mass.
    """

    def __init__(self, dataset, batch_size, num_batches, *, weights=None, seed=0):
        self.indices = {
            bucket: [i for i, row in enumerate(dataset.rows) if row["bucket"] == bucket]
            for bucket in BUCKET_WEIGHTS
        }
        requested = BUCKET_WEIGHTS if weights is None else weights
        if (
            not requested
            or set(requested) - set(BUCKET_WEIGHTS)
            or any(value < 0 for value in requested.values())
        ):
            raise ValueError("bucket weights must be nonnegative known buckets")
        if weights is not None and any(
            value > 0 and not self.indices[bucket]
            for bucket, value in requested.items()
        ):
            raise ValueError("a positive configured bucket has no training records")
        selected = {
            bucket: float(value)
            for bucket, value in requested.items()
            if value > 0 and self.indices[bucket]
        }
        if not selected or batch_size < 1 or num_batches < 1:
            raise ValueError(
                "bucket sampler requires positive batch size, batches, and available mass"
            )
        total = sum(selected.values())
        self.effective_weights = {
            bucket: value / total for bucket, value in selected.items()
        }
        self.batch_size, self.num_batches, self.seed = batch_size, num_batches, seed

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        names = list(self.effective_weights)
        weights = torch.tensor(list(self.effective_weights.values()))
        for _ in range(self.num_batches):
            bucket = names[int(torch.multinomial(weights, 1, generator=generator))]
            indices = self.indices[bucket]
            draws = torch.randint(
                len(indices), (self.batch_size,), generator=generator
            ).tolist()
            yield [indices[i] for i in draws]


def collate_t2i(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}


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
