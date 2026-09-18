#!/usr/bin/env python3
"""Can the UND expert read a VAE latent through the KV cache (no ViT, no decode)?

BAGEL's understanding path is pretrained on ViT tokens
(``interleave_inference`` uses ``vae=False, vit=True``). Because MoT attention is
shared and only the per-token projections pick an expert, a ``mode="und"`` text
query can attend to cached rows written by
``forward_cache_update_vae_latent`` (GEN expert). Whether it can *understand*
them is an empirical question.

This probe answers it on a REAL image, zero-shot, by comparing three context
constructions for the same question:

  vit   : forward_cache_update_vit(vit_transform(image))            (official)
  vae   : forward_cache_update_vae_latent(image -> VAE latent)      (proposed)
  both  : vae rows then vit rows

If ``vae`` (or ``both``) describes the image correctly, the loop can drop the
per-round decode entirely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from bagel_common import add_native_model_args, load_native_bagel
from qwen_latent_cot.bagel import accelerator
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--image", required=True)
    parser.add_argument("--user-text", default="Describe this image in detail.")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--modalities", default="vit,vae,both")
    return parser.parse_args()


@torch.no_grad()
def image_to_latent(inferencer, image, device):
    """Replicate prepare_vae_images + forward_cache_update_vae's encode+patchify."""

    model = inferencer.model
    prepared = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
    gen_input, _, _ = model.prepare_vae_images(
        curr_kvlens=[0],
        curr_rope=[0],
        images=[prepared],
        transforms=inferencer.vae_transform,
        new_token_ids=inferencer.new_token_ids,
    )
    padded_images = gen_input["padded_images"].to(device)
    shapes = gen_input["patchified_vae_latent_shapes"]
    vae_dtype = next(inferencer.vae_model.parameters()).dtype
    padded_latent = inferencer.vae_model.encode(padded_images.to(dtype=vae_dtype))

    p = int(model.latent_patch_size)
    chunks = []
    for latent, (h, w) in zip(padded_latent, shapes):
        latent = latent[:, : h * p, : w * p].reshape(
            int(model.latent_channel), int(h), p, int(w), p
        )
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(
            -1, p * p * int(model.latent_channel)
        )
        chunks.append(latent)
    image_shape = (int(padded_images.shape[2]), int(padded_images.shape[3]))
    return torch.cat(chunks, dim=0), image_shape


def main() -> None:
    args = parse_args()
    modalities = [m.strip() for m in str(args.modalities).split(",") if m.strip()]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[model] loading frozen native BAGEL checkpoint", flush=True)
    backbone, inferencer = load_native_bagel(args)
    device = accelerator.resolve_device(args.device)

    image = Image.open(args.image).convert("RGB")
    image.save(output_dir / "input.png")
    latent, used_shape = image_to_latent(inferencer, image, device)
    print(f"[latent] shape={tuple(latent.shape)} image_shape={used_shape}", flush=True)
    torch.save(latent.cpu(), output_dir / "vae_latent.pt")

    results = {}
    for modality in modalities:
        ctx = inferencer.init_gen_context()
        if modality in ("vae", "both"):
            ctx = inferencer.update_context_vae_latent(
                latent, used_shape, ctx, timestep=0.0
            )
        if modality in ("vit", "both"):
            ctx = inferencer.update_context_image(image, ctx, vae=False, vit=True)
        ctx = inferencer.update_context_text(str(args.user_text), ctx)
        text = inferencer.gen_text(
            ctx, max_length=int(args.max_tokens), do_sample=False
        )
        results[modality] = text
        print("=" * 28, f"UND reading {modality.upper()} KV", "=" * 28)
        print(text, flush=True)

    (output_dir / "und_reads_latent_probe.json").write_text(
        json.dumps(
            {
                "image": str(Path(args.image).expanduser().resolve()),
                "user_text": str(args.user_text),
                "latent_shape": list(latent.shape),
                "image_shape": list(used_shape),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output_dir / 'und_reads_latent_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
