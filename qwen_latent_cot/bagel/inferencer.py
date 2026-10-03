"""Native T2I preparation, CFG caches, and image decoding.

Both generation and training use Bagel.forward_t2i_loop; this module owns no
recurrent algorithm. Prompt caches remain fixed across denoising steps.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from PIL import Image

from .accelerator import autocast_for
from .modeling.bagel.qwen2_navit import NaiveCache


def to_device(values, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


@dataclass
class T2ICondition:
    inputs: dict
    shape: tuple[int, int]


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, new_token_ids):
        self.model, self.vae_model, self.tokenizer = model, vae_model, tokenizer
        self.new_token_ids = new_token_ids

    @property
    def device(self):
        return next(self.model.parameters()).device

    def empty_context(self, batch=1):
        return {
            "kv_lens": [0] * batch,
            "ropes": [0] * batch,
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }

    @torch.no_grad()
    def prepare_condition(self, prompts, image_shape=(1024, 1024)):
        height, width = image_shape
        if (
            height % self.model.latent_downsample
            or width % self.model.latent_downsample
        ):
            raise ValueError("image dimensions must be divisible by latent_downsample")
        context = self.empty_context(len(prompts))
        prompt_input, lens, ropes = self.model.prepare_prompts(
            curr_kvlens=context["kv_lens"],
            curr_rope=context["ropes"],
            prompts=prompts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        cache = self.model.forward_cache_update_text(
            context["past_key_values"], **to_device(prompt_input, self.device)
        )
        flow = self.model.prepare_vae_latent(
            lens, ropes, [image_shape] * len(prompts), self.new_token_ids
        )
        flow.pop("packed_init_noises")
        flow = to_device(flow, self.device)
        flow["past_key_values"] = cache
        for branch, branch_lens, branch_ropes, branch_cache in [
            (
                "text",
                [0] * len(prompts),
                [0] * len(prompts),
                self.empty_context(len(prompts))["past_key_values"],
            ),
            ("img", lens, ropes, deepcopy(cache)),
        ]:
            cfg = self.model.prepare_vae_latent_cfg(
                branch_lens, branch_ropes, [image_shape] * len(prompts)
            )
            cfg = to_device(cfg, self.device)
            flow.update(
                {
                    key.replace("cfg_", f"cfg_{branch}_", 1): value
                    for key, value in cfg.items()
                }
            )
            flow[f"cfg_{branch}_past_key_values"] = branch_cache
        return T2ICondition(flow, tuple(image_shape))

    @torch.no_grad()
    def generate(self, prompts, *, image_shape=(1024, 1024), seed=0, **sampler):
        condition = self.prepare_condition(prompts, image_shape)
        count = condition.inputs["packed_vae_token_indexes"].numel()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(count, self.model.patch_latent_dim, generator=generator).to(
            self.device
        )
        with autocast_for(self.device):
            latents = self.model.generate_image(
                packed_init_noises=noise, **condition.inputs, **sampler
            )
        return [self.decode_image(latent, image_shape) for latent in latents]

    @torch.no_grad()
    def decode_image(self, latent, image_shape):
        height, width = image_shape
        h, w = (
            height // self.model.latent_downsample,
            width // self.model.latent_downsample,
        )
        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent).reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(
            latent.to(next(self.vae_model.parameters()).dtype)
        )
        pixels = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(
            torch.uint8
        )
        return Image.fromarray(pixels.cpu().numpy())
