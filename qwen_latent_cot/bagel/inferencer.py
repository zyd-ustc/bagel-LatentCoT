# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any, Tuple

from PIL import Image
import torch

from qwen_latent_cot.bagel import accelerator
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

GEN_THINK_SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"""


@dataclass
class ImageConditionBundle:
    """Native BAGEL flow inputs for one conditional/dual-CFG cache triplet."""

    name: str
    full_context: Dict[str, Any]
    text_removed_context: Dict[str, Any]
    image_removed_context: Dict[str, Any]
    flow_input: Dict[str, Any]
    cfg_text_input: Dict[str, Any]
    cfg_img_input: Dict[str, Any]
    has_visual_condition: bool = False


@dataclass
class MemoryReadConditionBundle:
    """Minimal conditional state for prefix -> strict Read -> STOP."""

    name: str
    context: Dict[str, Any]
    flow_input: Dict[str, Any]


def _move_to_device(generation_input, device):
    """Mirror of BAGEL eval/gen/gen_images_mp.py:22 -- move every tensor in
    a `prepare_*` output dict onto `device`. Needed when the model is loaded
    via plain `model.to(device)` (no accelerate hooks), because the upstream
    `prepare_*` functions build their tensors on CPU."""
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def filter_old_prompt(input_lists, enabled: bool = True):
    if not enabled:
        return list(input_lists)
    image_indices = [
        index
        for index, item in enumerate(input_lists)
        if isinstance(item, Image.Image)
    ]
    if not image_indices:
        return list(input_lists)
    last_image = image_indices[-1]
    return [
        item
        for index, item in enumerate(input_lists)
        if not (isinstance(item, str) and index < last_image)
    ]


def flowedit_time_branch(timestep: float, n_min: float, n_max: float) -> str:
    """Window on shifted t in generate_image: src / delta / tar."""

    if float(n_min) > float(n_max):
        raise ValueError("n_min must be <= n_max")
    value = float(timestep)
    if value > float(n_max):
        return "src"
    if value < float(n_min):
        return "tar"
    return "delta"


class InterleaveInferencer:
    def __init__(
        self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        # `prepare_*` builds CPU tensors; we need to know where the model lives
        # so we can move them before forward_cache_update_* / generate_* calls.
        # Single-device assumption (BAGEL's official inferencer is single-device);
        # use the first model parameter as the canonical device.
        self.device = next(model.parameters()).device

    def init_gen_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference,

        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = _move_to_device(generation_input, self.device)

        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference,

        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = _move_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = _move_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_vae_latent(self, latent, image_shape, gen_context, timestep=0.0):
        """Append a raw patchified VAE latent as a GEN-side condition.

        ``latent`` is ``[num_image_tokens, latent_channel*patch_size^2]`` in the
        same space as BAGEL's flow target, so an ``x0_hat`` draft estimate can be
        injected without decoding to pixels and re-encoding.
        """

        generation_input, kv_lens, ropes = self.model.prepare_vae_latent_condition(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            packed_latents=[latent],
            image_shapes=[tuple(image_shape)],
            new_token_ids=self.new_token_ids,
            timestep=float(timestep),
        )
        generation_input = _move_to_device(generation_input, self.device)
        past_key_values = self.model.forward_cache_update_vae_latent(
            gen_context["past_key_values"], **generation_input
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def generate_image_reflection(
        self,
        image,
        *,
        user_text: str,
        system_prompt: Optional[str] = None,
        max_length: int = 128,
        do_sample: bool = False,
        temperature: float = 0.3,
    ) -> str:
        """Full understanding-side pass over a draft image (ViT only, no VAE)."""

        gen_context = self.init_gen_context()
        if system_prompt:
            gen_context = self.update_context_text(str(system_prompt), gen_context)
        # Match the official understanding path: pil_img2rgb + vae_transform
        # resize before update_context_image applies the ViT transform.
        image = self.vae_transform.resize_transform(pil_img2rgb(image))
        gen_context = self.update_context_image(image, gen_context, vae=False, vit=True)
        gen_context = self.update_context_text(str(user_text), gen_context)
        return self.gen_text(
            gen_context,
            max_length=int(max_length),
            do_sample=bool(do_sample),
            temperature=float(temperature),
        )

    @torch.no_grad()
    def prepare_velocity_bundle(
        self,
        *,
        name: str,
        contexts: Dict[str, Any],
        image_shape: Tuple[int, int],
        init_noise: Optional[torch.Tensor] = None,
        num_loop_tokens: Optional[int] = 0,
    ) -> ImageConditionBundle:
        """Materialize one native BAGEL GEN/dual-CFG velocity layout.

        ``num_loop_tokens`` is a per-call property of the current query.  The
        cached source/text context is unchanged, so Base/Teacher can use K=0
        while Student uses K>0 on the same loaded model instance.
        """

        required = {"full", "text_removed", "image_removed"}
        missing = required.difference(contexts)
        if missing:
            raise ValueError(f"condition contexts missing keys: {sorted(missing)}")

        full = contexts["full"]
        text_removed = contexts["text_removed"]
        image_removed = contexts["image_removed"]
        loop_k = 0 if num_loop_tokens is None else int(num_loop_tokens)
        flow_input = self.model.prepare_vae_latent(
            curr_kvlens=full["kv_lens"],
            curr_rope=full["ropes"],
            image_sizes=[tuple(image_shape)],
            new_token_ids=self.new_token_ids,
            num_loop_tokens=loop_k,
        )
        flow_input = _move_to_device(flow_input, self.device)
        if init_noise is not None:
            expected = flow_input["packed_init_noises"]
            if tuple(init_noise.shape) != tuple(expected.shape):
                raise ValueError(
                    f"init_noise shape {tuple(init_noise.shape)} != {tuple(expected.shape)}"
                )
            flow_input["packed_init_noises"] = init_noise.to(
                device=self.device, dtype=expected.dtype
            )

        cfg_text_input = self.model.prepare_vae_latent_cfg(
            curr_kvlens=text_removed["kv_lens"],
            curr_rope=text_removed["ropes"],
            image_sizes=[tuple(image_shape)],
            num_loop_tokens=loop_k,
        )
        cfg_img_input = self.model.prepare_vae_latent_cfg(
            curr_kvlens=image_removed["kv_lens"],
            curr_rope=image_removed["ropes"],
            image_sizes=[tuple(image_shape)],
            num_loop_tokens=loop_k,
        )
        return ImageConditionBundle(
            name=str(name),
            full_context=full,
            text_removed_context=text_removed,
            image_removed_context=image_removed,
            flow_input=flow_input,
            cfg_text_input=_move_to_device(cfg_text_input, self.device),
            cfg_img_input=_move_to_device(cfg_img_input, self.device),
            has_visual_condition=bool(contexts.get("has_visual_condition", False)),
        )

    @torch.no_grad()
    def prepare_memory_read_bundle(
        self,
        *,
        name: str,
        context: Dict[str, Any],
        image_shape: Tuple[int, int],
        num_loop_tokens: int = 8,
    ) -> MemoryReadConditionBundle:
        """Materialize only the conditional query needed by Phase 1.1."""

        loop_k = int(num_loop_tokens)
        if loop_k <= 0:
            raise ValueError("memory Read requires num_loop_tokens > 0")
        flow_input = self.model.prepare_vae_latent(
            curr_kvlens=context["kv_lens"],
            curr_rope=context["ropes"],
            image_sizes=[tuple(image_shape)],
            new_token_ids=self.new_token_ids,
            num_loop_tokens=loop_k,
        )
        return MemoryReadConditionBundle(
            name=str(name),
            context=context,
            flow_input=_move_to_device(flow_input, self.device),
        )

    def build_memory_read_kwargs(
        self,
        *,
        x_t: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        condition: MemoryReadConditionBundle,
        memory_loop_start: int = 12,
        memory_loop_end: int = 20,
    ) -> Dict[str, Any]:
        raw_timestep = torch.as_tensor(
            timestep, device=x_t.device, dtype=x_t.dtype
        ).reshape(-1)
        if int(raw_timestep.numel()) != 1:
            raise ValueError("memory Read requires one shared timestep")
        flow = condition.flow_input
        return {
            "x_t": x_t,
            "timestep": torch.full(
                (int(x_t.shape[0]),),
                float(raw_timestep.detach().float()[0]),
                dtype=x_t.dtype,
                device=x_t.device,
            ),
            "packed_vae_token_indexes": flow["packed_vae_token_indexes"],
            "packed_vae_position_ids": flow["packed_vae_position_ids"],
            "packed_text_ids": flow["packed_text_ids"],
            "packed_text_indexes": flow["packed_text_indexes"],
            "packed_indexes": flow["packed_indexes"],
            "packed_position_ids": flow["packed_position_ids"],
            "packed_seqlens": flow["packed_seqlens"],
            "key_values_lens": flow["key_values_lens"],
            "past_key_values": condition.context["past_key_values"],
            "packed_key_value_indexes": flow["packed_key_value_indexes"],
            "packed_loop_token_indexes": flow["packed_loop_token_indexes"],
            "packed_boundary_token_indexes": flow[
                "packed_boundary_token_indexes"
            ],
            "memory_loop_start": int(memory_loop_start),
            "memory_loop_end": int(memory_loop_end),
        }

    @torch.no_grad()
    def prepare_image_condition_bundle(self, **kwargs) -> ImageConditionBundle:
        """Backward-compatible alias for ``prepare_velocity_bundle``."""

        return self.prepare_velocity_bundle(**kwargs)

    def build_image_velocity_kwargs(
        self,
        *,
        x_t: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        condition: ImageConditionBundle,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        within_step_loop_start: Optional[int] = None,
        within_step_loop_end: Optional[int] = None,
        within_step_loop_repeat: int = 1,
        within_step_loop_damping: float = 1.0,
    ) -> Dict[str, Any]:
        """Build native BAGEL flow kwargs without choosing K=0 or loop forward."""

        raw_timestep = torch.as_tensor(
            timestep, device=x_t.device, dtype=x_t.dtype
        ).reshape(-1)
        if int(raw_timestep.numel()) != 1:
            raise ValueError("velocity replay requires one shared timestep")
        value = float(raw_timestep.detach().float()[0])
        in_cfg_interval = float(cfg_interval[0]) < value <= float(cfg_interval[1])
        text_scale = float(cfg_text_scale) if in_cfg_interval else 1.0
        image_scale = (
            float(cfg_img_scale)
            if in_cfg_interval and condition.has_visual_condition
            else 1.0
        )
        timestep_tensor = torch.full(
            (int(x_t.shape[0]),),
            value,
            dtype=x_t.dtype,
            device=x_t.device,
        )
        flow = condition.flow_input
        cfg_text = condition.cfg_text_input
        cfg_img = condition.cfg_img_input
        return dict(
            x_t=x_t,
            timestep=timestep_tensor,
            packed_vae_token_indexes=flow["packed_vae_token_indexes"],
            packed_vae_position_ids=flow["packed_vae_position_ids"],
            packed_text_ids=flow["packed_text_ids"],
            packed_text_indexes=flow["packed_text_indexes"],
            packed_position_ids=flow["packed_position_ids"],
            packed_indexes=flow["packed_indexes"],
            packed_seqlens=flow["packed_seqlens"],
            key_values_lens=flow["key_values_lens"],
            past_key_values=condition.full_context["past_key_values"],
            packed_key_value_indexes=flow["packed_key_value_indexes"],
            cfg_renorm_min=float(cfg_renorm_min),
            cfg_renorm_type=str(cfg_renorm_type),
            cfg_text_scale=text_scale,
            cfg_text_packed_position_ids=cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=cfg_text["cfg_key_values_lens"],
            cfg_text_past_key_values=condition.text_removed_context[
                "past_key_values"
            ],
            cfg_text_packed_key_value_indexes=cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_scale=image_scale,
            cfg_img_packed_position_ids=cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=cfg_img["cfg_key_values_lens"],
            cfg_img_past_key_values=condition.image_removed_context[
                "past_key_values"
            ],
            cfg_img_packed_key_value_indexes=cfg_img[
                "cfg_packed_key_value_indexes"
            ],
            within_step_loop_start=within_step_loop_start,
            within_step_loop_end=within_step_loop_end,
            within_step_loop_repeat=int(within_step_loop_repeat),
            within_step_loop_damping=float(within_step_loop_damping),
            packed_boundary_token_indexes=flow["packed_boundary_token_indexes"],
        )

    @torch.no_grad()
    def predict_image_velocity(
        self,
        *,
        x_t: torch.Tensor,
        timestep: float,
        condition: ImageConditionBundle,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        within_step_loop_start: Optional[int] = None,
        within_step_loop_end: Optional[int] = None,
        within_step_loop_repeat: int = 1,
        within_step_loop_damping: float = 1.0,
    ):
        """Evaluate one native BAGEL guided velocity without advancing ``x_t``."""

        self.model.language_model.model.enable_taylorseer = False
        kwargs = self.build_image_velocity_kwargs(
            x_t=x_t,
            timestep=timestep,
            condition=condition,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            within_step_loop_start=within_step_loop_start,
            within_step_loop_end=within_step_loop_end,
            within_step_loop_repeat=within_step_loop_repeat,
            within_step_loop_damping=within_step_loop_damping,
        )
        return self.model.predict_image_velocity(**kwargs)

    @torch.no_grad()
    def predict_dynamic_prompt_velocity(
        self,
        *,
        x_t: torch.Tensor,
        timestep: float,
        condition: ImageConditionBundle,
        alpha: float,
        body_start: int = 12,
        body_end: int = 20,
        delta_mode: str = "dynamic",
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        return_diagnostics: bool = False,
    ):
        """Evaluate one fixed-x_t anchored prompt Read/Write counterfactual."""

        self.model.language_model.model.enable_taylorseer = False
        kwargs = self.build_image_velocity_kwargs(
            x_t=x_t,
            timestep=timestep,
            condition=condition,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
        )
        return self.model._forward_dynamic_prompt(
            **kwargs,
            prompt_body_start=int(body_start),
            prompt_body_end=int(body_end),
            prompt_alpha=float(alpha),
            prompt_delta_mode=str(delta_mode),
            return_diagnostics=bool(return_diagnostics),
        )

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
        init_noise: Optional[torch.Tensor] = None,
        return_latent: bool = False,
        sde_step_indices: Optional[tuple[int, ...]] = None,
        sde_noise_level: float = 0.0,
        sde_seed: int = 0,
        return_trajectory: bool = False,
        return_loop_diagnostics: bool = False,
        num_loop_tokens: Optional[int] = None,
        capture_step_indices: Optional[tuple[int, ...]] = None,
        decode_output: bool = True,
        loop_depth: Optional[int] = None,
        loop_uncond_memory: Optional[str] = None,
        loop_recycle_mode: Optional[str] = None,
        loop_memory_persist: Optional[bool] = None,
        memory_loop_start: Optional[int] = None,
        memory_loop_end: Optional[int] = None,
        round0_memory_write_enabled: Optional[bool] = None,
        dynamic_prompt_alpha: Optional[float] = None,
        dynamic_prompt_body_start: int = 12,
        dynamic_prompt_body_end: int = 20,
        dynamic_prompt_step_fraction: float = 0.35,
        dynamic_prompt_delta_mode: str = "dynamic",
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
            num_loop_tokens=num_loop_tokens,
        )
        generation_input = _move_to_device(generation_input, self.device)
        if init_noise is not None:
            expected_noise = generation_input["packed_init_noises"]
            if tuple(init_noise.shape) != tuple(expected_noise.shape):
                raise ValueError(
                    "init_noise shape does not match BAGEL's native latent geometry: "
                    f"got {tuple(init_noise.shape)}, expected {tuple(expected_noise.shape)} "
                    f"for image_shape={tuple(image_shape)}"
                )
            # Keep the dtype produced by BAGEL's prepare_vae_latent (fp32 in the
            # reference implementation). The caller can now share the exact
            # random draw with another generation backend without changing the
            # native denoiser's numerical contract.
            generation_input["packed_init_noises"] = init_noise.to(
                device=self.device,
                dtype=expected_noise.dtype,
            )

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext["past_key_values"]
        kv_lens_cfg = cfg_text_precontext["kv_lens"]
        ropes_cfg = cfg_text_precontext["ropes"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
            num_loop_tokens=num_loop_tokens,
        )
        generation_input_cfg_text = _move_to_device(
            generation_input_cfg_text, self.device
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]
        kv_lens_cfg = cfg_img_precontext["kv_lens"]
        ropes_cfg = cfg_img_precontext["ropes"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
            num_loop_tokens=num_loop_tokens,
        )
        generation_input_cfg_img = _move_to_device(
            generation_input_cfg_img, self.device
        )

        generation_result = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
            enable_taylorseer=enable_taylorseer,
            sde_step_indices=sde_step_indices,
            sde_noise_level=sde_noise_level,
            sde_seed=sde_seed,
            return_trajectory=return_trajectory,
            return_loop_diagnostics=bool(return_loop_diagnostics),
            capture_step_indices=capture_step_indices,
            loop_depth=loop_depth,
            loop_uncond_memory=loop_uncond_memory,
            loop_recycle_mode=loop_recycle_mode,
            loop_memory_persist=loop_memory_persist,
            memory_loop_start=memory_loop_start,
            memory_loop_end=memory_loop_end,
            round0_memory_write_enabled=round0_memory_write_enabled,
            dynamic_prompt_alpha=dynamic_prompt_alpha,
            dynamic_prompt_body_start=int(dynamic_prompt_body_start),
            dynamic_prompt_body_end=int(dynamic_prompt_body_end),
            dynamic_prompt_step_fraction=float(dynamic_prompt_step_fraction),
            dynamic_prompt_delta_mode=str(dynamic_prompt_delta_mode),
        )

        if return_trajectory:
            unpacked_latent, trajectory = generation_result
        else:
            unpacked_latent = generation_result
            trajectory = None

        latent = unpacked_latent[0]
        image = self.decode_image(latent, image_shape) if decode_output else None
        if return_trajectory:
            replay_context = {
                "packed_vae_token_indexes": generation_input[
                    "packed_vae_token_indexes"
                ],
                "packed_vae_position_ids": generation_input["packed_vae_position_ids"],
                "packed_text_ids": generation_input["packed_text_ids"],
                "packed_text_indexes": generation_input["packed_text_indexes"],
                "packed_boundary_token_indexes": generation_input[
                    "packed_boundary_token_indexes"
                ],
                "packed_position_ids": generation_input["packed_position_ids"],
                "packed_indexes": generation_input["packed_indexes"],
                "packed_seqlens": generation_input["packed_seqlens"],
                "key_values_lens": generation_input["key_values_lens"],
                "past_key_values": past_key_values,
                "packed_key_value_indexes": generation_input[
                    "packed_key_value_indexes"
                ],
                "cfg_renorm_min": float(cfg_renorm_min),
                "cfg_renorm_type": cfg_renorm_type,
                "cfg_interval": tuple(cfg_interval),
                "cfg_text_scale": float(cfg_text_scale),
                "cfg_text_packed_position_ids": generation_input_cfg_text[
                    "cfg_packed_position_ids"
                ],
                "cfg_text_packed_query_indexes": generation_input_cfg_text[
                    "cfg_packed_query_indexes"
                ],
                "cfg_text_key_values_lens": generation_input_cfg_text[
                    "cfg_key_values_lens"
                ],
                "cfg_text_past_key_values": cfg_text_past_key_values,
                "cfg_text_packed_key_value_indexes": generation_input_cfg_text[
                    "cfg_packed_key_value_indexes"
                ],
                "cfg_img_scale": float(cfg_img_scale),
                "cfg_img_packed_position_ids": generation_input_cfg_img[
                    "cfg_packed_position_ids"
                ],
                "cfg_img_packed_query_indexes": generation_input_cfg_img[
                    "cfg_packed_query_indexes"
                ],
                "cfg_img_key_values_lens": generation_input_cfg_img[
                    "cfg_key_values_lens"
                ],
                "cfg_img_past_key_values": cfg_img_past_key_values,
                "cfg_img_packed_key_value_indexes": generation_input_cfg_img[
                    "cfg_packed_key_value_indexes"
                ],
                "packed_loop_token_indexes": generation_input.get(
                    "packed_loop_token_indexes"
                ),
                "recycle_mode": str(
                    loop_recycle_mode
                    if loop_recycle_mode is not None
                    else getattr(self.model.config, "loop_recycle_mode", "same_depth")
                ),
                "memory_loop_repeat": int(
                    loop_depth
                    if loop_depth is not None
                    else getattr(self.model.config, "loop_depth", 2)
                ),
                "memory_loop_start": int(
                    memory_loop_start
                    if memory_loop_start is not None
                    else getattr(self.model.config, "memory_loop_start_layer", 16)
                ),
                "memory_loop_end": int(
                    memory_loop_end
                    if memory_loop_end is not None
                    else getattr(self.model.config, "memory_loop_end_layer", 24)
                ),
                "round0_memory_write_enabled": bool(
                    round0_memory_write_enabled
                    if round0_memory_write_enabled is not None
                    else getattr(
                        self.model.config,
                        "round0_memory_write_enabled",
                        getattr(self.model.config, "round0_gen_reads_memory", False),
                    )
                ),
                "loop_memory_persist": bool(
                    loop_memory_persist
                    if loop_memory_persist is not None
                    else getattr(self.model.config, "loop_memory_persist", False)
                ),
                "loop_uncond_memory": str(
                    loop_uncond_memory
                    if loop_uncond_memory is not None
                    else getattr(self.model.config, "loop_uncond_memory", "m0")
                ),
                "embed_memory": getattr(self.model, "loop_memory", None),
                "num_loop_tokens": int(
                    generation_input["packed_loop_token_indexes"].numel()
                )
                // max(1, int(generation_input["packed_vae_seqlens"].numel())),
            }
            return {
                "image": image,
                "latent": latent,
                "trajectory": trajectory,
                "replay_context": replay_context,
            }
        if return_latent:
            return image, latent
        return image

    def encode_image(self, image, image_shape):
        """Inverse of ``decode_image``: RGB → packed VAE tokens.

        Caller should already have applied ``vae_transform.resize_transform``;
        ``image_shape`` is that PIL size as ``(H, W)``, matching
        ``interleave_inference``.
        """

        height, width = int(image_shape[0]), int(image_shape[1])
        image = pil_img2rgb(image)
        if image.size != (width, height):
            raise ValueError(
                f"encode_image expected PIL size {(width, height)}, got {image.size}"
            )
        tensor = self.vae_transform(image).unsqueeze(0).to(device=self.device)
        latent = self.vae_model.encode(tensor)
        patch = int(self.model.latent_patch_size)
        channel = int(self.model.latent_channel)
        rows = height // int(self.model.latent_downsample)
        cols = width // int(self.model.latent_downsample)
        latent = latent.reshape(1, channel, rows, patch, cols, patch)
        latent = torch.einsum("nchpwq->nhwpqc", latent)
        return latent.reshape(rows * cols, patch * patch * channel)

    def _prepare_flowedit_side(self, ctx, cfg_text_ctx, cfg_img_ctx, image_shape):
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
            num_loop_tokens=0,
        )
        generation_input = _move_to_device(generation_input, self.device)
        cfg_text_input = _move_to_device(
            self.model.prepare_vae_latent_cfg(
                curr_kvlens=cfg_text_ctx["kv_lens"],
                curr_rope=cfg_text_ctx["ropes"],
                image_sizes=[image_shape],
                num_loop_tokens=0,
            ),
            self.device,
        )
        cfg_img_input = _move_to_device(
            self.model.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_ctx["kv_lens"],
                curr_rope=cfg_img_ctx["ropes"],
                image_sizes=[image_shape],
                num_loop_tokens=0,
            ),
            self.device,
        )
        return generation_input, cfg_text_input, cfg_img_input

    def _forward_flow_side(
        self,
        *,
        x_t,
        timestep,
        ctx,
        cfg_text_ctx,
        cfg_img_ctx,
        generation_input,
        cfg_text_input,
        cfg_img_input,
        cfg_text_scale,
        cfg_img_scale,
        cfg_renorm_min,
        cfg_renorm_type,
    ):
        return self.model._forward_flow(
            x_t=x_t,
            timestep=timestep,
            packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
            packed_vae_position_ids=generation_input["packed_vae_position_ids"],
            packed_text_ids=generation_input["packed_text_ids"],
            packed_text_indexes=generation_input["packed_text_indexes"],
            packed_indexes=generation_input["packed_indexes"],
            packed_position_ids=generation_input["packed_position_ids"],
            packed_seqlens=generation_input["packed_seqlens"],
            key_values_lens=generation_input["key_values_lens"],
            past_key_values=ctx["past_key_values"],
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            packed_boundary_token_indexes=generation_input[
                "packed_boundary_token_indexes"
            ],
            cfg_renorm_min=float(cfg_renorm_min),
            cfg_renorm_type=str(cfg_renorm_type),
            cfg_text_scale=float(cfg_text_scale),
            cfg_text_packed_position_ids=cfg_text_input["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_text_input["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=cfg_text_input["cfg_key_values_lens"],
            cfg_text_past_key_values=cfg_text_ctx["past_key_values"],
            cfg_text_packed_key_value_indexes=cfg_text_input[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_scale=float(cfg_img_scale),
            cfg_img_packed_position_ids=cfg_img_input["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=cfg_img_input["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=cfg_img_input["cfg_key_values_lens"],
            cfg_img_past_key_values=cfg_img_ctx["past_key_values"],
            cfg_img_packed_key_value_indexes=cfg_img_input[
                "cfg_packed_key_value_indexes"
            ],
            cfg_type="parallel",
        )

    @torch.no_grad()
    def gen_image_flowedit(
        self,
        image_shape,
        x_src0,
        src_ctx,
        src_cfg_text_ctx,
        src_cfg_img_ctx,
        tar_ctx,
        tar_cfg_text_ctx,
        tar_cfg_img_ctx,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        n_min=0.2,
        n_max=0.8,
        n_avg=1,
    ):
        """FlowEdit integrator on official ``_forward_flow``. Does not touch KV."""

        self.model.language_model.model.enable_taylorseer = False
        src_in, src_ct, src_ci = self._prepare_flowedit_side(
            src_ctx, src_cfg_text_ctx, src_cfg_img_ctx, image_shape
        )
        tar_in, tar_ct, tar_ci = self._prepare_flowedit_side(
            tar_ctx, tar_cfg_text_ctx, tar_cfg_img_ctx, image_shape
        )
        if not torch.equal(
            src_in["packed_vae_token_indexes"], tar_in["packed_vae_token_indexes"]
        ):
            raise ValueError("src/tar packed_vae_token_indexes must match")
        expected = src_in["packed_init_noises"]
        if tuple(x_src0.shape) != tuple(expected.shape):
            raise ValueError(
                f"x_src0 shape {tuple(x_src0.shape)} != {tuple(expected.shape)}"
            )
        x_src0 = x_src0.to(device=self.device, dtype=expected.dtype)
        x_edit = x_src0.clone()
        timesteps, dts = self.model.prepare_image_schedule(
            int(num_timesteps), float(timestep_shift), x_edit.device
        )
        n_avg = int(n_avg)
        if n_avg < 1:
            raise ValueError("n_avg must be >= 1")

        with accelerator.autocast_for(self.device):
            for index, timestep_value in enumerate(timesteps):
                timestep = torch.tensor(
                    [timestep_value] * x_edit.shape[0], device=x_edit.device
                )
                if (
                    timestep_value > cfg_interval[0]
                    and timestep_value <= cfg_interval[1]
                ):
                    text_scale, img_scale = cfg_text_scale, cfg_img_scale
                else:
                    text_scale, img_scale = 1.0, 1.0

                branch = flowedit_time_branch(
                    float(timestep_value), float(n_min), float(n_max)
                )
                if branch == "src":
                    velocity = self._forward_flow_side(
                        x_t=x_edit,
                        timestep=timestep,
                        ctx=src_ctx,
                        cfg_text_ctx=src_cfg_text_ctx,
                        cfg_img_ctx=src_cfg_img_ctx,
                        generation_input=src_in,
                        cfg_text_input=src_ct,
                        cfg_img_input=src_ci,
                        cfg_text_scale=text_scale,
                        cfg_img_scale=img_scale,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    x_edit = self.model.image_euler_step(x_edit, velocity, dts[index])
                    continue
                if branch == "tar":
                    velocity = self._forward_flow_side(
                        x_t=x_edit,
                        timestep=timestep,
                        ctx=tar_ctx,
                        cfg_text_ctx=tar_cfg_text_ctx,
                        cfg_img_ctx=tar_cfg_img_ctx,
                        generation_input=tar_in,
                        cfg_text_input=tar_ct,
                        cfg_img_input=tar_ci,
                        cfg_text_scale=text_scale,
                        cfg_img_scale=img_scale,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    x_edit = self.model.image_euler_step(x_edit, velocity, dts[index])
                    continue

                delta = torch.zeros_like(x_edit)
                for _ in range(n_avg):
                    noise = torch.randn_like(x_src0)
                    z_src = (1.0 - float(timestep_value)) * x_src0 + float(
                        timestep_value
                    ) * noise
                    z_tar = x_edit + z_src - x_src0
                    v_src = self._forward_flow_side(
                        x_t=z_src,
                        timestep=timestep,
                        ctx=src_ctx,
                        cfg_text_ctx=src_cfg_text_ctx,
                        cfg_img_ctx=src_cfg_img_ctx,
                        generation_input=src_in,
                        cfg_text_input=src_ct,
                        cfg_img_input=src_ci,
                        cfg_text_scale=text_scale,
                        cfg_img_scale=img_scale,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    v_tar = self._forward_flow_side(
                        x_t=z_tar,
                        timestep=timestep,
                        ctx=tar_ctx,
                        cfg_text_ctx=tar_cfg_text_ctx,
                        cfg_img_ctx=tar_cfg_img_ctx,
                        generation_input=tar_in,
                        cfg_text_input=tar_ct,
                        cfg_img_input=tar_ci,
                        cfg_text_scale=text_scale,
                        cfg_img_scale=img_scale,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    delta = delta + (v_tar - v_src)
                x_edit = self.model.image_euler_step(
                    x_edit, delta / n_avg, dts[index]
                )

        return self.decode_image(x_edit, image_shape)

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )
        generation_input = _move_to_device(generation_input, self.device)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        init_noise: Optional[torch.Tensor] = None,
        return_latent: bool = False,
        remove_old_prompt: Optional[bool] = None,
        return_loop_diagnostics: bool = False,
        dynamic_prompt_alpha: Optional[float] = None,
        dynamic_prompt_body_start: int = 12,
        dynamic_prompt_body_end: int = 20,
        dynamic_prompt_step_fraction: float = 0.35,
        dynamic_prompt_delta_mode: str = "dynamic",
    ) -> List[Union[str, Image.Image]]:
        """Official interleaved entry point.

        ``init_noise`` lets a caller pin the flow's initial noise so several
        conditions can be compared on the same trajectory; the produced latent
        is exposed on ``self.last_latent`` when ``return_latent=True``.
        """

        output_list = []
        if remove_old_prompt is None:
            remove_old_prompt = True
        input_lists = filter_old_prompt(input_lists, bool(remove_old_prompt))
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with accelerator.autocast_for(self.device):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                gen_result = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                    init_noise=init_noise,
                    return_latent=bool(return_latent),
                    return_loop_diagnostics=bool(return_loop_diagnostics),
                    dynamic_prompt_alpha=dynamic_prompt_alpha,
                    dynamic_prompt_body_start=int(dynamic_prompt_body_start),
                    dynamic_prompt_body_end=int(dynamic_prompt_body_end),
                    dynamic_prompt_step_fraction=float(
                        dynamic_prompt_step_fraction
                    ),
                    dynamic_prompt_delta_mode=str(dynamic_prompt_delta_mode),
                )
                if return_latent:
                    img, self.last_latent = gen_result
                else:
                    img = gen_result
                    self.last_latent = None

                output_list.append(img)

        return output_list

    def __call__(
        self, image: Optional[Image.Image] = None, text: Optional[str] = None, **kargs
    ) -> Dict[str, Any]:
        output_dict = {"image": None, "text": None}

        if image is None and text is None:
            print("Please provide at least one input: either an image or text.")
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict["image"] = i
            elif isinstance(i, str):
                output_dict["text"] = i
        return output_dict
