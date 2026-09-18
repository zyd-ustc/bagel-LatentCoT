"""BAGEL-native recurrent loop and loop-gated attention LoRA."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


GENERATION_ATTENTION_PROJECTIONS: Tuple[str, ...] = (
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
)

# In BAGEL generation mode, text tokens use the shared/understanding branch
# while VAE tokens use the generation expert.  Updating only text K/V during
# the gated loop pass lets the frozen generation prior read a trainable
# semantic representation without changing the depth-one path.
TEXT_ATTENTION_PROJECTIONS: Tuple[str, ...] = (
    "k_proj",
    "v_proj",
)


class LoopLoRALinear(nn.Module):
    """A LoRA linear whose residual is active only in an explicit loop pass.

    The pretrained linear is retained as ``base_layer`` and is always used.
    Unlike a global PEFT adapter, this module cannot silently alter BAGEL's
    depth-1 path: ``loop_enabled`` defaults to false and is toggled only by the
    segmented loop executor.
    """

    is_loop_lora = True

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoopLoRALinear requires an nn.Linear base layer")
        if int(rank) <= 0 or int(alpha) <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        # Keep trainable adapter/master weights in FP32.  BAGEL runs in BF16,
        # where the 1e-6-scale RL updates can otherwise quantize away.
        self.lora_A.to(device=base_layer.weight.device, dtype=torch.float32)
        self.lora_B.to(device=base_layer.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.loop_enabled = False

    def set_loop_enabled(self, enabled: bool) -> None:
        self.loop_enabled = bool(enabled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(inputs)
        if not self.loop_enabled:
            return output
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            residual = self.lora_B(self.lora_A(self.dropout(inputs.float())))
        return output + residual.to(output.dtype) * self.scaling


def _actual_layer(layer: nn.Module) -> nn.Module:
    return getattr(layer, "_checkpoint_wrapped_module", layer)


def inject_loop_lora(
    model: nn.Module,
    *,
    start_layer: int,
    end_layer: int,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    include_text_kv: bool = False,
) -> None:
    """Inject loop-gated LoRA into selected attention routes of a body."""

    layers = model.language_model.model.layers
    start, end = int(start_layer), int(end_layer)
    if not 0 <= start < end <= len(layers):
        raise ValueError(
            f"invalid loop layers [{start}, {end}) for {len(layers)} layers"
        )
    for layer_index in range(start, end):
        layer = _actual_layer(layers[layer_index])
        attention = layer.self_attn
        projections = GENERATION_ATTENTION_PROJECTIONS + (
            TEXT_ATTENTION_PROJECTIONS if include_text_kv else ()
        )
        for projection in projections:
            current = getattr(attention, projection, None)
            if current is None:
                raise RuntimeError(
                    f"layer {layer_index} has no generation projection {projection}"
                )
            if isinstance(current, LoopLoRALinear):
                raise RuntimeError(
                    f"loop LoRA already injected at layer {layer_index}.{projection}"
                )
            setattr(
                attention,
                projection,
                LoopLoRALinear(
                    current,
                    rank=int(rank),
                    alpha=int(alpha),
                    dropout=float(dropout),
                ),
            )


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def loop_trainable_names(
    model: nn.Module,
    *,
    start_layer: int,
    end_layer: int,
    include_text_kv: bool = False,
) -> List[str]:
    """Fail closed unless every trainable tensor is an allowed loop LoRA."""

    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not names:
        raise RuntimeError("loop policy produced no trainable parameters")
    invalid = []
    for name in names:
        match = _LAYER_PATTERN.search(name)
        layer_index = int(match.group(1)) if match else -1
        is_lora = ".lora_A." in name or ".lora_B." in name
        allowed_projections = GENERATION_ATTENTION_PROJECTIONS + (
            TEXT_ATTENTION_PROJECTIONS if include_text_kv else ()
        )
        is_allowed_attention = any(
            f".{projection}." in name for projection in allowed_projections
        )
        if not (
            is_lora
            and is_allowed_attention
            and int(start_layer) <= layer_index < int(end_layer)
        ):
            invalid.append(name)
    if invalid:
        raise RuntimeError(
            "trainable parameter escaped loop GEN-attention LoRA allowlist: "
            + ", ".join(invalid[:8])
        )
    return names


def loop_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only loop LoRA tensors with names relative to ``model``."""

    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    if not state:
        raise RuntimeError("model contains no loop LoRA tensors")
    return state


def load_loop_adapter_state_dict(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    allow_missing_projections: Sequence[str] = (),
) -> List[str]:
    """Strictly load a loop adapter without accepting base-model tensors."""

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    incoming = set(state)
    expected = set(current)
    missing = sorted(expected - incoming)
    unexpected = sorted(incoming - expected)
    invalid_missing = [
        name
        for name in missing
        if not any(
            f".{projection}." in name for projection in allow_missing_projections
        )
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "loop adapter key mismatch: "
            f"missing={invalid_missing[:8]}, unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            if name not in state:
                continue
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"loop adapter shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return missing


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _device_dict(payload: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in payload.items()
    }


def _encode_clean_latents(
    model: nn.Module,
    vae_model: nn.Module,
    batch: Dict[str, Any],
) -> torch.Tensor:
    images: torch.Tensor = batch["padded_vae_images"]
    shapes: Sequence[Tuple[int, int]] = batch["patchified_vae_latent_shapes"]
    vae_dtype = next(vae_model.parameters()).dtype
    with torch.no_grad():
        encoded = vae_model.encode(images.to(dtype=vae_dtype))
    if encoded.ndim != 4 or int(encoded.shape[0]) != len(shapes):
        raise RuntimeError(
            "VAE output must be [B,C,H,W] and match latent shapes; "
            f"got {tuple(encoded.shape)} for {len(shapes)} samples"
        )

    patch = int(model.latent_patch_size)
    channels = int(model.latent_channel)
    chunks = []
    for latent, (height, width) in zip(encoded, shapes):
        height, width = int(height), int(width)
        latent = latent[:, : height * patch, : width * patch]
        latent = latent.reshape(channels, height, patch, width, patch)
        latent = torch.einsum("chpwq->hwpqc", latent)
        chunks.append(latent.reshape(-1, patch * patch * channels))
    return torch.cat(chunks, dim=0)


@dataclass(frozen=True)
class LoopFlowOutput:
    loss: torch.Tensor
    per_step_flow_losses: Tuple[torch.Tensor, ...]
    flow_token_win_rate: torch.Tensor
    prediction: torch.Tensor
    flow_target: torch.Tensor
    timestep: torch.Tensor
    state_rms: torch.Tensor


class BagelCrossStepFlowModule(nn.Module):
    """Local-flow warm-up for BAGEL's native semantic-state recurrence.

    Aligned with Looped Flows (arXiv 2609.11801): a rollout of the stateful
    denoiser over an ordered sequence of decreasing noise levels with shared
    noise and target. Exactly K pre-image text tokens carry the state through
    BAGEL's UND branch; GEN reads them either through native joint attention or
    through a K/V-only prefix ablation. Each step has a local flow loss and the
    incoming state is stop-gradiented, so no BPTT is required.
    """

    def __init__(
        self,
        bagel: nn.Module,
        *,
        tokenizer,
        token_ids,
        vae_model: nn.Module,
        loop_start_layer: int,
        loop_end_layer: int,
        loop_state_scale: float,
        loop_state_mode: str = "semantic_token",
        loop_state_tokens: int = 16,
        loop_state_token_id: Optional[int] = None,
        loop_draft_state_scale: float = 0.2,
        rollout_steps: int = 4,
        timestep_shift: float = 1.0,
        loop_state_timestep_threshold: Optional[float] = None,
        loop_residual_alpha: float = 1.0,
        cache_factory=None,
    ) -> None:
        super().__init__()
        if float(loop_state_scale) < 0.0:
            raise ValueError("loop_state_scale must be non-negative")
        if int(rollout_steps) < 1:
            raise ValueError("rollout_steps must be >= 1")
        state_mode = str(loop_state_mode).strip().lower()
        if state_mode not in {"semantic_token", "kv_prefix"}:
            raise ValueError(
                "loop_state_mode must be 'semantic_token' or 'kv_prefix'"
            )
        if int(loop_state_tokens) < 1:
            raise ValueError("loop_state_tokens must be >= 1")
        if float(timestep_shift) <= 0.0:
            raise ValueError("timestep_shift must be positive")
        if float(loop_draft_state_scale) < 0.0:
            raise ValueError("loop_draft_state_scale must be non-negative")
        if loop_state_timestep_threshold is not None and not (
            0.0 <= float(loop_state_timestep_threshold) <= 1.0
        ):
            raise ValueError("loop state timestep threshold must be in [0, 1]")
        self.bagel = bagel
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.vae_model = vae_model
        self.loop_start_layer = int(loop_start_layer)
        self.loop_end_layer = int(loop_end_layer)
        self.loop_state_scale = float(loop_state_scale)
        self.loop_state_mode = state_mode
        self.loop_state_tokens = int(loop_state_tokens)
        self.loop_state_token_id = (
            None if loop_state_token_id is None else int(loop_state_token_id)
        )
        self.loop_draft_state_scale = float(loop_draft_state_scale)
        self.rollout_steps = int(rollout_steps)
        self.timestep_shift = float(timestep_shift)
        self.loop_state_timestep_threshold = (
            None
            if loop_state_timestep_threshold is None
            else float(loop_state_timestep_threshold)
        )
        self.loop_residual_alpha = float(loop_residual_alpha)
        self.cache_factory = cache_factory

    def _shift_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        shift = self.timestep_shift
        return shift * timestep / (1.0 + (shift - 1.0) * timestep)

    def _sample_rollout_timesteps(self, device: torch.device) -> torch.Tensor:
        """Ordered decreasing shifted times inside the state-active region."""

        floor = (
            0.05
            if self.loop_state_timestep_threshold is None
            else float(self.loop_state_timestep_threshold)
        )
        if not 0.0 < floor < 1.0:
            floor = 0.05
        raw = torch.rand(self.rollout_steps, device=device)
        shifted = self._shift_timestep(floor + (1.0 - floor) * raw)
        return shifted.sort(descending=True).values

    @property
    def _special_ids(self) -> Dict[str, int]:
        return {
            "bos_token_id": int(self.token_ids.im_start),
            "eos_token_id": int(self.token_ids.im_end),
            "start_of_image": int(self.token_ids.vision_start),
            "end_of_image": int(self.token_ids.vision_end),
        }

    def _velocity(self, layout, x_t, timestep, cache, *, loop_state_in=None):
        return self.bagel._forward_flow(
            x_t=x_t,
            timestep=timestep,
            packed_vae_token_indexes=layout["packed_vae_token_indexes"],
            packed_vae_position_ids=layout["packed_vae_position_ids"],
            packed_text_ids=layout["packed_text_ids"],
            packed_text_indexes=layout["packed_text_indexes"],
            packed_indexes=layout["packed_indexes"],
            packed_position_ids=layout["packed_position_ids"],
            packed_seqlens=layout["packed_seqlens"],
            key_values_lens=layout["key_values_lens"],
            past_key_values=cache,
            packed_key_value_indexes=layout["packed_key_value_indexes"],
            loop_start_layer=self.loop_start_layer,
            loop_end_layer=self.loop_end_layer,
            loop_state_in=loop_state_in,
            loop_state_scale=self.loop_state_scale,
            loop_state_mode=self.loop_state_mode,
            loop_residual_alpha=self.loop_residual_alpha,
            packed_boundary_token_indexes=layout["packed_boundary_token_indexes"],
            packed_loop_semantic_token_indexes=layout[
                "packed_loop_semantic_token_indexes"
            ],
            return_loop_state=True,
        )

    def forward(self, batch: Dict[str, Any]) -> LoopFlowOutput:
        model = self.bagel
        prompts = [str(prompt).strip() for prompt in batch["prompts"]]
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("loop batch requires non-empty prompts")
        if not bool(getattr(model.config, "visual_gen", False)) or not bool(
            model.use_moe
        ):
            raise RuntimeError("loop flow requires BAGEL visual_gen with MoT experts")

        device = next(model.parameters()).device
        if self.cache_factory is None:
            from .modeling.bagel.qwen2_navit import NaiveCache

            cache_factory = NaiveCache
        else:
            cache_factory = self.cache_factory
        cache = cache_factory(model.config.llm_config.num_hidden_layers)
        prompt_layout, prompt_lens, prompt_rope = model.prepare_prompts(
            curr_kvlens=[0] * len(prompts),
            curr_rope=[0] * len(prompts),
            prompts=prompts,
            tokenizer=self.tokenizer,
            new_token_ids=self._special_ids,
        )
        prompt_layout = _device_dict(prompt_layout, device)
        with torch.no_grad():
            cache = model.forward_cache_update_text(cache, **prompt_layout)

        image_sizes = [
            (
                int(height) * int(model.latent_downsample),
                int(width) * int(model.latent_downsample),
            )
            for height, width in batch["patchified_vae_latent_shapes"]
        ]
        state_token_id = (
            int(self.token_ids.im_start)
            if self.loop_state_token_id is None
            else self.loop_state_token_id
        )
        semantic_text_ids = [
            (state_token_id,) * self.loop_state_tokens for _ in prompts
        ]
        layout = model.prepare_vae_latent(
            prompt_lens,
            prompt_rope,
            image_sizes,
            self._special_ids,
            semantic_text_ids=semantic_text_ids,
        )
        layout = _device_dict(layout, device)
        clean = _encode_clean_latents(model, self.vae_model, batch).to(device)

        noise = batch.get("noise")
        noise = torch.randn_like(clean) if noise is None else noise.to(device=device)
        if noise.shape != clean.shape:
            raise ValueError("noise and clean latent shapes differ")
        target = noise - clean

        with torch.no_grad():
            rollout_t = self._sample_rollout_timesteps(device)
        x_t = (1.0 - rollout_t[0]) * clean + rollout_t[0] * noise

        loop_state = None
        prediction = None
        per_step_losses = []
        for step_index, t_value in enumerate(rollout_t.tolist()):
            timestep = torch.full(
                (int(x_t.shape[0]),),
                float(t_value),
                dtype=x_t.dtype,
                device=device,
            )
            prediction, loop_state = self._velocity(
                layout,
                x_t,
                timestep,
                cache,
                loop_state_in=loop_state,
            )
            with torch.no_grad():
                x0_hat = x_t - timestep * prediction.detach()
                loop_state = model.fuse_loop_draft_state(
                    loop_state,
                    x0_hat,
                    layout["packed_vae_seqlens"],
                    state_tokens_per_sample=self.loop_state_tokens,
                    residual_scale=self.loop_draft_state_scale,
                )
            if step_index == 0 and torch.is_grad_enabled():
                if not prediction.requires_grad:
                    raise RuntimeError(
                        "loop flow prediction is detached; BAGEL._forward_flow "
                        "must remain differentiable for loop-adapter training"
                    )
            per_step = (prediction.float() - target.float()).square().mean(-1)
            per_step_losses.append(per_step.mean())
            if step_index + 1 < len(rollout_t):
                next_t = float(rollout_t[step_index + 1].item())
                # Detached rollout: gradients flow only through each step's
                # local loss (Looped Flows Eq. 9 with sg(z) between steps).
                with torch.no_grad():
                    x_t = x_t - prediction * (float(t_value) - next_t)

        loss = torch.stack(per_step_losses).mean()
        if prediction is None:
            raise RuntimeError("rollout produced no prediction; check rollout_steps")
        final_prediction = prediction
        state_rms = (
            loop_state.detach().float().square().mean().sqrt()
            if loop_state is not None
            else final_prediction.new_zeros(())
        )
        return LoopFlowOutput(
            loss=loss,
            per_step_flow_losses=tuple(per_step_losses),
            flow_token_win_rate=final_prediction.new_zeros(()),
            prediction=final_prediction,
            flow_target=target,
            timestep=rollout_t.detach(),
            state_rms=state_rms,
        )


__all__ = [
    "BagelCrossStepFlowModule",
    "GENERATION_ATTENTION_PROJECTIONS",
    "TEXT_ATTENTION_PROJECTIONS",
    "LoopFlowOutput",
    "LoopLoRALinear",
    "inject_loop_lora",
    "load_loop_adapter_state_dict",
    "loop_adapter_state_dict",
    "loop_trainable_names",
    "move_batch_to_device",
]
