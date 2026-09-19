"""BAGEL-native recurrent loop and loop-gated attention LoRA."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


UND_Q_PROJECTIONS: Tuple[str, ...] = ("q_proj",)
GEN_Q_PROJECTIONS: Tuple[str, ...] = ("q_proj_moe_gen",)
GEN_O_PROJECTIONS: Tuple[str, ...] = ("o_proj_moe_gen",)
K_V_PROJECTIONS: Tuple[str, ...] = (
    "k_proj",
    "v_proj",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
)
GENERATION_ATTENTION_PROJECTIONS: Tuple[str, ...] = (
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
)
TEXT_ATTENTION_PROJECTIONS: Tuple[str, ...] = (
    "k_proj",
    "v_proj",
)


def loop_lora_projections(
    *,
    gen_attention_o_lora: bool = False,
    k_v_lora: bool = False,
) -> Tuple[str, ...]:
    names: Tuple[str, ...] = UND_Q_PROJECTIONS + GEN_Q_PROJECTIONS
    if gen_attention_o_lora:
        names = names + GEN_O_PROJECTIONS
    if k_v_lora:
        names = names + K_V_PROJECTIONS
    return names


class LoopLoRALinear(nn.Module):
    """A LoRA linear whose residual is active only in an allowed loop mode.

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
        read_enabled: bool = True,
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
        self.read_enabled = bool(read_enabled)
        self.loop_mode = "off"

    @property
    def loop_enabled(self) -> bool:
        """Backward-compatible view used by older checkpoints and tests."""

        return self.loop_mode != "off"

    def set_loop_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in ("off", "read", "write"):
            raise ValueError("loop mode must be 'off', 'read', or 'write'")
        self.loop_mode = mode

    def set_loop_enabled(self, enabled: bool) -> None:
        self.set_loop_mode("write" if enabled else "off")

    def adapter_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return self.lora_B(self.lora_A(self.dropout(inputs.float())))

    def forward_rows(
        self,
        inputs: torch.Tensor,
        row_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.base_layer(inputs)
        if self.loop_mode == "off":
            return output
        if self.loop_mode == "read" and (not self.read_enabled or row_mask is None):
            return output
        if self.loop_mode not in ("read", "write"):
            raise ValueError("loop mode must be 'off', 'read', or 'write'")
        if row_mask is None:
            residual = self.adapter_residual(inputs)
            return output + residual.to(output.dtype) * self.scaling
        selected = row_mask.to(device=inputs.device, dtype=torch.bool)
        if selected.shape[0] != inputs.shape[0]:
            raise ValueError(
                "row_mask length must match the packed input: "
                f"{int(selected.shape[0])} != {int(inputs.shape[0])}"
            )
        result = output.clone()
        if bool(selected.any()):
            residual = self.adapter_residual(inputs[selected])
            result[selected] = result[selected] + residual.to(result.dtype) * self.scaling
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_rows(inputs, row_mask=None)


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
    gen_attention_o_lora: bool = False,
    k_v_lora: bool = False,
) -> None:
    """Inject loop-gated LoRA into selected attention routes of a body."""

    layers = model.language_model.model.layers
    start, end = int(start_layer), int(end_layer)
    if not 0 <= start < end <= len(layers):
        raise ValueError(
            f"invalid loop layers [{start}, {end}) for {len(layers)} layers"
        )
    projections = loop_lora_projections(
        gen_attention_o_lora=bool(gen_attention_o_lora),
        k_v_lora=bool(k_v_lora),
    )
    for layer_index in range(start, end):
        layer = _actual_layer(layers[layer_index])
        attention = layer.self_attn
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
                    read_enabled=projection in UND_Q_PROJECTIONS,
                ),
            )


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def loop_trainable_names(
    model: nn.Module,
    *,
    start_layer: int,
    end_layer: int,
    gen_attention_o_lora: bool = False,
    k_v_lora: bool = False,
) -> List[str]:
    """Fail closed unless every trainable tensor is an allowed loop LoRA."""

    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not names:
        raise RuntimeError("loop policy produced no trainable parameters")
    allowed_projections = loop_lora_projections(
        gen_attention_o_lora=bool(gen_attention_o_lora),
        k_v_lora=bool(k_v_lora),
    )
    invalid = []
    for name in names:
        match = _LAYER_PATTERN.search(name)
        layer_index = int(match.group(1)) if match else -1
        is_lora = ".lora_A." in name or ".lora_B." in name
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
            "trainable parameter escaped loop Q-attention LoRA allowlist: "
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


__all__ = [
    "GENERATION_ATTENTION_PROJECTIONS",
    "GEN_O_PROJECTIONS",
    "GEN_Q_PROJECTIONS",
    "K_V_PROJECTIONS",
    "LoopLoRALinear",
    "TEXT_ATTENTION_PROJECTIONS",
    "UND_Q_PROJECTIONS",
    "inject_loop_lora",
    "load_loop_adapter_state_dict",
    "loop_adapter_state_dict",
    "loop_lora_projections",
    "loop_trainable_names",
    "move_batch_to_device",
]
