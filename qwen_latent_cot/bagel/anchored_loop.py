"""One recurrent algorithm for inference and direct-flow post-training.

R counts EXTRA body executions. The original body always runs once without
workspace tokens, gates, or adapters. No state survives a diffusion timestep.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LoopConfig:
    enable_t2i_loop: bool = False
    loop_mode: str = "gen_memory_anchored"
    loop_start_layer: int = 16
    loop_end_layer: int = 24
    loop_depth: int = 2
    memory_slots: int = 8
    freeze_prompt_kv_in_loop: bool = True
    reentry_adapter_type: str = "low_rank"
    reentry_rank: int | None = None
    fixed_reentry_scale: float = 0.05
    loop_gate_init: float = 0.02
    loop_output_alpha_init: float = 0.0
    loop_deep_supervision: bool = True
    loop_ds_weight: float = 0.5
    log_loop_stats: bool = False
    memory_control: str = "correct"

    def __post_init__(self):
        if self.loop_mode not in {
            "memory_only",
            "gen_only",
            "gen_memory_anchored",
            "direct_native",
        }:
            raise ValueError("unsupported loop_mode")
        if not 0 <= self.loop_start_layer < self.loop_end_layer:
            raise ValueError("require 0 <= loop_start_layer < loop_end_layer")
        if self.loop_depth < 0 or self.memory_slots < 0:
            raise ValueError("loop_depth and memory_slots must be nonnegative")
        if self.loop_mode == "gen_only" and self.memory_slots != 0:
            raise ValueError("gen_only requires memory_slots=0")
        if self.loop_mode == "memory_only" and self.memory_slots == 0:
            raise ValueError("memory_only requires memory_slots>0")
        if not self.freeze_prompt_kv_in_loop:
            raise ValueError("prompt KV must remain frozen")
        if self.reentry_adapter_type not in {"low_rank", "fixed"}:
            raise ValueError("reentry_adapter_type must be low_rank or fixed")
        if self.reentry_rank is not None and self.reentry_rank < 1:
            raise ValueError("reentry_rank must be positive")
        if not 0 <= self.loop_gate_init < 1:
            raise ValueError("loop_gate_init must lie in [0, 1)")
        if self.loop_ds_weight < 0:
            raise ValueError("loop_ds_weight must be nonnegative")
        if self.memory_control not in {"correct", "zero", "frozen", "shuffled"}:
            raise ValueError("unsupported memory_control")

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return normed.to(x.dtype) * self.weight


class ReentryAdapter(nn.Module):
    def __init__(self, width, rank, use_memory):
        super().__init__()
        self.norm = RMSNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.memory_projection = (
            nn.Linear(width, width, bias=False) if use_memory else None
        )
        if self.memory_projection is not None:
            nn.init.zeros_(self.memory_projection.weight)

    def forward(self, delta, memory=None):
        correction = self.up(self.down(self.norm(delta)))
        if memory is not None and self.memory_projection is not None:
            # delta: [batch, image tokens, hidden]; M: [batch, K, hidden].
            correction = correction + self.memory_projection(
                memory.mean(1, keepdim=True)
            )
        return correction


class LoopModules(nn.Module):
    def __init__(self, width: int, config: LoopConfig):
        super().__init__()
        self.config = config
        self.reentry = ReentryAdapter(
            width, config.reentry_rank or max(1, width // 8), config.memory_slots > 0
        )
        depth = config.loop_end_layer - config.loop_start_layer
        self.gate_logits = nn.Parameter(
            torch.full(
                (depth,),
                math.log(config.loop_gate_init / (1 - config.loop_gate_init))
                if config.loop_gate_init
                else -math.inf,
            )
        )
        self.output_alpha = nn.Parameter(
            torch.full(
                (config.loop_depth,),
                float(config.loop_output_alpha_init),
                dtype=torch.float32,
            )
        )
        if config.memory_slots:
            self.memory_init = nn.Parameter(torch.empty(config.memory_slots, width))
            nn.init.normal_(self.memory_init, std=0.01)
        else:
            self.register_parameter("memory_init", None)

    def initial_memory(self, batch, slots=None):
        slots = self.config.memory_slots if slots is None else slots
        if slots == 0:
            return None
        if self.memory_init is None or slots > self.memory_init.shape[0]:
            raise ValueError("requested memory slots exceed allocated workspace")
        return self.memory_init[:slots].unsqueeze(0).expand(batch, -1, -1).clone()

    def entry(self, anchor, delta, memory, config):
        if config.loop_mode == "memory_only":
            return anchor.clone()
        if config.loop_mode == "direct_native":
            # The runner replaces this with the previous body endpoint.
            raise RuntimeError("direct_native entry must use previous endpoint")
        if config.reentry_adapter_type == "fixed":
            correction = config.fixed_reentry_scale * delta
        else:
            correction = self.reentry(delta, memory)
        return anchor + correction

    def gate(self, layer_offset, before, after):
        return before + self.gate_logits[layer_offset].sigmoid().to(before.dtype) * (
            after - before
        )


@dataclass
class AnchorState:
    gen_entry: torch.Tensor
    gen_base: torch.Tensor


@dataclass
class LoopResult:
    velocity: torch.Tensor
    base_velocity: torch.Tensor
    velocities: list[torch.Tensor]
    stats: list[dict]


def ratio(delta, reference):
    return float(
        delta.detach().float().norm()
        / reference.detach().float().norm().clamp_min(1e-12)
    )


def run_anchored_loop(
    anchor: AnchorState,
    modules: LoopModules,
    config: LoopConfig,
    body: Callable,
    readout: Callable,
) -> LoopResult:
    """body(entry, memory, modules) returns GEN and M at the SAME endpoint.

    Callers supply the native base endpoint and enforce read-only condition
    caches in both callbacks. Recurrent states are local to this invocation.
    """
    base_velocity = readout(anchor.gen_base)
    if (
        not config.enable_t2i_loop
        or config.loop_depth == 0
        or not bool(torch.count_nonzero(modules.gate_logits.detach().sigmoid()).item())
    ):
        return LoopResult(base_velocity, base_velocity, [], [])
    if config.loop_depth > modules.output_alpha.numel():
        raise ValueError(
            "requested depth exceeds allocated output_alpha; allocate maximum curriculum depth"
        )
    memory = modules.initial_memory(anchor.gen_entry.shape[0], config.memory_slots)
    initial_memory = memory.clone() if memory is not None else None
    delta = torch.zeros_like(anchor.gen_base)
    previous_gen = anchor.gen_base
    velocities, stats = [], []
    previous_velocity_delta = None
    for r in range(config.loop_depth):
        if memory is not None:
            if config.memory_control == "zero":
                memory = torch.zeros_like(memory)
            elif config.memory_control == "frozen":
                memory = initial_memory.clone()
            elif config.memory_control == "shuffled":
                if memory.shape[0] < 2:
                    raise ValueError("shuffled-across-sample memory requires batch>=2")
                memory = memory.roll(1, dims=0)
        entry = (
            previous_gen
            if config.loop_mode == "direct_native"
            else modules.entry(anchor.gen_entry, delta, memory, config)
        )
        current_gen, memory, layer_stats = body(entry, memory, modules)
        delta = current_gen - anchor.gen_base
        previous_gen = current_gen
        merged = anchor.gen_base + modules.output_alpha[r].to(delta.dtype) * delta
        velocity = readout(merged)
        velocities.append(velocity)
        if config.log_loop_stats:
            dv = velocity - base_velocity
            item = {
                "round": r + 1,
                "velocity_delta_ratio": ratio(dv, base_velocity),
                "gen_delta_ratio": ratio(delta, anchor.gen_base),
                "entry_delta_ratio": ratio(entry - anchor.gen_entry, anchor.gen_entry),
                "gen_norm": float(current_gen.detach().float().norm()),
                "memory_norm": float(memory.detach().float().norm())
                if memory is not None
                else 0.0,
                "alpha": float(modules.output_alpha[r].detach()),
                "layers": layer_stats,
            }
            if previous_velocity_delta is not None:
                item["velocity_direction_cosine"] = float(
                    F.cosine_similarity(
                        dv.detach().float().flatten(),
                        previous_velocity_delta.flatten(),
                        dim=0,
                    )
                )
            previous_velocity_delta = dv.detach().float()
            stats.append(item)
    return LoopResult(velocities[-1], base_velocity, velocities, stats)


def direct_flow_loss(result: LoopResult, target: torch.Tensor, config: LoopConfig):
    """Final flow MSE + lambda * mean intermediate MSE; never distillation."""
    predictions = result.velocities or [result.velocity]
    losses = [F.mse_loss(v.float(), target.float()) for v in predictions]
    total = losses[-1]
    if config.loop_deep_supervision and len(losses) > 1:
        total = total + config.loop_ds_weight * torch.stack(losses[:-1]).mean()
    return total


def configure_stage1(model: nn.Module):
    """Freeze every native tensor. Open only the newly added loop modules."""
    model.requires_grad_(False)
    model.t2i_loop.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]
