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
    runtime_loop_depth: int = 2
    allocated_max_loop_depth: int = 4
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
            "legacy_memory_only",
            "gen_only",
            "gen_memory_anchored",
            "direct_native_gen_only",
            "direct_native_memory",
        }:
            raise ValueError("unsupported loop_mode")
        if not 0 <= self.loop_start_layer < self.loop_end_layer:
            raise ValueError("require 0 <= loop_start_layer < loop_end_layer")
        if not 0 <= self.runtime_loop_depth <= self.allocated_max_loop_depth:
            raise ValueError(
                "runtime_loop_depth must lie within allocated_max_loop_depth"
            )
        if self.allocated_max_loop_depth < 1:
            raise ValueError("allocated_max_loop_depth must be positive")
        if self.runtime_loop_depth < 0 or self.memory_slots < 0:
            raise ValueError("runtime_loop_depth and memory_slots must be nonnegative")
        if (
            self.loop_mode in {"gen_only", "direct_native_gen_only"}
            and self.memory_slots != 0
        ):
            raise ValueError("gen_only/direct_native_gen_only requires memory_slots=0")
        if (
            self.loop_mode
            in {"memory_only", "legacy_memory_only", "direct_native_memory"}
            and self.memory_slots == 0
        ):
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
        if self.memory_control not in {"correct", "zero", "frozen", "shuffled", "no_read"}:
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
        self.up = nn.Linear(rank, width, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.memory_projection = (
            nn.Linear(width, width, bias=False) if use_memory else None
        )
        if self.memory_projection is not None:
            nn.init.zeros_(self.memory_projection.weight)

    def forward(self, delta, memory=None, gen_counts=None):
        # Small post-training updates need the module's parameter precision,
        # rather than the frozen backbone's BF16 autocast precision.
        with torch.autocast(device_type=delta.device.type, enabled=False):
            delta = delta.to(self.norm.weight.dtype)
            correction = self.up(self.down(self.norm(delta)))
            if memory is not None and self.memory_projection is not None:
                projected = self.memory_projection(
                    memory.to(self.memory_projection.weight.dtype).mean(1)
                )
                if delta.ndim == 2:
                    if gen_counts is None:
                        raise ValueError(
                            "packed GEN memory conditioning requires gen_counts"
                        )
                    projected = projected.repeat_interleave(gen_counts, dim=0)
                else:
                    projected = projected.unsqueeze(1)
                correction = correction + projected
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
                (1,),
                float(config.loop_output_alpha_init),
                dtype=torch.float32,
            )
        )
        if config.memory_slots:
            self.memory_init = nn.Parameter(torch.empty(config.memory_slots, width))
            nn.init.normal_(self.memory_init, std=0.01)
        else:
            self.register_parameter("memory_init", None)

    @torch.no_grad()
    def initialize_memory_from_boundaries(self, embedding_weight, token_ids, *, seed=0):
        """Parent initialization, including dtype/order and independent slot noise."""
        if self.memory_init is None:
            return
        base = embedding_weight[token_ids].float().mean(dim=0)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            self.memory_init.shape, generator=generator, dtype=torch.float32
        )
        noise = noise.to(device=self.memory_init.device, dtype=self.memory_init.dtype)
        noise = 1e-4 * noise
        self.memory_init.copy_(
            base.to(self.memory_init).unsqueeze(0).expand_as(self.memory_init) + noise
        )

    def initial_memory(self, batch, slots=None):
        slots = self.config.memory_slots if slots is None else slots
        if slots == 0:
            return None
        if self.memory_init is None or slots > self.memory_init.shape[0]:
            raise ValueError("requested memory slots exceed allocated workspace")
        return self.memory_init[:slots].unsqueeze(0).expand(batch, -1, -1).clone()

    def entry(self, anchor, delta, memory, config, gen_counts=None):
        if config.loop_mode == "memory_only":
            return anchor.clone()
        if config.loop_mode.startswith("direct_native_"):
            # The runner replaces this with the previous body endpoint.
            raise RuntimeError("direct_native entry must use previous endpoint")
        if config.reentry_adapter_type == "fixed":
            correction = config.fixed_reentry_scale * delta
        else:
            correction = self.reentry(delta, memory, gen_counts)
        return anchor.float() + correction.float()

    def gate(self, layer_offset, reference_gen, loop_gen, incoming_delta):
        """Carry existing GEN correction; gate only this layer's extra write.

        Let d be the incoming correction relative to the native layer input.
        The extra write is (loop_output - native_output) - d. Multiplying the
        entire output difference by g would multiply d by every layer's gate.
        """
        with torch.autocast(device_type=loop_gen.device.type, enabled=False):
            reference = reference_gen.float()
            delta = incoming_delta.float()
            extra_write = loop_gen.float() - reference - delta
            gate = self.gate_logits[layer_offset].float().sigmoid()
            return reference + delta + gate * extra_write


@dataclass
class AnchorState:
    gen_entry: torch.Tensor
    gen_base: torch.Tensor
    gen_counts: torch.Tensor | None = None


@dataclass
class LoopResult:
    velocity: torch.Tensor
    base_velocity: torch.Tensor | None
    velocities: list[torch.Tensor]
    stats: list[dict]


def ratio(delta, reference):
    return float(
        delta.detach().float().norm()
        / reference.detach().float().norm().clamp_min(1e-12)
    )


def memory_slot_stats(memory):
    """Measure slots per sample; effective rank is of centered slot variation.

    Identical slots have centered rank 0. Samples are never pooled into one
    slot matrix, which would inflate rank by mixing unrelated images.
    """
    if memory is None or memory.numel() == 0:
        return {
            "effective_rank": 0.0,
            "mean_abs_pairwise_cosine": None,
            "slot_variation_norm": 0.0,
            "slot_std": 0.0,
            "mean_pairwise_cosine": None,
            "effective_rank_min": 0.0,
            "sigma1_ratio": 0.0,
        }
    slots = memory.detach().float().cpu()
    if slots.ndim == 2:
        slots = slots.unsqueeze(0)
    k = slots.shape[1]
    centered = slots - slots.mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum(dim=-1)
    share = singular / total.unsqueeze(-1).clamp_min(1e-12)
    ranks = torch.where(
        total > 1e-12,
        torch.exp(-(share * share.clamp_min(1e-12).log()).sum(-1)),
        torch.zeros_like(total),
    )
    normed = F.normalize(slots, dim=-1)
    gram = normed @ normed.transpose(-1, -2)
    mask = ~torch.eye(k, dtype=torch.bool)
    cosine = float(gram[:, mask].abs().mean()) if k > 1 else None
    signed_cosine = float(gram[:, mask].mean()) if k > 1 else None
    return {
        "effective_rank": float(ranks.mean()),
        "effective_rank_min": float(ranks.min()),
        "mean_abs_pairwise_cosine": cosine,
        "mean_pairwise_cosine": signed_cosine,
        "slot_std": float(slots.std(dim=1, unbiased=False).mean()),
        "slot_variation_norm": float(centered.norm()),
        "sigma1_ratio": float((singular[:, 0] / total.clamp_min(1e-12)).mean()),
    }


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
    need_readouts = config.log_loop_stats or (
        torch.is_grad_enabled() and config.loop_deep_supervision
    )
    base_velocity = readout(anchor.gen_base) if need_readouts else None
    direct = config.loop_mode.startswith("direct_native_")
    if (
        not config.enable_t2i_loop
        or config.runtime_loop_depth == 0
        or (
            not direct
            and not bool(
                torch.count_nonzero(modules.gate_logits.detach().sigmoid()).item()
            )
        )
    ):
        if base_velocity is None:
            base_velocity = readout(anchor.gen_base)
        return LoopResult(base_velocity, base_velocity, [], [])
    batch = (
        len(anchor.gen_counts)
        if anchor.gen_counts is not None
        else anchor.gen_entry.shape[0]
    )
    memory = modules.initial_memory(batch, config.memory_slots)
    initial_memory = memory.clone() if memory is not None else None
    delta = torch.zeros_like(anchor.gen_base)
    previous_gen = anchor.gen_base
    # Canonical writer follows the correct recurrence, never reader feedback.
    canonical_delta = torch.zeros_like(anchor.gen_base)
    canonical_previous_gen = anchor.gen_base
    permutation = None
    if memory is not None and config.memory_control == "shuffled":
        if memory.shape[0] < 2:
            raise ValueError("shuffled-across-sample memory requires batch>=2")
        permutation = torch.arange(memory.shape[0], device=memory.device).roll(1)
    velocities, stats = [], []
    previous_velocity_delta = None
    for r in range(config.runtime_loop_depth):
        if memory is not None:
            if config.memory_control == "zero":
                memory = torch.zeros_like(memory)
            elif config.memory_control == "frozen":
                memory = initial_memory.clone()
        previous_memory = (
            memory.clone() if memory is not None and config.log_loop_stats else None
        )
        read_memory = (
            None if config.memory_control == "no_read"
            else memory if permutation is None else memory[permutation]
        )
        entry = (
            previous_gen
            if direct
            else modules.entry(
                anchor.gen_entry, delta, read_memory, config, anchor.gen_counts
            )
        )
        separate_writer = permutation is not None or (
            memory is not None and config.memory_control == "no_read"
        )
        if not separate_writer:
            current_gen, memory, layer_stats = body(entry, memory, modules)
        else:
            canonical_entry = (
                canonical_previous_gen
                if direct
                else modules.entry(
                    anchor.gen_entry, canonical_delta, memory, config, anchor.gen_counts
                )
            )
            layer_memories = []
            canonical_gen, canonical_memory, _ = body(
                canonical_entry, memory, modules, capture_memory_reads=layer_memories
            )
            # Replace only depth-aligned memory inputs for the reader. Its
            # memory outputs are discarded, so permutation never compounds.
            reader_options = (
                dict(block_memory_reads=True)
                if config.memory_control == "no_read"
                else dict(
                    memory_read_overrides=[value[permutation] for value in layer_memories],
                    memory_reference_reads=layer_memories,
                )
            )
            current_gen, _, layer_stats = body(entry, memory, modules, **reader_options)
            memory = canonical_memory
            canonical_delta = canonical_gen - anchor.gen_base
            canonical_previous_gen = canonical_gen
        delta = current_gen - anchor.gen_base
        previous_gen = current_gen
        merged = (
            current_gen
            if direct
            else anchor.gen_base.float() + modules.output_alpha[0].float() * delta
        )
        if need_readouts or r == config.runtime_loop_depth - 1:
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
                "alpha": None if direct else float(modules.output_alpha[0].detach()),
                "memory_permutation": permutation.tolist()
                if permutation is not None
                else None,
                "memory_writer": "canonical_correct"
                if separate_writer
                else "shared",
                "gen_memory_read_enabled": config.memory_control != "no_read",
                "body_passes_this_round": 2 if separate_writer else 1,
                "memory_update_ratio": ratio(memory - previous_memory, previous_memory)
                if memory is not None
                else 0.0,
                "memory_slot_stats": memory_slot_stats(memory),
                "memory_initial_slot_stats": memory_slot_stats(initial_memory),
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
    model.t2i_loop.float()
    model.t2i_loop.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]
