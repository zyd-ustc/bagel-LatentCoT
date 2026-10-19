"""Map GEN hidden and UND feedback onto a bounded residual at the body entry."""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn

from .config import SLOT_HEADS
from .feedback import OUTPUT_INIT_STD, RMSNorm, _small_linear
from .native_path import BranchContext, sample_groups


# Backward scale used while the nonnegative gate is below this value. The
# forward product uses max(gamma, 0), so a zero or negative gate writes an
# exact zero.
GATE_WARMUP_SCALE = 1e-3


def _float32_math(device: torch.device):
    """Keep the residual in float32 while the body runs under bf16 autocast.

    NPU ``index_put`` does not cast. Autocast turns ``zeros_like`` into bf16
    and leaves the fp32 adapter row as float32, so the write fails. The gate
    product and the RMS cap still run in float32. The one cast into the
    anchor dtype stays in ``written_residual``.
    """
    if device.type in ("cpu", "cuda", "npu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _blank_float(like: torch.Tensor) -> torch.Tensor:
    return torch.zeros(like.shape, device=like.device, dtype=torch.float32)


def time_features(flow_time: float, width: int, device, dtype) -> torch.Tensor:
    half = width // 2
    freqs = torch.arange(half, device=device, dtype=torch.float32)
    angles = float(flow_time) * math.pi * (freqs + 1.0)
    features = torch.cat((angles.sin(), angles.cos()))
    if width % 2:
        features = torch.cat((features, features.new_zeros(1)))
    return features.to(dtype=dtype)


def bound_residual(
    delta: torch.Tensor,
    anchor: torch.Tensor,
    max_ratio: float,
    vae_indexes: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMS cap relative to the native body entry. Text rows stay zero."""
    if max_ratio <= 0.0:
        raise ValueError("max_delta_rms_ratio must be positive")
    with _float32_math(delta.device):
        bounded = _blank_float(delta)
        raw = delta[vae_indexes].float()
        reference = anchor[vae_indexes].float()
        # Floor the mean square before the square root. A zero residual then has a
        # finite backward: sqrt(0) is not on the graph, and a closed gate stays exact.
        raw_rms = raw.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        ref_rms = reference.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        factor = (max_ratio * ref_rms / raw_rms).clamp(max=1.0)
        bounded[vae_indexes] = raw * factor
    return bounded


def _nonnegative_gate(gamma: torch.Tensor) -> torch.Tensor:
    """Forward scale is ``max(gamma, 0)``.

    The backward stays the identity, including below zero and at the zero
    initialization. A negative parameter can move back above zero, and the
    residual it multiplies does not change sign.
    """
    forward = torch.where(gamma < 0, gamma.new_zeros(()), gamma)
    if not gamma.requires_grad:
        return forward
    restore = torch.where(gamma < 0, gamma - gamma.detach(), gamma.new_zeros(()))
    return forward + restore


def _gated_product(scale: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Float32 ``max(gamma, 0) * delta``. The forward value is that product.

    A zero or negative gate writes an exact zero. While the gate requires
    grad and the nonnegative scale is below ``GATE_WARMUP_SCALE``, the adapter
    backward uses that positive floor. The gate backward still multiplies by
    ``delta``. A detached or frozen scale is a plain product, so
    ``force_full_zero_path`` does not train the adapter.
    """
    gamma = scale.to(dtype=torch.float32).reshape(())
    values = delta.to(dtype=torch.float32)
    nonnegative = _nonnegative_gate(gamma)
    if not (torch.is_grad_enabled() and nonnegative.requires_grad):
        return nonnegative * values
    detached = nonnegative.detach()
    warmup = detached.new_tensor(GATE_WARMUP_SCALE)
    surrogate = torch.where(detached < GATE_WARMUP_SCALE, warmup, detached).detach()
    return values * surrogate + values.detach() * (nonnegative - surrogate)


def written_residual(
    delta: torch.Tensor,
    scale: torch.Tensor,
    anchor: torch.Tensor,
    max_ratio: float,
    vae_indexes: torch.Tensor,
) -> torch.Tensor:
    """Cap the residual after the gate, then cast once into the anchor dtype.

    ``max_ratio`` applies to ``gamma * A_phi``, the tensor added at the body
    entry. The product uses the float32 residual, then the cap, then one
    cast. A zero gate adds an exact zero in the anchor dtype. A small gate
    can still round that product to a bf16 zero. The cap is enforced on the
    value that is written.
    """
    scaled = _gated_product(scale, delta)
    capped = bound_residual(scaled, anchor.to(dtype=torch.float32), max_ratio, vae_indexes)
    return capped.to(dtype=anchor.dtype)


def slot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
) -> torch.Tensor:
    """Each image token attends over that sample's slots in several subspaces.

    ``query``, ``key``, and ``value`` already have total width ``rank``. That
    width is split into ``num_heads`` of dimension ``rank // num_heads``.
    Each head normalizes its own query and key before the scores, so a large
    projection magnitude does not scale the logits. Heads keep separate slot
    weights and the result is concatenated back to width ``rank``.
    """
    if query.ndim != 2 or key.ndim != 2 or key.shape != value.shape:
        raise ValueError("slot attention expects [tokens, width] queries and [slots, width] memory")
    width = query.shape[-1]
    if width != key.shape[-1] or key.shape[0] == 0:
        raise ValueError("slot attention width or slot count is invalid")
    if num_heads < 1 or width % num_heads != 0:
        raise ValueError(f"slot attention rank {width} must be divisible by num_heads {num_heads}")
    head_dim = width // num_heads
    if q_norm.weight.numel() != head_dim or k_norm.weight.numel() != head_dim:
        raise ValueError("Q/K RMSNorm width must equal the head dimension")
    tokens = query.shape[0]
    slots = key.shape[0]
    q = q_norm(query.reshape(tokens, num_heads, head_dim)).permute(1, 0, 2)
    k = k_norm(key.reshape(slots, num_heads, head_dim)).permute(1, 0, 2)
    v = value.reshape(slots, num_heads, head_dim).permute(1, 0, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)
    weights = torch.softmax(scores, dim=-1)
    mixed = torch.matmul(weights, v)
    return mixed.permute(1, 0, 2).reshape(tokens, width)


class AnchoredReentryAdapter(nn.Module):
    """``A_phi(H_r, M_r, P_t, t) -> delta`` on image tokens only.

    Memory stays a set of slots. Image token ``i`` reads them with multi-head
    cross-attention of total width ``rank``, ``W_O Attn(W_Q H_i, W_K M, W_V M)``.
    ``W_Q``, ``W_K``, ``W_V``, and ``W_O`` keep that width. The attention splits
    it into ``SLOT_HEADS`` heads and applies RMSNorm to each head's query and
    key. The anchor and the flow time stay on the low-rank path and are added
    to that read. Text rows stay zero. The gate and the RMS bound sit outside
    this module.
    """

    def __init__(self, hidden: int, rank: int, time_width: int = 32):
        super().__init__()
        if rank % SLOT_HEADS != 0:
            raise ValueError(f"rank {rank} must be divisible by {SLOT_HEADS} slot heads")
        self.rank = rank
        self.num_heads = SLOT_HEADS
        self.head_dim = rank // SLOT_HEADS
        self.time_width = time_width
        self.norm_h = RMSNorm(hidden)
        self.norm_m = RMSNorm(hidden)
        self.norm_p = RMSNorm(hidden)
        self.q_proj = _small_linear(hidden, rank)
        self.k_proj = _small_linear(hidden, rank)
        self.v_proj = _small_linear(hidden, rank)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.mem_out = _small_linear(rank, hidden, std=OUTPUT_INIT_STD)
        self.h_down = _small_linear(hidden, rank)
        self.p_down = _small_linear(hidden, rank)
        self.time_down = _small_linear(time_width, rank)
        self.up = _small_linear(rank, hidden, std=OUTPUT_INIT_STD)

    def forward(self, hidden, feedback, anchor, flow_time: float, branch: BranchContext) -> torch.Tensor:
        groups = sample_groups(branch, anchor.shape[0])
        with _float32_math(anchor.device):
            hidden = hidden.float()
            anchor_f = anchor.float()
            memory_value = feedback.value.float()
            if memory_value.shape[0] != len(groups) or memory_value.shape[-1] != hidden.shape[-1]:
                raise ValueError("feedback slots do not match the packed samples")
            delta = _blank_float(anchor)
            clock = time_features(flow_time, self.time_width, anchor.device, torch.float32)
            time_bias = self.time_down(clock)
            for sample, group in enumerate(groups):
                vae = group["vae"]
                image = self.norm_h(hidden[vae])
                slots = self.norm_m(memory_value[sample])
                read = slot_attention(
                    self.q_proj(image), self.k_proj(slots), self.v_proj(slots),
                    num_heads=self.num_heads, q_norm=self.q_norm, k_norm=self.k_norm,
                )
                context = self.h_down(image) + self.p_down(self.norm_p(anchor_f[vae])) + time_bias
                delta[vae] = (self.mem_out(read) + self.up(context)).float()
            return delta

    def state_free_delta(self, hidden, anchor, flow_time: float, branch: BranchContext) -> torch.Tensor:
        """``ΔH(H_r, P_t, t)`` on image tokens. Memory slots are not an input."""
        groups = sample_groups(branch, anchor.shape[0])
        with _float32_math(anchor.device):
            hidden = hidden.float()
            anchor_f = anchor.float()
            delta = _blank_float(anchor)
            clock = time_features(flow_time, self.time_width, anchor.device, torch.float32)
            time_bias = self.time_down(clock)
            for _sample, group in enumerate(groups):
                vae = group["vae"]
                image = self.norm_h(hidden[vae])
                context = self.h_down(image) + self.p_down(self.norm_p(anchor_f[vae])) + time_bias
                delta[vae] = self.up(context).float()
            return delta


class EntryGate(nn.Module):
    """Learned scale. Zero is an exact bypass multiplier, not a new token."""

    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        return self.raw
