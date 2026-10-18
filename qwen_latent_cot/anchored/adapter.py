"""Map GEN hidden and UND feedback onto a bounded residual at the body entry."""
from __future__ import annotations

import math

import torch
from torch import nn

from .feedback import RMSNorm, _small_linear
from .native_path import BranchContext, sample_groups


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
    bounded = torch.zeros_like(delta)
    raw = delta[vae_indexes]
    reference = anchor[vae_indexes]
    raw_rms = raw.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    ref_rms = reference.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    factor = (max_ratio * ref_rms / raw_rms.clamp_min(1e-6)).clamp(max=1.0)
    bounded[vae_indexes] = (raw.float() * factor).to(dtype=delta.dtype)
    return bounded


def slot_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Each image token attends over that sample's memory slots."""
    if query.ndim != 2 or key.ndim != 2 or key.shape != value.shape:
        raise ValueError("slot attention expects [tokens, width] queries and [slots, width] memory")
    if query.shape[-1] != key.shape[-1] or key.shape[0] == 0:
        raise ValueError("slot attention width or slot count is invalid")
    scores = query @ key.transpose(0, 1) * (query.shape[-1] ** -0.5)
    return torch.softmax(scores, dim=-1) @ value


class AnchoredReentryAdapter(nn.Module):
    """``A_phi(H_r, M_r, P_t, t) -> delta`` on image tokens only.

    Memory stays a set of slots. Image token ``i`` reads them with one-head
    cross-attention, ``W_O Attn(W_Q H_i, W_K M, W_V M)``. The anchor and the
    flow time stay on the low-rank path and are added to that read. Text rows
    stay zero. The gate and the RMS bound sit outside this module.
    """

    def __init__(self, hidden: int, rank: int, time_width: int = 32):
        super().__init__()
        self.rank = rank
        self.time_width = time_width
        self.norm_h = RMSNorm(hidden)
        self.norm_m = RMSNorm(hidden)
        self.norm_p = RMSNorm(hidden)
        self.q_proj = _small_linear(hidden, rank)
        self.k_proj = _small_linear(hidden, rank)
        self.v_proj = _small_linear(hidden, rank)
        self.mem_out = _small_linear(rank, hidden)
        self.h_down = _small_linear(hidden, rank)
        self.p_down = _small_linear(hidden, rank)
        self.time_down = _small_linear(time_width, rank)
        self.up = _small_linear(rank, hidden)

    def forward(self, hidden, feedback, anchor, flow_time: float, branch: BranchContext) -> torch.Tensor:
        groups = sample_groups(branch, anchor.shape[0])
        restored = anchor.dtype
        hidden = hidden.float()
        anchor_f = anchor.float()
        memory_value = feedback.value.float()
        if memory_value.shape[0] != len(groups) or memory_value.shape[-1] != hidden.shape[-1]:
            raise ValueError("feedback slots do not match the packed samples")
        delta = torch.zeros_like(anchor_f)
        clock = time_features(flow_time, self.time_width, anchor.device, torch.float32)
        time_bias = self.time_down(clock)
        for sample, group in enumerate(groups):
            vae = group["vae"]
            image = self.norm_h(hidden[vae])
            slots = self.norm_m(memory_value[sample])
            read = slot_attention(self.q_proj(image), self.k_proj(slots), self.v_proj(slots))
            context = self.h_down(image) + self.p_down(self.norm_p(anchor_f[vae])) + time_bias
            delta[vae] = self.mem_out(read) + self.up(context)
        return delta.to(dtype=restored)


class EntryGate(nn.Module):
    """Learned scale. Zero is an exact bypass multiplier, not a new token."""

    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        return self.raw
