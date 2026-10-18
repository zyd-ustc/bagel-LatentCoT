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


class AnchoredReentryAdapter(nn.Module):
    """``A_phi(H_r, M_r, P_t, t) -> delta`` on image tokens only."""

    def __init__(self, hidden: int, rank: int, time_width: int = 32):
        super().__init__()
        self.rank = rank
        self.time_width = time_width
        self.norm_h = RMSNorm(hidden)
        self.norm_m = RMSNorm(hidden)
        self.norm_p = RMSNorm(hidden)
        self.h_down = _small_linear(hidden, rank)
        self.m_down = _small_linear(hidden, rank)
        self.p_down = _small_linear(hidden, rank)
        self.time_down = _small_linear(time_width, rank)
        self.up = _small_linear(rank, hidden)

    def forward(self, hidden, feedback, anchor, flow_time: float, branch: BranchContext) -> torch.Tensor:
        groups = sample_groups(branch, anchor.shape[0])
        restored = anchor.dtype
        hidden = hidden.float()
        anchor_f = anchor.float()
        memory_value = feedback.value.float()
        delta = torch.zeros_like(anchor_f)
        clock = time_features(flow_time, self.time_width, anchor.device, torch.float32)
        time_bias = self.time_down(clock)
        for sample, group in enumerate(groups):
            vae = group["vae"]
            memory = memory_value[sample].mean(dim=0, keepdim=True).expand(len(vae), -1)
            mixed = (
                self.h_down(self.norm_h(hidden[vae]))
                + self.m_down(self.norm_m(memory))
                + self.p_down(self.norm_p(anchor_f[vae]))
                + time_bias
            )
            delta[vae] = self.up(mixed)
        return delta.to(dtype=restored)


class EntryGate(nn.Module):
    """Learned scale. Zero is an exact bypass multiplier, not a new token."""

    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(()))

    def forward(self) -> torch.Tensor:
        return self.raw
