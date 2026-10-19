"""Feedback state, its update, and same-step content interventions."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FeedbackState:
    """Per-branch feedback slots. ``round_index`` 0 is the initialization."""

    value: torch.Tensor
    branch: str
    round_index: int
    flow_time: float

    def __post_init__(self):
        if self.value.ndim != 3:
            raise ValueError("feedback value must be [samples, slots, hidden]")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Parameters stay float32. Activations may be the native bf16.
        restored = hidden.dtype
        hidden = hidden.float()
        scale = hidden.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (hidden * scale * self.weight.float()).to(dtype=restored)


# Earlier projections stay small. Output projections are larger so the float32
# residual is nonzero on the first step, while the written product can still
# underflow bf16 until the gate grows.
PROJECTION_INIT_STD = 1e-3
OUTPUT_INIT_STD = 1e-2


def _small_linear(in_features: int, out_features: int, *, std: float = PROJECTION_INIT_STD) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(layer.weight, std=std)
    return layer


class FeedbackStateUpdater(nn.Module):
    """``M_{r+1} = base + U_phi(base + O_r)``. The state is not written into GEN tokens.

    A zero state, which is the initialization, uses the observation as ``base``.
    Later rounds keep that state and add a pre-norm residual, so the update does
    not replace the previous magnitude with a normalized vector.
    """

    def __init__(self, hidden: int, slots: int, rank: int):
        super().__init__()
        self.slots = slots
        self.hidden = hidden
        self.norm = RMSNorm(hidden)
        self.down = _small_linear(hidden, rank)
        self.up = _small_linear(rank, hidden, std=OUTPUT_INIT_STD)

    def initial(self, samples: int, like: torch.Tensor, branch: str, flow_time: float) -> FeedbackState:
        value = torch.zeros(samples, self.slots, self.hidden, device=like.device, dtype=like.dtype)
        return FeedbackState(value, branch, 0, float(flow_time))

    def forward(self, state: FeedbackState, observation: torch.Tensor) -> FeedbackState:
        if observation.shape != state.value.shape:
            raise ValueError("observation and feedback slots have different shapes")
        obs_f = observation.float()
        state_f = state.value.float()
        # The whole initial state is zero. One sample being zero does not count.
        empty = state_f.detach().abs().max().item() == 0.0
        base = obs_f if empty else state_f
        joint = self.norm(base + obs_f)
        delta = self.up(F.silu(self.down(joint)))
        value = (base + delta).to(dtype=state.value.dtype)
        return FeedbackState(value, state.branch, state.round_index + 1, state.flow_time)


def intervene_feedback(
    state: FeedbackState,
    mode: str,
    *,
    initial: FeedbackState,
    donor: FeedbackState | None = None,
) -> FeedbackState:
    """Apply a content control at the same time, round, and CFG branch."""
    if initial.branch != state.branch or initial.flow_time != state.flow_time:
        raise ValueError("feedback intervention crossed time or CFG branch")
    if mode == "correct":
        return state
    if mode == "zero":
        return replace(state, value=torch.zeros_like(state.value))
    if mode == "mean":
        mean = state.value.mean(dim=0, keepdim=True).expand_as(state.value).contiguous()
        return replace(state, value=mean)
    if mode == "frozen":
        return replace(state, value=initial.value.detach())
    if mode != "shuffled":
        raise ValueError(f"unknown feedback mode {mode}")
    if donor is None:
        if state.value.shape[0] < 2:
            raise ValueError("shuffled feedback needs another sample in this branch or an explicit donor")
        return replace(state, value=state.value.roll(1, dims=0))
    if donor.branch != state.branch:
        raise ValueError("shuffled donor crossed CFG branches")
    if donor.flow_time != state.flow_time or donor.round_index != state.round_index:
        raise ValueError("shuffled donor must match flow time and round")
    if donor.value.shape != state.value.shape or donor.value.dtype != state.value.dtype:
        raise ValueError("shuffled donor must match shape and dtype")
    return replace(state, value=donor.value)
