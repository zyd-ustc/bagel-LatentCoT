"""Native BAGEL logit-normal flow time and inference schedule transform."""

from __future__ import annotations

import torch


def shift_flow_timestep(t, timestep_shift=1.0):
    if timestep_shift <= 0:
        raise ValueError("timestep_shift must be positive")
    return timestep_shift * t / (1 + (timestep_shift - 1) * t)


def sample_native_flow_timestep(
    shape=(), *, device=None, generator=None, timestep_shift=1.0, raw_t=None
):
    """Raw N(0,1) -> sigmoid -> native shift; raw_t permits native packed inputs.

    Native forward already receives raw normal logits from its data pipeline.
    Stage 1 draws those logits here. Both use the same transform; new samples use FP32 time.
    """
    if raw_t is None:
        raw_t = torch.randn(
            shape, device=device, generator=generator, dtype=torch.float32
        )
    return shift_flow_timestep(torch.sigmoid(raw_t), timestep_shift)
