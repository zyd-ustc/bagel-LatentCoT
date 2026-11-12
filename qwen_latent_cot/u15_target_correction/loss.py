"""State flow matching and the evaluation-only image metrics."""

import torch


def _masked_mean(error, valid):
    weight = valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1) / error.shape[1]


def state_flow_matching(predicted_speed, target_speed, time, valid=None):
    """mean_valid [((u - (b-a)) / t)^2] with a strict interior time."""
    time = float(time)
    if not time > 0:
        raise ValueError("State flow matching needs a positive native time")
    error = ((predicted_speed - target_speed) / time).square()
    if valid is None:
        return error.mean()
    return _masked_mean(error, valid)
