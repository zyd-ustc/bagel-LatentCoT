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


def masked_l1(prediction, target, valid):
    return _masked_mean((prediction - target).abs(), valid)


def render(image, valid):
    """Clamp to the display range and crop the shared valid rectangle."""
    shown = image.clamp(-1, 1)
    occupied = valid[:, :1] > 0
    rows = occupied[0, 0].any(1).nonzero(as_tuple=False).flatten()
    cols = occupied[0, 0].any(0).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        return shown
    return shown[:, :, int(rows[0]):int(rows[-1]) + 1, int(cols[0]):int(cols[-1]) + 1]


def build_perceptual(device):
    """LPIPS is used by cache validation and evaluation, never by the loss."""
    import lpips

    network = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


@torch.no_grad()
def perceptual_distance(perceptual, prediction, target, valid):
    value = perceptual(render(prediction, valid), render(target, valid))
    return float(value.mean() if value.ndim else value)


def reconstruction_metrics(prediction, target, valid, perceptual=None):
    metrics = {"l1": float(masked_l1(prediction, target, valid))}
    if perceptual is not None:
        metrics["lpips"] = perceptual_distance(perceptual, prediction, target, valid)
    return metrics
