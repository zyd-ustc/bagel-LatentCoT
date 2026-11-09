"""Terminal image supervision: judge the artwork, not the internal state.

The label image and the clean image cannot be the same picture. A label that
carries the edit is produced by a sampler that is itself rougher than native
generation (measured: 1.7-2.1x the high-frequency energy of the parent's own
completion), while the parent's own completion is clean but has no edit. This
module keeps the two roles apart:

* structure inside the edit region comes from the label image;
* everything outside it is anchored on the parent's native completion;
* and the total high-frequency energy is given an explicit budget, so the edit
  axis can no longer buy displacement by injecting texture.

Nothing here looks at internal states, so no target state has to be inverse
solved, inverted, or accepted by a reconstruction gate.
"""

import torch
from torch.nn import functional as F

from .loss import high_frequency, masked_l1


def edit_mask(teacher, source, valid, lo=0.6, hi=0.95, blur=3):
    """Soft mask of where the label image departs from the parent completion.

    Quantiles are taken inside the valid region only, so letterboxing cannot
    drive the threshold. ``blur`` averages the mask so a hard seam cannot put
    neighbouring pixels in opposite halves of the loss.
    """
    with torch.no_grad():
        difference = (teacher - source).abs().mean(1, keepdim=True)
        mask = torch.zeros_like(difference)
        for index in range(difference.shape[0]):
            inside = difference[index][0][valid[index][0] > 0]
            if inside.numel() < 16:
                continue
            low = torch.quantile(inside, float(lo))
            high = torch.quantile(inside, float(hi))
            span = (high - low).clamp_min(1e-4)
            mask[index] = ((difference[index] - low) / span).clamp(0, 1)
        if blur > 1:
            weight = F.avg_pool2d(valid, blur, stride=1, padding=blur // 2)
            mask = F.avg_pool2d(mask, blur, stride=1, padding=blur // 2)
            mask = mask * valid / weight.clamp_min(1e-6)
        return (mask * valid).clamp(0, 1)


ROUTING_KEYS = frozenset({"enabled", "roles"})


def loss_options(terminal):
    """Drop routing keys: ``roles`` selects which roles use this loss, the rest are
    its hyper-parameters. Passing everything used to raise a TypeError."""
    return {key: value for key, value in (terminal or {}).items() if key not in ROUTING_KEYS}


def terminal_loss(prediction, teacher, source, valid, *, preserve_weight=1.0,
                  frequency_weight=0.5, mask_low=0.6, mask_high=0.95,
                  ratio_low=0.75, ratio_high=1.15,
                  frequency_floor=0.01):
    """Structure on the edit, preservation outside it, and an explicit HF band.

    The band is two-sided on purpose: too rough is the failure we measured, but
    an unguarded penalty would happily trade texture for blur, which the audits
    would then score as "clean" while the picture is mush.
    """
    mask = edit_mask(teacher, source, valid, mask_low, mask_high).detach()
    outside = (1.0 - mask) * valid
    structure = masked_l1(prediction, teacher, mask)
    preserve = masked_l1(prediction, source, outside)
    reference = high_frequency(source).detach().clamp_min(float(frequency_floor))
    ratio = high_frequency(prediction) / reference

    def _huber(distance):
        """Quadratic near the band, linear far from it.

        A plain square with a hard clamp would flatten to zero gradient once the
        ratio passed the cap, which is exactly the sample that needs suppressing.
        """
        return torch.where(distance <= 1.0, 0.5 * distance.square(), distance - 0.5)

    frequency = (
        _huber((ratio - float(ratio_high)).clamp_min(0))
        + _huber((float(ratio_low) - ratio).clamp_min(0))
    ).mean()
    total = structure + float(preserve_weight) * preserve + float(frequency_weight) * frequency
    return total, dict(
        loss=float(total.detach()),
        structure=float(structure.detach()),
        preserve=float(preserve.detach()),
        frequency=float(frequency.detach()),
        hf_ratio=float(
            (high_frequency(prediction).detach() / high_frequency(source).detach().clamp_min(1e-6)).mean()
        ),
        edited_fraction=float((mask > 0.5).float().mean().detach()),
    )
