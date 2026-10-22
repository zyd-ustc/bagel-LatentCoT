"""RGB flow uses t=0 noise, t=1 clean; all projections are per image."""
import torch


def coefficient(delta, guidance):
    a, g = delta.float().flatten(1), guidance.detach().float().flatten(1)
    denominator = g.square().sum(1)
    return torch.where(denominator > 1e-12, (a*g).sum(1)/denominator.clamp_min(1e-12), 0.)


def project(delta, guidance):
    scale = coefficient(delta, guidance).clamp_max(0)
    return delta.float()-scale.reshape(-1, *([1]*(delta.ndim-1)))*guidance.detach().float()


def residual_target(clean, noise, sigma, conditional, guided):
    truth = clean.float()-float(sigma)*noise.float()
    return project(truth-conditional.detach().float(), guided.detach().float()-conditional.detach().float())


def cfg_velocity(conditional, unconditional, scale, norm, patch):
    """Match upstream CFG norm in PATCH layout, including 'channel' semantics."""
    if scale <= 1: return conditional
    value = unconditional+scale*(conditional-unconditional)
    if norm == 'none': return value
    c, v = patch(conditional), patch(value)
    axes = (1, 2) if norm == 'global' else -1
    factor = (c.norm(dim=axes, keepdim=True)/(v.norm(dim=axes, keepdim=True)+1e-8)).clamp(0, 1)
    return v*factor  # Caller unpatchifies this explicitly.
