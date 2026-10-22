"""Conditional residual targets and one shared no-anti-guidance projection."""
import torch

MODE='conditional_no_anti_guidance'


def guidance_safe(config):
    mode=config.get('correction_objective')
    if mode is None:return False
    if mode!=MODE or config.get('correction_space')!='post_cfg':
        raise ValueError('conditional_no_anti_guidance requires post_cfg sampling')
    return True


def guidance_coefficient(delta,guidance):
    delta,guidance=delta.float(),guidance.detach().float()
    return (delta*guidance).sum()/guidance.square().sum().clamp_min(1e-12)


def remove_anti_guidance(delta,guidance):
    """One packed image per call; project onto <delta,g> >= 0, preserving the orthogonal part."""
    guidance=guidance.detach().float()
    return delta.float()-guidance_coefficient(delta,guidance).clamp_max(0)*guidance


def residual_target(base,x0,epsilon,config,conditional=None):
    if guidance_safe(config):
        if conditional is None:raise ValueError('Guidance-safe target requires conditional velocity')
        residual=epsilon.detach().float()-x0.detach().float()-conditional.detach().float()
        return remove_anti_guidance(residual,base.detach().float()-conditional.detach().float())
    return epsilon.detach().float()-x0.detach().float()-base.detach().float()


def applied_correction(delta,base,config,conditional=None):
    if guidance_safe(config):
        if conditional is None:raise ValueError('Guidance-safe sampling requires conditional velocity')
        return remove_anti_guidance(delta,base.detach().float()-conditional.detach().float())
    return delta
