"""FP32 Flow-GRPO transitions shared by sampling and replay (one image)."""
from dataclasses import dataclass
import torch


@dataclass
class Transition:
    sample: torch.Tensor
    mean: torch.Tensor
    variance: float
    score: torch.Tensor


def transition(x, velocity, t, s, noise_level, *, next_sample=None, noise=None):
    if not 0 < s < t < 1 or noise_level <= 0:
        raise ValueError('Stochastic transition requires 0 < s < t < 1 and positive noise')
    if x.ndim != 2 or x.shape != velocity.shape:
        raise ValueError('Expected ONE packed image [tokens, channels]')
    x, velocity = x.float(), velocity.float()
    dt = s-t
    diffusion_squared = t/(1-t)*noise_level**2
    mean = x*(1+diffusion_squared/(2*t)*dt) + velocity*(1+diffusion_squared*(1-t)/(2*t))*dt
    variance = diffusion_squared*(-dt)
    if next_sample is None:
        if noise is None or noise.shape != x.shape:
            raise ValueError('Rollout requires explicit transition noise of the image shape')
        next_sample = mean + variance**.5*noise.float()
    # The native proxy averages latent dimensions and omits fixed constants.
    score = -(next_sample.detach().float()-mean).square().mean()/(2*variance)
    return Transition(next_sample, mean, variance, score)


def gaussian_kl(policy, reference):
    if policy.variance != reference.variance:
        raise ValueError('KL requires identical fixed transition variances')
    return (policy.mean-reference.mean.detach()).square().mean()/(2*policy.variance)


def advantages(records):
    groups = {}
    for row in records:
        groups.setdefault(row['group'], []).append(row)
    output, flat = {}, 0
    for items in groups.values():
        if len(items) != 8 or len({r['candidate'] for r in items}) != 8:
            raise ValueError('Each prompt group requires exactly eight candidates')
        rewards = torch.tensor([r['reward'] for r in items], dtype=torch.float32)
        std = rewards.std(unbiased=False)
        flat += int(std < 1e-6)
        normalized = torch.zeros_like(rewards) if std < 1e-6 else (rewards-rewards.mean())/(std+1e-6)
        for row, value in zip(items, normalized):
            output[(row['group'], row['candidate'])] = float(value)
    return output, flat/len(groups)


def policy_loss(current, old_score, advantage, reference, clip, beta):
    ratio = torch.exp(current.score-old_score)
    objective = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    kl = gaussian_kl(current, reference)
    return -objective+beta*kl, ratio, kl
