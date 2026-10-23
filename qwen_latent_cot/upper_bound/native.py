"""Native BAGEL sampling with early stop, preview decoding and rollback.

Only the frozen generation path is used: CFG + global renorm velocity, Euler
steps, VAE decode. No corrector, no training.
"""
import torch

from ..correction.runtime import Runtime
from ..correction.sampling import times
from ..correction.numerics import autocast_for


def _randn(shape, seed, device):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=generator).to(device)


class NativeSampler:
    """Runs one trajectory and exposes its intermediate states."""

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.config = runtime.config
        self.grid = times(self.config)

    def channels(self, flow):
        model = self.runtime.model
        return model.latent_channel * model.latent_patch_size ** 2

    def start(self, prompt, seed):
        flow = self.runtime.flow(prompt)
        tokens = len(flow['packed_vae_token_indexes'])
        x = _randn((tokens, self.channels(flow)), seed, self.runtime.device)
        return flow, x

    def step(self, x, velocity, index):
        """One Euler step from node `index` to `index+1`, native arithmetic."""
        t, following = self.grid[index], self.grid[index + 1]
        return x - (t - following) * velocity

    def velocity(self, x, flow, index):
        return self.runtime.proposal(x, self.grid[index], flow, capture=False)[0]

    @torch.no_grad()
    def run(self, prompt, seed, *, save_indexes=(), save_velocity=()):
        """Full trajectory; returns final latent, saved states and saved velocities."""
        flow, x = self.start(prompt, seed)
        states, velocities = {}, {}
        for index in range(len(self.grid) - 1):
            velocity = self.velocity(x, flow, index)
            if index in save_indexes:
                states[index] = x.detach().clone()
            if index in save_velocity:
                velocities[index] = velocity.detach().clone()
            x = self.step(x, velocity, index)
        if (len(self.grid) - 1) in save_indexes:
            states[len(self.grid) - 1] = x.detach().clone()
        return dict(latent=x, flow=flow, states=states, velocities=velocities)

    @torch.no_grad()
    def resume(self, latent, flow, start_index):
        """Continue the Euler loop from `start_index` with a (possibly new) prompt flow."""
        x = latent
        for index in range(start_index, len(self.grid) - 1):
            x = self.step(x, self.velocity(x, flow, index), index)
        return x

    def renoise(self, latent, velocity, index, seed):
        """Fresh noise at node `index`, keeping the current one-step clean estimate."""
        t = self.grid[index]
        clean = (latent.float() - t * velocity.float())
        noise = _randn(tuple(clean.shape), seed, clean.device)
        return ((1 - t) * clean + t * noise).to(latent.dtype)

    def decode(self, latent):
        with torch.no_grad():
            return self.runtime.image(latent)

    def flow(self, prompt):
        return self.runtime.flow(prompt)

    def t_at(self, index):
        return self.grid[index]
