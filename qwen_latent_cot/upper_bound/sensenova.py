"""SenseNova U1.5 adapters for the explicit closed-loop upper bound.

The U1.5 T2I path works in pixel space: the state is an RGB tensor, t runs
0 (noise) -> 1 (clean), and x_t = t*x0 + (1-t)*sigma*noise with velocity
x0 - sigma*noise. Preview decoding is therefore a clamp, not a VAE decode.
"""
import numpy as np
import torch
from PIL import Image

from ..u15.runtime import autocast


class SenseNovaSampler:
    def __init__(self, runtime):
        self.runtime = runtime
        self.config = runtime.config
        self.grid = [float(v) for v in runtime.times().tolist()]
        self.num_steps = len(self.grid) - 1
        self._reference = None

    def _ref(self):
        if self._reference is None:
            self._reference = self.runtime.prefix(None)
        return self._reference

    def start(self, prompt, seed):
        prefix = self.runtime.prefix(prompt)
        dtype = next(self.runtime.model.parameters()).dtype
        generator = torch.Generator(device=self.runtime.device).manual_seed(int(seed))
        image = self.runtime.noise_scale() * torch.randn(
            (1, 3, self.runtime.height, self.runtime.width),
            device=self.runtime.device, dtype=dtype, generator=generator)
        return prefix, image

    def velocity(self, image, prefix, index):
        t = self.grid[index]
        conditional = self.runtime.predict(image, t, prefix).velocity
        if float(self.config['cfg_scale']) > 1:
            reference = self.runtime.predict(image, t, self._ref()).velocity
        else:
            reference = conditional
        return self.runtime.guided(conditional, reference, float(self.config['cfg_scale']))

    def step(self, image, velocity, index):
        return image + (self.grid[index + 1] - self.grid[index]) * velocity

    @torch.no_grad()
    def run(self, prompt, seed, *, save_indexes=(), save_velocity=()):
        prefix, image = self.start(prompt, seed)
        states, velocities = {}, {}
        for index in range(self.num_steps):
            velocity = self.velocity(image, prefix, index)
            if index in save_indexes:
                states[index] = image.detach().clone()
            if index in save_velocity:
                velocities[index] = velocity.detach().clone()
            image = self.step(image, velocity, index)
        if self.num_steps in save_indexes:
            states[self.num_steps] = image.detach().clone()
        return dict(latent=image, flow=prefix, states=states, velocities=velocities)

    @torch.no_grad()
    def resume(self, latent, prefix, start_index):
        image = latent
        for index in range(start_index, self.num_steps):
            image = self.step(image, self.velocity(image, prefix, index), index)
        return image

    def renoise(self, latent, velocity, index, seed):
        """New seed at node `index`, keeping the one-step clean estimate."""
        t = self.grid[index]
        sigma = self.runtime.noise_scale()
        clean = latent.float() + (1 - t) * velocity.float()
        generator = torch.Generator(device=latent.device).manual_seed(int(seed))
        noise = sigma * torch.randn(latent.shape, device=latent.device,
                                    dtype=torch.float32, generator=generator)
        return (t * clean + (1 - t) * noise).to(latent.dtype)

    def decode(self, latent):
        array = ((latent[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
        return Image.fromarray(np.asarray(array.permute(1, 2, 0).cpu().numpy()))

    def flow(self, prompt):
        return self.runtime.prefix(prompt)

    def t_at(self, index):
        return self.grid[index]


def strip_think(text):
    if '</think>' in text:
        text = text.split('</think>')[-1]
    return text.strip()


class SenseNovaReader:
    """Vision-language chat on the native U1.5 UND path."""

    def __init__(self, runtime, template='{question}', max_answer_tokens=8):
        self.runtime = runtime
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.device = runtime.device
        self.template = template
        self.max_answer_tokens = int(max_answer_tokens)

    @torch.no_grad()
    def ask(self, image, question):
        from sensenova_u1.models.neo_unify.utils import load_image_native
        pixel_values, grid_hw = load_image_native(image)
        pixel_values = pixel_values.to(self.device)
        grid_hw = grid_hw.to(self.device)
        prompt = self.template.format(question=question)
        generation = dict(max_new_tokens=self.max_answer_tokens, do_sample=False)
        with autocast(self.device):
            answer = self.model.chat(self.tokenizer, pixel_values, prompt,
                                     generation, grid_hw=grid_hw)
        return strip_think(answer)

    def score(self, image, vqa_list):
        return [self.ask(image, question) for question, _ in vqa_list]
