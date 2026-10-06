"""Native T2I preparation, deterministic per-prompt noise and one final readout."""
from contextlib import nullcontext
from types import MethodType
import hashlib
import torch
from PIL import Image
from .modeling.bagel.qwen2_navit import NaiveCache


def to_device(values, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in values.items()}


class InvalidGeneratedImage(ValueError):
    noise_hashes = None


class T2IGenerator:
    def __init__(self, bundle, runtime=None):
        self.bundle, self.runtime = bundle, runtime
        self.model, self.vae = bundle.model, bundle.vae
        self.device = next(self.model.parameters()).device

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()

    @torch.no_grad()
    def prepare(self, prompts, shapes, seeds):
        if not prompts or len(prompts) != len(shapes) or len(prompts) != len(seeds):
            raise ValueError('prompts, shapes and seeds must have equal nonzero counts')
        if any(h <= 0 or w <= 0 or h%self.model.latent_downsample or w%self.model.latent_downsample for h,w in shapes):
            raise ValueError('image shape must be positive and divisible by native latent_downsample')
        n = len(prompts)
        cache = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        inputs, lengths, ropes = self.model.prepare_prompts([0]*n, [0]*n, prompts, self.bundle.tokenizer, self.bundle.token_ids)
        self.prompt_lengths = tuple(int(length) for length in lengths)
        inputs = to_device(inputs, self.device)
        if self.runtime:
            self.runtime.begin_prefill(cache, inputs['packed_text_ids'], inputs['text_token_lens'].tolist(),
                set(self.bundle.tokenizer.all_special_ids) | set(self.bundle.token_ids.values()))
        try:
            with self.autocast():
                cache = self.model.forward_cache_update_text(cache, **inputs)
        finally:
            if self.runtime: self.runtime.end_prefill()
        flow = to_device(self.model.prepare_vae_latent(lengths, ropes, shapes, self.bundle.token_ids), self.device)
        noises, hashes = [], []
        for shape, seed in zip(shapes, seeds):
            count = (shape[0]//self.model.latent_downsample)*(shape[1]//self.model.latent_downsample)
            noise = torch.randn(count, self.model.patch_latent_dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
            hashes.append(hashlib.sha256(noise.numpy().tobytes()).hexdigest())
            noises.append(noise)
        flow['packed_init_noises'] = torch.cat(noises).to(self.device)
        flow['past_key_values'] = cache
        nullcache = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        cfg = to_device(self.model.prepare_vae_latent_cfg([0]*n, [0]*n, shapes), self.device)
        flow.update({k.replace('cfg_', 'cfg_text_', 1):v for k,v in cfg.items()})
        flow['cfg_text_past_key_values'] = nullcache
        return flow, hashes

    @torch.no_grad()
    def generate(self, prompts, shapes, seeds, num_timesteps=50, **sampler):
        if num_timesteps < 2: raise ValueError('native schedule needs at least two time points')
        if self.runtime: self.runtime.diagnostics.clear()
        try:
            flow, hashes = self.prepare(prompts, shapes, seeds)
        except Exception:
            if self.runtime and hasattr(self.runtime,'clear_prompt_state'): self.runtime.clear_prompt_state()
            raise
        original = self.model._forward_flow
        step = 0
        def forward(this, **kwargs):
            nonlocal step
            if self.runtime:
                self.runtime.progress = step/max(num_timesteps-2, 1)
                self.runtime.step_index = step
            result = original(**kwargs)
            capture = self.runtime.probe_capture if self.runtime else None
            if capture is not None and step in capture.steps:
                if len(shapes)!=1:raise ValueError('probe images require batch=1')
                t = float(kwargs['timestep'][0])
                capture.images[step] = self.decode(kwargs['x_t']-t*result,shapes[0])
                capture.timesteps[step] = t
            step += 1
            return result
        self.model._forward_flow = MethodType(forward, self.model)
        try:
            with self.autocast():
                latents = self.model.generate_image(**flow, num_timesteps=num_timesteps, cfg_img_scale=1.0,
                    enable_taylorseer=False, **sampler)
                images = [self.decode(latent, shape) for latent,shape in zip(latents, shapes)]
        except InvalidGeneratedImage as error:
            error.noise_hashes = hashes
            raise
        finally:
            self.model._forward_flow = original
            if self.runtime and hasattr(self.runtime,'clear_prompt_state'): self.runtime.clear_prompt_state()
        return images, hashes

    def decode(self, latent, shape):
        height, width = shape; h, w = height//self.model.latent_downsample, width//self.model.latent_downsample
        p, c = self.model.latent_patch_size, self.model.latent_channel
        latent = latent.reshape(1,h,w,p,p,c)
        latent = torch.einsum('nhwpqc->nchpwq', latent).reshape(1,c,h*p,w*p)
        output = self.vae.decode(latent.to(next(self.vae.parameters()).dtype))
        if not torch.isfinite(output).all(): raise InvalidGeneratedImage('nonfinite VAE output')
        pixels = ((output*.5+.5).clamp(0,1)[0].permute(1,2,0)*255).to(torch.uint8)
        return Image.fromarray(pixels.cpu().numpy())
