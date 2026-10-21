"""Frozen native proposal, same-pass feature capture, and native latent codec."""
from contextlib import contextmanager
import torch
from PIL import Image
import numpy as np
from .attention import Context, project, rotate
from ..bagel.accelerator import seeded_context
from .numerics import autocast_for


def device_inputs(inputs, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in inputs.items()}


class Runtime:
    def __init__(self, bundle, config):
        self.bundle, self.config = bundle, dict(config)
        self.model = bundle.model.eval().requires_grad_(False)
        self.bundle.vae.eval().requires_grad_(False)
        self.decoder = self.model.language_model.model
        self.device = next(self.decoder.parameters()).device
        self.decoder.enable_taylorseer = False
        for layer in self.decoder.layers: layer.enable_taylorseer = False

    def empty_cache(self):
        from ..bagel.modeling.bagel.qwen2_navit import NaiveCache
        return NaiveCache(len(self.decoder.layers))

    @torch.no_grad()
    def flow(self, prompt):
        inputs, lengths, positions = self.model.prepare_prompts([0], [0], [prompt], self.bundle.tokenizer, self.bundle.token_ids)
        with autocast_for(self.device):
            cache = self.model.forward_cache_update_text(self.empty_cache(), **device_inputs(inputs, self.device))
        size = self.config['resolution']
        with seeded_context(self.device, 0):
            flow = self.model.prepare_vae_latent(lengths, positions, [(size,size)], self.bundle.token_ids)
        flow.pop('packed_init_noises')
        flow = device_inputs(flow, self.device); flow['past_key_values'] = cache
        cfg = self.model.prepare_vae_latent_cfg([0], [0], [(size,size)])
        flow.update({k.replace('cfg_', 'cfg_text_', 1):v for k,v in device_inputs(cfg, self.device).items()})
        flow['cfg_text_past_key_values'] = self.empty_cache()
        return flow

    @contextmanager
    def capture(self, flow):
        context, handles = {}, []
        cache = flow['past_key_values']
        def observer(index):
            def hook(module, args, kwargs):
                if kwargs.get('past_key_values') is not cache or kwargs.get('mode') != 'gen': return
                indexes = kwargs['packed_vae_token_indexes']
                normalized = kwargs['packed_query_sequence'][indexes]
                _, keys, values = project(module, normalized, generation=True)
                cos,sin = kwargs['packed_query_position_embeddings']
                keys = rotate(keys, (cos[indexes], sin[indexes])).to(torch.bfloat16)
                # Prompt and image features are frozen context, never a trainable Memory bank.
                context[index] = Context(torch.cat((cache.key_cache[index], keys)).detach(),
                                         torch.cat((cache.value_cache[index], values.to(torch.bfloat16))).detach())
            return hook
        try:
            for index,layer in enumerate(self.decoder.layers):
                handles.append(layer.self_attn.register_forward_pre_hook(observer(index), with_kwargs=True))
            yield context
            if len(context)!=len(self.decoder.layers): raise RuntimeError('Incomplete GEN context capture')
        finally:
            for handle in handles: handle.remove()

    @torch.no_grad()
    def predict(self, x_t, t, flow, *, capture=True):
        with self.capture(flow) if capture else _empty_context() as context:
            with autocast_for(self.device):
                velocity = self.model._forward_flow(x_t=x_t, timestep=torch.full((len(x_t),), t, device=self.device),
                    **flow, cfg_text_scale=1., cfg_img_scale=1., cfg_renorm_min=self.config['cfg_renorm_min'],
                    cfg_renorm_type=self.config['cfg_renorm_type'])
        return velocity.detach(), tuple(context[i] for i in range(len(context)))

    @torch.no_grad()
    def reference(self, x_t, t, flow):
        # Native unconditional pass. No adapter or correction is applied to this branch.
        reference = dict(flow)
        for name in ('packed_position_ids','packed_indexes','key_values_lens','past_key_values','packed_key_value_indexes'):
            cfg_name = {'packed_indexes':'packed_query_indexes'}.get(name,name)
            reference[name] = flow['cfg_text_'+cfg_name]
        return self.predict(x_t,t,reference,capture=False)[0]

    @torch.no_grad()
    def encode(self, path, posterior_seed):
        with Image.open(path) as image:
            image = image.convert('RGB').resize((self.config['resolution'],)*2, Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(image)).permute(2,0,1)[None].float().to(self.device)/127.5-1
        with seeded_context(self.device,posterior_seed), autocast_for(self.device):
            latent = self.bundle.vae.encode(pixels.to(next(self.bundle.vae.parameters()).dtype))
        p = self.model.latent_patch_size
        _,c,h,w = latent.shape
        return torch.einsum('chpwq->hwpqc',latent[0].reshape(c,h//p,p,w//p,p)).reshape(-1,p*p*c)

    @torch.no_grad()
    def image(self, x):
        p = self.model.latent_patch_size; size = self.config['resolution']//self.model.latent_downsample
        c = self.model.latent_channel
        latent = torch.einsum('hwpqc->chpwq',x.reshape(size,size,p,p,c)).reshape(1,c,size*p,size*p)
        with autocast_for(self.device): image = self.bundle.vae.decode(latent)
        array = ((image[0].float().clamp(-1,1)+1)*127.5).permute(1,2,0).cpu().byte().numpy()
        return Image.fromarray(array)


@contextmanager
def _empty_context():
    yield {}
