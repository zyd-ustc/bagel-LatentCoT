"""Batch-one BAGEL diagnostic orchestration with effective-batch accumulation."""
from pathlib import Path
import time
import torch
from PIL import Image
from .memory import MemoryWriter, neutral_seeds, consume_bank
from .native import text_source, device_inputs, empty_cache, capture_gen, gen_memory_call
from .data import controlled_noise, tensor_hash
from ..bagel.accelerator import seeded_context, synchronize
from .numerics import autocast_for


class DiagnosticRuntime:
    def __init__(self, bundle, config, topology):
        self.bundle, self.config = bundle, config
        self.model = bundle.model
        self.decoder = self.model.language_model.model
        self.decoder.enable_taylorseer = False
        for layer in self.decoder.layers:
            layer.enable_taylorseer = False
        self.model.eval().requires_grad_(False); bundle.vae.eval().requires_grad_(False)
        self.device = self.decoder.embed_tokens.weight.device
        # Do not register special tokens or train embeddings. Fix selection once.
        candidates = bundle.tokenizer.encode(config['neutral_text'], add_special_tokens=False)
        special = set(bundle.tokenizer.all_special_ids)
        ids = [i for i in candidates if i not in special][:config['slots']]
        if len(ids) != config['slots']:
            raise ValueError('neutral text must contain at least slots non-special tokens')
        self.seed_ids = ids
        self.writer = MemoryWriter(self.decoder, topology, config['slots'], config['rank'])
        with autocast_for(self.device):
            self.seeds = neutral_seeds(self.decoder, torch.tensor(ids, device=self.device))
        self.instruction_ids = [bundle.token_ids['bos_token_id']] + bundle.tokenizer.encode(
            config['caption_instruction'], add_special_tokens=False)
        self._neutral_context = None

    def answer_ids(self, caption):
        return self.bundle.tokenizer.encode(caption, add_special_tokens=False) + [self.bundle.token_ids['eos_token_id']]

    def text(self, text):
        with autocast_for(self.device):
            source, inputs, positions = text_source(self.bundle, [text])
        return source, inputs, positions

    @torch.no_grad()
    def prompt_context(self, prompt):
        source, inputs, positions = self.text(prompt)
        cache = empty_cache(self.decoder)
        for i, kv in enumerate(source):
            cache.key_cache[i], cache.value_cache[i] = kv.keys, kv.values
        return cache, source[0].lengths, positions

    @torch.no_grad()
    def encode_image(self, path, posterior_seed):
        image = Image.open(path).convert('RGB')
        # Explicit fixed resize is recorded; no hidden model image transforms.
        size = self.config['resolution']
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        import numpy as np
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float().to(self.device)/127.5-1
        with seeded_context(self.device, posterior_seed), autocast_for(self.device):
            latent = self.bundle.vae.encode(pixels.to(next(self.bundle.vae.parameters()).dtype))
        p = self.model.latent_patch_size
        _, c, h, w = latent.shape
        return torch.einsum('chpwq->hwpqc', latent[0].reshape(c, h//p, p, w//p, p)).reshape(-1, p*p*c)

    def flow_inputs(self, prompt):
        cache, lengths, positions = self.prompt_context(prompt)
        size = self.config['resolution']
        # Native preparation samples unused initial noise; isolate that RNG.
        with seeded_context(self.device, 0):
            flow = self.model.prepare_vae_latent(lengths, positions, [(size, size)], self.bundle.token_ids)
        flow.pop('packed_init_noises')
        flow = device_inputs(flow, self.device)
        flow['past_key_values'] = cache
        cfg = self.model.prepare_vae_latent_cfg([0], [0], [(size, size)])
        flow.update({k.replace('cfg_', 'cfg_text_', 1): v for k, v in device_inputs(cfg, self.device).items()})
        flow['cfg_text_past_key_values'] = empty_cache(self.decoder)
        return flow, lengths, positions

    @torch.no_grad()
    def state(self, row, t, noise_seed):
        if row.get('state'):
            saved = torch.load(row['state'], map_location=self.device, weights_only=True)
            if saved.get('state_source') != 'trajectory' or saved.get('prompt') != row['prompt']:
                raise ValueError('trajectory state must record source and unchanged prompt')
            if abs(float(saved['t'])-t) > 1e-7 or saved.get('noise_seed') != noise_seed:
                raise ValueError('requested t/seed differs from trajectory metadata')
            x = saved['x_t'].to(self.device)
            source = 'trajectory'; step = saved['step_index']
            prompt = row['prompt']
        else:
            if row['label_type'] != 'clean' or not row.get('image'):
                raise ValueError('controlled visual states require verified clean image labels')
            x0 = self.encode_image(row['image'], self.config['posterior_seed'])
            x = controlled_noise(x0, t, noise_seed)
            source = 'controlled'; step = None
            prompt = self.config['neutral_prompt']
        flow, lengths, positions = self.flow_inputs(prompt)
        metadata = dict(state_source=source, t=t, step_index=step, noise_seed=noise_seed,
                        posterior_seed=self.config['posterior_seed'], x_t_hash=tensor_hash(x),
                        prompt=prompt, label_type=row['label_type'])
        return x, flow, lengths, positions, metadata

    @torch.no_grad()
    def velocity(self, x, t, flow):
        timestep = torch.full((len(x),), t, device=self.device)
        # Save the conditional prediction BEFORE native CFG/renormalization.
        conditional = []
        def head_hook(module, args, output):
            if not conditional:
                conditional.append(output[flow['packed_vae_token_indexes']].detach().clone())
        handle = self.model.llm2vae.register_forward_hook(head_hook)
        try:
            with autocast_for(self.device):
                final = self.model._forward_flow(x_t=x, timestep=timestep, **flow,
                    cfg_text_scale=self.config['cfg_text_scale'], cfg_img_scale=1.,
                    cfg_renorm_min=self.config['cfg_renorm_min'], cfg_renorm_type=self.config['cfg_renorm_type'])
        finally:
            handle.remove()
        if not torch.isfinite(final).all():
            raise FloatingPointError('non-finite native velocity')
        return conditional[0], final

    def visual_source(self, x, t, flow):
        if int(flow['packed_position_ids'].max()) >= self.config['writer_position_start']:
            raise ValueError('GEN source positions overlap writer slots')
        with capture_gen(self.decoder, flow['past_key_values']) as captured:
            self.velocity(x, t, flow)
        return tuple(captured[i] for i in range(len(self.decoder.layers)))

    def memory(self, source, source_off=False):
        synchronize(self.device)
        start_time = time.perf_counter()
        # Place writer slots after the largest native source RoPE position.
        # Callers supply positions explicitly from source preparation/capture.
        positions = torch.arange(self.config['writer_position_start'],
                                 self.config['writer_position_start']+self.writer.slots, device=self.device)
        with autocast_for(self.device):
            result = self.writer(self.seeds, source, positions, source_off)
        synchronize(self.device)
        self.last_writer_seconds = time.perf_counter()-start_time
        return result

    def caption_prefix(self, bank):
        return consume_bank(self.decoder, bank, [0])

    @torch.no_grad()
    def gen_arm(self, x, t, flow, bank, lengths, positions, masked=False):
        with gen_memory_call(self.decoder, flow['past_key_values'], bank, lengths, positions, masked):
            return self.velocity(x, t, flow)

    @torch.no_grad()
    def image(self, x):
        size = self.config['resolution']; p = self.model.latent_patch_size
        h = w = size//self.model.latent_downsample; c = self.model.latent_channel
        latent = torch.einsum('hwpqc->chpwq', x.reshape(h, w, p, p, c)).reshape(1, c, h*p, w*p)
        with autocast_for(self.device):
            image = self.bundle.vae.decode(latent)
        array = ((image[0].float().clamp(-1, 1)+1)*127.5).permute(1, 2, 0).cpu().byte().numpy()
        return Image.fromarray(array)

    @torch.no_grad()
    def continue_once(self, x, t, final_velocity, flow, remaining_times):
        """Inject only the selected node; subsequent calls are native."""
        times = [t] + list(remaining_times)
        if times[-1] != 0 or any(a <= b for a, b in zip(times, times[1:])):
            raise ValueError('continuation requires actual decreasing native times ending at zero')
        state = x.clone()
        for index, (current, following) in enumerate(zip(times, times[1:])):
            velocity = final_velocity if index == 0 else self.velocity(state, current, flow)[1]
            state = state - (current-following)*velocity
        return state
