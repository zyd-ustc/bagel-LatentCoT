"""Batch-one BAGEL diagnostic orchestration with effective-batch accumulation."""
from pathlib import Path
import time
import torch
from PIL import Image
from .memory import MemoryWriter, neutral_seeds, consume_bank
from .native import text_source, device_inputs, empty_cache, capture_gen, gen_memory_call
from .data import controlled_noise, tensor_hash, file_hash
from .schedule import native_suffix
from ..bagel.accelerator import seeded_context, synchronize
from .numerics import autocast_for


class DiagnosticRuntime:
    def __init__(self, bundle, config, topology):
        for key, expected in dict(adapter_type='linear_low_rank_v1',
                                  training_state_sampling='independent_uniform_v1',
                                  posterior_seed_policy='image_content_hash_v1',
                                  fact_scoring='locked_literal_assertion_v2').items():
            if config.get(key) != expected:
                raise ValueError('unsupported diagnostic policy: '+key)
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
        self.instruction_ids = self.prompt_ids(config['caption_instruction'])
        self.c0_instruction_ids = self.prompt_ids(config['c0_instruction'])
        self._neutral_context = None

    def prompt_ids(self, instruction):
        # Use exactly the native tokenizer call and BOS/EOS wrapper.
        inputs, _, _ = self.model.prepare_prompts([0], [0], [instruction],
                                                  self.bundle.tokenizer, self.bundle.token_ids)
        return inputs['packed_text_ids'].tolist()

    def image_posterior_seed(self, path):
        import hashlib
        digest = hashlib.sha256(f"{self.config['posterior_seed']}:{file_hash(path)}".encode()).digest()
        return int.from_bytes(digest[:8], 'big') % (2**63-1)

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
            if (saved.get('sampler_steps'), saved.get('timestep_shift')) != (self.config['sampler_steps'], self.config['timestep_shift']):
                raise ValueError('trajectory native sampler grid differs from runtime configuration')
            x = saved['x_t'].to(self.device)
            source = 'trajectory'; step = saved['step_index']; posterior_seed = None
            prompt = row['prompt']
        else:
            if row['label_type'] != 'clean' or not row.get('image'):
                raise ValueError('controlled visual states require verified clean image labels')
            posterior_seed = self.image_posterior_seed(row['image'])
            x0 = self.encode_image(row['image'], posterior_seed)
            x = controlled_noise(x0, t, noise_seed)
            source = 'controlled'; step = None
            prompt = self.config['neutral_prompt']
        flow, lengths, positions = self.flow_inputs(prompt)
        metadata = dict(state_source=source, t=t, step_index=step, noise_seed=noise_seed,
                        posterior_seed=posterior_seed, posterior_seed_base=self.config['posterior_seed'],
                        posterior_seed_policy=self.config['posterior_seed_policy'], x_t_hash=tensor_hash(x),
                        prompt=prompt, label_type=row['label_type'])
        return x, flow, lengths, positions, metadata

    @torch.no_grad()
    def training_state(self, row, t, noise_seed):
        """Keep sampled epsilon/x0; trajectory states deliberately have no flow target."""
        from .flow_training import TrainingState
        if row.get('state'):
            x, flow, lengths, positions, meta = self.state(row, t, noise_seed)
            x0 = epsilon = None
            saved = torch.load(row['state'], map_location='cpu', weights_only=True)
            if saved.get('checkpoint_hash') != row.get('trajectory_checkpoint_hash'):
                raise ValueError('trajectory checkpoint provenance differs from label')
        else:
            posterior_seed = self.image_posterior_seed(row['image'])
            x0 = self.encode_image(row['image'], posterior_seed)
            generator = torch.Generator(device='cpu').manual_seed(noise_seed)
            epsilon = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device, x0.dtype)
            x = epsilon if t == 1 else (1-t)*x0+t*epsilon
            if self.config.get('source_prompt_policy', 'neutral') == 'neutral':
                prompt = self.config['neutral_prompt']
            elif self.config.get('source_prompt_policy') == 'deployment' and self.config.get('allow_target_prompt'):
                prompt = row['prompt']
            else:
                raise ValueError('deployment/target prompt requires explicit allow_target_prompt')
            flow, lengths, positions = self.flow_inputs(prompt)
            meta = dict(state_source='controlled_paired', t=t, noise_seed=noise_seed,
                        posterior_seed=posterior_seed, posterior_seed_policy=self.config['posterior_seed_policy'],
                        x_t_hash=tensor_hash(x), image_hash=file_hash(row['image']), prompt=prompt)
        if row.get('label_kind') in ('observation_proxy', 'target_observation_gap'):
            if abs(float(row['t'])-t) > 1e-7 or row['noise_seed'] != noise_seed:
                raise ValueError('observation label must match actual state t/noise seed')
            if row['observation_state_hash'] != tensor_hash(x):
                raise ValueError('observation label belongs to a different x_t')
            if row.get('observation_image_hash') and meta.get('image_hash') != row['observation_image_hash']:
                raise ValueError('observation image provenance differs')
        meta.update(label_kind=row.get('label_kind', 'target_clean'), round=0,
                    confirmed_label=row.get('confirmed_label', row.get('verified', False)))
        return TrainingState(x, x0, epsilon, float(t), flow, lengths, positions, meta)

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
        expected = native_suffix(self.config, t)
        if len(expected) != len(remaining_times) or any(abs(a-b) > 1e-7 for a, b in zip(expected, remaining_times)):
            raise ValueError('continuation must use the complete native sampler suffix')
        times = [t] + expected
        if times[-1] != 0 or any(a <= b for a, b in zip(times, times[1:])):
            raise ValueError('continuation requires actual decreasing native times ending at zero')
        state = x.clone()
        for index, (current, following) in enumerate(zip(times, times[1:])):
            velocity = final_velocity if index == 0 else self.velocity(state, current, flow)[1]
            state = state - (current-following)*velocity
        return state
