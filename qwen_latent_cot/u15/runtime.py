"""Upstream SenseNova model, native cache/THW/head, explicit SDPA on Ascend."""
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import subprocess
import sys
import torch
from ..bagel.accelerator import set_device


def upstream(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source/'src'))
    module = importlib.import_module('sensenova_u1')
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError('A different sensenova_u1 package is already imported')
    module.set_attn_backend('sdpa')
    return module


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16) if device.type != 'cpu' else nullcontext()


@dataclass
class Prefix:
    cache: object
    indexes: torch.Tensor
    hidden: torch.Tensor


@dataclass
class Prediction:
    velocity: torch.Tensor
    observation: tuple[torch.Tensor, ...] | None
    clean_prediction: torch.Tensor | None = None


class Runtime:
    def __init__(self, model, tokenizer, config):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer, self.config = tokenizer, dict(config)
        self.device = next(model.parameters()).device
        self.decoder = model.language_model.model
        self.patch = int(model.patch_size / model.downsample_ratio)
        self.width, self.height = config['width'], config['height']
        if self.width % self.patch or self.height % self.patch:
            raise ValueError('Resolution must divide the native patch edge')
        self.grid = (self.height//self.patch, self.width//self.patch)
        self.tokens = math.prod(self.grid)
        self.model.config.t_eps = config['t_eps']
        self.identity = None

    @classmethod
    def load(cls, model_path, source, config, device='npu:0', gen_weights=None):
        native = upstream(source)
        from transformers import AutoTokenizer
        device = set_device(device)
        dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
        model, info = native.NEOChatModel.from_pretrained(model_path, torch_dtype=dtype,
            output_loading_info=True, low_cpu_mem_usage=True, local_files_only=True)
        if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
            raise ValueError('Native checkpoint inventory differs: '+str(info))
        if model.config.is_moe_llm: raise ValueError('First migration targets the dense U1.5 8B model')
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = model.to(device)
        result = cls(model, tokenizer, config)
        revision_file=Path(source)/'SOURCE_REVISION'
        revision = revision_file.read_text().strip() if revision_file.exists() else subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
        result.identity = dict(model=str(Path(model_path).resolve()), source_revision=revision,
                               model_config=__import__('json').loads((Path(model_path)/'config.json').read_text()),
                               npu_patch=(Path(source)/'NPU_PATCH').read_text().strip() if (Path(source)/'NPU_PATCH').exists() else 'unpatched', gen_weights=None)
        if gen_weights:
            saved = torch.load(gen_weights, map_location='cpu', weights_only=False)
            if saved['kind'] != 'u15_portable_v1' or saved['stage'] != 't2i': raise ValueError('Requires U1.5 GEN export')
            if saved['identity'] != result.identity: raise ValueError('GEN export has a different base')
            result.apply_gen(saved['weights'])
            result.identity['gen_weights'] = str(Path(gen_weights).resolve())
        return result

    def apply_gen(self, weights):
        expected = {n:p for n,p in self.model.named_parameters() if gen_parameter(n)}
        if set(expected) != set(weights): raise ValueError('GEN parameter inventory differs')
        with torch.no_grad():
            for n,p in expected.items(): p.copy_(weights[n].to(p))

    @torch.no_grad()
    def prefix(self, prompt):
        from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN
        suffix = '<think>\n\n</think>\n\n<img>' if prompt is not None else '<img>'
        kwargs = dict(system_message=SYSTEM_MESSAGE_FOR_GEN) if prompt is not None else {}
        query = self.model._build_t2i_query(prompt or '', append_text=suffix, **kwargs)
        ids, positions, mask = self.model._build_t2i_text_inputs(self.tokenizer, query)
        with autocast(self.device):
            cache, hidden = self.model._t2i_prefix_forward(ids, positions, mask)
        indexes = self.model._build_t2i_image_indexes(*self.grid, positions.shape[1], device=self.device)
        return Prefix(cache, indexes, hidden.detach())

    def noise_scale(self):
        model = self.model; sigma = float(model.noise_scale)
        if model.noise_scale_mode in ('resolution','dynamic','dynamic_sqrt'):
            sigma *= math.sqrt(self.tokens/float(model.noise_scale_base_image_seq_len))
            if model.noise_scale_mode == 'dynamic_sqrt': sigma = math.sqrt(sigma)
        return min(sigma, float(model.noise_scale_max_value))

    def times(self):
        t = torch.linspace(0, 1, self.config['num_steps']+1, device=self.device)
        return self.model._apply_time_schedule(t, self.tokens, self.config['timestep_shift'])

    def predict(self, image, t, prefix, capture=False):
        model = self.model; batch = len(image); gh, gw = self.height//model.patch_size, self.width//model.patch_size
        grid = torch.tensor([[gh,gw]]*batch, device=self.device)
        latent = model.patchify(image, self.patch)
        pixels = model.patchify(image, model.patch_size, channel_first=True)
        observation = []; clean_prediction = []
        handles = [layer.input_layernorm_mot_gen.register_forward_pre_hook(
            lambda module, args: observation.append(args[0].detach())) for layer in self.decoder.layers] if capture else []
        if capture:
            handles.append(model.fm_modules['fm_head'].register_forward_hook(
                lambda module,args,value: clean_prediction.append(value.detach())))
        try:
            with autocast(self.device):
                hidden = model.extract_feature(pixels.reshape(batch*gh*gw,-1), gen_model=True, grid_hw=grid).reshape(batch,self.tokens,-1)
                expanded = torch.as_tensor(t, device=self.device).expand(batch*self.tokens)
                condition = model.fm_modules['timestep_embedder'](expanded).reshape_as(hidden)
                if model.add_noise_scale_embedding:
                    scale = torch.full_like(expanded, self.noise_scale()/model.noise_scale_max_value)
                    condition = condition+model.fm_modules['noise_scale_embedder'](scale).reshape_as(hidden)
                velocity = model._t2i_predict_v(hidden+condition, prefix.indexes,
                    {'full_attention':None}, prefix.cache, torch.as_tensor(t,device=self.device),
                    latent, image_token_num=self.tokens, timestep_embeddings=condition,
                    image_size=(self.width,self.height))
            velocity = model.unpatchify(velocity, self.patch, self.height, self.width)
            clean = clean_prediction[0] if capture else None
            if clean is not None and not model.use_pixel_head:
                clean = model.unpatchify(clean,self.patch,self.height,self.width)
            return Prediction(velocity, tuple(observation) if capture else None, clean)
        finally:
            for handle in handles: handle.remove()

    def guided(self, conditional, reference, scale):
        from .objective import cfg_velocity
        value = cfg_velocity(conditional, reference, scale, self.config['cfg_norm'],
                             lambda x: self.model.patchify(x,self.patch))
        return self.model.unpatchify(value,self.patch,self.height,self.width) if value.ndim == 3 else value


def gen_parameter(name):
    return '_mot_gen' in name or name.startswith('fm_modules.')
