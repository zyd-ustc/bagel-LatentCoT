"""Native GEN remains untouched; a spatial UND expert predicts one velocity correction."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .attention import rope_for, spatial_block
from .numerics import autocast_for


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, hidden):
        return self.base(hidden)+F.linear(F.linear(hidden, self.a.to(hidden.dtype)), self.b.to(hidden.dtype))


class UNDLayer(nn.Module):
    """Copy only UND modules, never duplicate GEN experts or share mutable weights."""
    def __init__(self, native, rank):
        super().__init__()
        self.input_layernorm = copy.deepcopy(native.input_layernorm)
        self.post_attention_layernorm = copy.deepcopy(native.post_attention_layernorm)
        self.mlp = copy.deepcopy(native.mlp)
        original = native.self_attn
        self.self_attn = nn.Module()
        for name in ('num_heads', 'num_key_value_heads', 'head_dim'):
            setattr(self.self_attn, name, getattr(original, name))
        for name in ('q_norm', 'k_norm'):
            setattr(self.self_attn, name, copy.deepcopy(getattr(original, name)))
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self.self_attn, name, LoRALinear(copy.deepcopy(getattr(original, name)), rank))


class SpatialCorrector(nn.Module):
    def __init__(self, runtime, config):
        super().__init__()
        # Runtime is deliberately not a child module: checkpoints/optimizer contain only the corrector.
        object.__setattr__(self, 'runtime', runtime)
        self.config = dict(config)
        self.layers = nn.ModuleList(UNDLayer(layer, config['lora_rank']) for layer in runtime.decoder.layers)
        self.norm = copy.deepcopy(runtime.decoder.norm)
        width = runtime.decoder.config.hidden_size
        self.input_projection = nn.Linear(width, width, bias=False, device=runtime.device, dtype=torch.float32)
        nn.init.eye_(self.input_projection.weight)
        channels = runtime.model.latent_channel*runtime.model.latent_patch_size**2
        self.head = nn.Linear(width, channels, bias=True, device=runtime.device, dtype=torch.float32)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        if config.get('correction_space')=='post_cfg':
            from .calibration import GlobalCalibration
            from .sampling import times, active_steps
            self.calibration=GlobalCalibration(times(config)[:active_steps(config)],channels,runtime.device)

    def forward(self, proposal, t, flow, context):
        model, decoder = self.runtime.model, self.runtime.decoder
        if len(context) != len(self.layers): raise ValueError('Context must cover all UND layers')
        with autocast_for(self.runtime.device):
            # Native frozen input embedding, with spatial layout and original noise time retained.
            hidden = model.vae2llm(proposal.detach().to(next(model.vae2llm.parameters()).dtype))
            hidden = self.input_projection(hidden)
            hidden = hidden+model.time_embedder(torch.full((len(hidden),), t, device=hidden.device))
            hidden = hidden+model.latent_pos_embed(flow['packed_vae_position_ids'])
            hidden = hidden.to(next(decoder.parameters()).dtype)
            positions = flow['packed_position_ids'][flow['packed_vae_token_indexes']]
            rope = rope_for(decoder, hidden, positions)
            for layer, source in zip(self.layers, context):
                if self.config['activation_checkpointing'] and torch.is_grad_enabled():
                    def block(value, layer=layer, source=source):
                        return spatial_block(layer, value, rope, source)
                    hidden = checkpoint(block, hidden, use_reentrant=False)
                else:
                    hidden = spatial_block(layer, hidden, rope, source)
            hidden = self.norm(hidden)
        # FP32 dense readout avoids tiny corrections disappearing into a BF16 velocity cast.
        with torch.autocast(self.runtime.device.type, enabled=False):
            return self.head(hidden.float())

    def trainable(self):
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    def group_parameters(self):
        return {'und': [p for name,p in self.trainable().items() if name.startswith('layers.')],
                'input': list(self.input_projection.parameters()), 'head': list(self.head.parameters())}


def correction_loss(delta, base, x0, epsilon):
    target = epsilon.float()-x0.float()-base.detach().float()
    return F.mse_loss(delta.float(), target)


def proposal_from(x_t, t, base):
    return (x_t.float()-t*base.detach().float()).detach()
