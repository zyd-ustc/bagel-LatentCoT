"""Frozen UND cache; GEN-only LoRA residual against an identical frozen sketch path."""

import copy
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .native_mot import NativeMoTField
from .runtime import autocast


class GENLoRA(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        # Native weights are owned solely by Runtime, never optimizer/checkpoint.
        object.__setattr__(self, "base", base)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))


def module_view(module):
    view = copy.copy(module)
    view._modules = module._modules.copy()
    return view


class SketchResidualCorrector(NativeMoTField):
    def __init__(self, rt, cfg):
        # Reuse native prefix/gen/head utilities, but never its adapter installer.
        nn.Module.__init__(self)
        object.__setattr__(self, "rt", rt)
        object.__setattr__(self, "model", rt.model)
        self.config = cfg
        rt.model.eval().requires_grad_(False)
        stop = len(rt.decoder.layers)-cfg["freeze_gen_tail"]
        if not 0 < stop <= len(rt.decoder.layers) or cfg["lora_rank"] < 1:
            raise ValueError("Require nonempty GEN LoRA layers and positive rank")
        self.adapters = nn.ModuleDict()
        views = []
        for index, original in enumerate(rt.decoder.layers):
            if index >= stop:
                views.append(original)
                continue
            layer = module_view(original)
            layer.self_attn = module_view(original.self_attn)
            layer.mlp_mot_gen = module_view(original.mlp_mot_gen)
            for group, names in (
                ("self_attn", ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")),
                ("mlp_mot_gen", ("gate_proj", "up_proj", "down_proj")),
            ):
                owner = getattr(layer, group)
                for name in names:
                    adapter = GENLoRA(getattr(owner, name), cfg["lora_rank"])
                    self.adapters[f"layer_{index}_{name}"] = adapter
                    setattr(owner, name, adapter)
            views.append(layer)
        # Views share frozen tensors but never replace modules in rt.model.
        object.__setattr__(self, "adapted_layers", tuple(views))
        width = rt.decoder.norm_mot_gen.weight.numel()
        self.start_embedding = nn.Linear(1, width, bias=False, device=rt.device)
        nn.init.zeros_(self.start_embedding.weight)
        self.proposal_proj = nn.Linear(width, width, bias=False, device=rt.device)
        nn.init.zeros_(self.proposal_proj.weight)
        self.layer_count = stop
        self.calls = 0

    def prefix_inputs(self, draft, prompt):
        rt, m, tok = self.rt, self.model, self.rt.tokenizer
        from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN
        m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
        mean = draft.new_tensor([.485,.456,.406])[None,:,None,None]
        std = draft.new_tensor([.229,.224,.225])[None,:,None,None]
        rgb = ((draft.clamp(-1,1)+1)/2-mean)/std
        gh,gw = rt.height//m.patch_size,rt.width//m.patch_size
        pixels = m.patchify(rgb,m.patch_size,channel_first=True).reshape(gh*gw,-1)
        pixels = pixels.to(m.vision_model.embeddings.patch_embedding.weight.dtype)
        grid = torch.tensor([[gh,gw]],device=rt.device)
        image = "<img>"+"<IMG_CONTEXT>"*rt.tokens+"</img>"
        query = m._build_t2i_query(image+"\n"+prompt,system_message=SYSTEM_MESSAGE_FOR_GEN,
                                  append_text="<think>\n\n</think>\n\n<img>")
        embeds,indexes,mask = m._build_it2i_inputs(tok,query,pixels,grid)
        gi = m._build_t2i_image_indexes(*rt.grid,int(indexes[0].max())+1,device=rt.device)
        return embeds,indexes,mask["full_attention"],gi

    @torch.no_grad()
    def proposal_features(self, image):
        rt,m = self.rt,self.model
        gh,gw = rt.height//m.patch_size,rt.width//m.patch_size
        pixels = m.patchify(image,m.patch_size,channel_first=True).reshape(gh*gw,-1)
        pixels = pixels.to(m.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding.weight.dtype)
        with autocast(rt.device):
            return m.extract_feature(pixels,gen_model=True,
                grid_hw=torch.tensor([[gh,gw]],device=rt.device)).reshape(1,rt.tokens,-1)

    def block(self, index, hidden, prefix, adapted):
        cache, indexes, mask = prefix
        layers = self.adapted_layers if adapted else self.rt.decoder.layers
        return layers[index](
            hidden, indexes=indexes,
            image_gen_indicators=torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool),
            exist_non_image_gen_tokens=False, exist_image_gen_tokens=True,
            attention_mask=mask, past_key_values=cache, use_cache=True, update_cache=False)

    def sketch_velocity(self, z, t, tau, prefix, adapted, proposal=None):
        self.calls += 1
        with autocast(self.rt.device):
            hidden = self.gen_inputs(z, t)
            if adapted:
                hidden = hidden + self.start_embedding(
                    torch.as_tensor(tau, device=z.device, dtype=self.start_embedding.weight.dtype).reshape(1,1,1))
                hidden = hidden + self.proposal_proj(proposal.to(self.proposal_proj.weight.dtype)).to(hidden.dtype)
            # Frozen theta_0 has the same tau input and a zero start embedding.
            for index in range(len(self.rt.decoder.layers)):
                block = partial(self.block, index, prefix=prefix, adapted=adapted)
                if self.config["activation_checkpointing"] and torch.is_grad_enabled():
                    hidden = checkpoint(block, hidden, use_reentrant=False)
                else:
                    hidden = block(hidden)
            clean = self.head(self.rt.decoder.norm_mot_gen(hidden), t)
        return (clean.float()-z.float())/(1-t).clamp_min(self.config["t_eps"])

    def forward(self, z, t, tau, sketch, proposal):
        features = self.proposal_features(proposal.detach())
        # For an on-policy rollout keep the reference derivative w.r.t. z.
        # Frozen parameters never get gradients, including UND K/V and GEN norms.
        with torch.set_grad_enabled(torch.is_grad_enabled() and z.requires_grad):
            reference = self.sketch_velocity(z, t, tau, sketch, adapted=False)
        adapted = self.sketch_velocity(z, t, tau, sketch, adapted=True, proposal=features)
        return adapted-reference
