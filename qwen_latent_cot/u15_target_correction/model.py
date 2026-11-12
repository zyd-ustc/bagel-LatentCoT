"""GEN LoRA, edit-progress embedding, and the anchored state field."""

import copy
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .conditions import edit_prompt
from .runtime import autocast


class GENLoRA(nn.Module):
    """One-sided-random, one-sided-zero adapter on a frozen native linear layer."""

    def __init__(self, base, rank):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        native = self.base(x)
        update = F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))
        return native + update


def module_view(module):
    view = copy.copy(module)
    view._modules = module._modules.copy()
    return view


def _layer_view(original):
    """A shallow copy whose attention and GEN MLP can be wrapped independently."""
    view = module_view(original)
    view.self_attn = module_view(original.self_attn)
    view.mlp_mot_gen = module_view(original.mlp_mot_gen)
    return view


class StateEditor(nn.Module):
    """All GEN layers get LoRA. The state field is H(q,t,s) minus a frozen anchor.

    ``image_clean`` uses an independent unadapted layer view, so native T2I,
    IT2I, and the frozen anchor never share a runtime LoRA switch. It provides
    ``b0`` from the original anchor. ``state_velocity`` is the trainable joint
    branch and returns displacement per unit edit progress.
    """

    def __init__(self, rt, cfg):
        super().__init__()
        object.__setattr__(self, "rt", rt)
        object.__setattr__(self, "model", rt.model)
        self.config = cfg
        rt.model.eval().requires_grad_(False)
        if int(cfg["lora_rank"]) < 1:
            raise ValueError("LoRA rank must be positive")

        self.adapters = nn.ModuleDict()
        views, native_views = [], []
        for index, original in enumerate(rt.decoder.layers):
            layer = _layer_view(original)
            native = _layer_view(original)
            for group, names in (
                ("self_attn", ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")),
                ("mlp_mot_gen", ("gate_proj", "up_proj", "down_proj")),
            ):
                owner = getattr(layer, group)
                for name in names:
                    adapter = GENLoRA(getattr(owner, name), int(cfg["lora_rank"]))
                    self.adapters[f"layer_{index}_{name}"] = adapter
                    setattr(owner, name, adapter)
            views.append(layer)
            native_views.append(native)
        object.__setattr__(self, "adapted_layers", tuple(views))
        object.__setattr__(self, "native_layers", tuple(native_views))

        hidden = int(rt.model.fm_modules["timestep_embedder"].mlp[0].out_features)
        self.edit_s_embedder = type(rt.model.fm_modules["timestep_embedder"])(hidden)
        with torch.no_grad():
            self.edit_s_embedder.mlp[2].weight.zero_()
            self.edit_s_embedder.mlp[2].bias.zero_()
        self.edit_s_embedder.to(rt.device)

        self.calls = 0
        self.baseline_calls = 0
        self.state_calls = 0
        self.prefix_calls = 0
        self.prefix_vision_calls = 0

    def gen_inputs(self, z, t, s=None):
        rt, m = self.rt, self.model
        batch = z.shape[0]
        gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
        grid = torch.tensor([[gh, gw]] * batch, device=z.device)
        pixels = m.patchify(z, m.patch_size, channel_first=True)
        dtype = m.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding.weight.dtype
        pixels = pixels.to(dtype)
        hidden = m.extract_feature(
            pixels.reshape(batch * gh * gw, -1), gen_model=True, grid_hw=grid
        ).reshape(batch, rt.tokens, -1)

        expanded = torch.as_tensor(t, device=z.device).reshape(()).expand(batch * rt.tokens)
        condition = m.fm_modules["timestep_embedder"](expanded).reshape_as(hidden)
        if m.add_noise_scale_embedding:
            sigma = torch.full_like(expanded, rt.noise_scale() / m.noise_scale_max_value)
            condition = condition + m.fm_modules["noise_scale_embedder"](sigma).reshape_as(hidden)
        if s is not None:
            progress = torch.as_tensor(s, device=z.device, dtype=torch.float32)
            progress = progress.reshape(()).expand(batch * rt.tokens)
            condition = condition + self.edit_s_embedder(progress).reshape_as(hidden).to(hidden.dtype)
        return hidden + condition

    def head(self, hidden, t):
        rt, m = self.rt, self.model
        batch = hidden.shape[0]
        if m.use_pixel_head:
            image = hidden.reshape(batch, *rt.grid, -1).permute(0, 3, 1, 2).contiguous()
            return m.fm_modules["fm_head"](image)
        if m.use_deep_fm_head:
            moment = torch.as_tensor(t, device=hidden.device).reshape(()).expand(batch * rt.tokens)
            clean = m.fm_modules["fm_head"](
                hidden.reshape(batch * rt.tokens, -1), moment
            ).reshape(batch, rt.tokens, -1)
        else:
            clean = m.fm_modules["fm_head"](hidden)
        return m.unpatchify(clean, rt.patch, rt.height, rt.width)

    @torch.no_grad()
    def prefixes(self, edit_request):
        """Text-only UND prefills — the edit branch never sees a picture.

        ``image``  the unconditional reference the frozen H0 is measured against
                   (text-only T2I, the same path the deployment baseline uses);
        ``joint``  the short instruction, also text-only.

        Conditioning on a preview was the training-time luxury that made the
        editor rely on captions and lookaheads the closed loop cannot supply.
        The start state enters only as the denoising input ``a_t``.
        """
        instruction = edit_prompt(edit_request)
        return {
            "image": self._t2i_prefix(None),
            "joint": self._t2i_prefix(instruction),
        }

    def _t2i_prefix(self, prompt):
        """One text-only T2I prefill, shaped like ``prefixes`` entries."""
        prefix = self.rt.prefix(prompt)
        self.prefix_calls += 1
        return (prefix.cache, prefix.indexes, None)

    def _block(self, layers, index, hidden, prefix):
        cache, indexes, mask = prefix
        return layers[index](
            hidden,
            indexes=indexes,
            image_gen_indicators=torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool),
            exist_non_image_gen_tokens=False,
            exist_image_gen_tokens=True,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            update_cache=False,
        )

    def _clean(self, z, t, prefix, *, s=None, use_lora):
        self.calls += 1
        layers = self.adapted_layers if use_lora else self.native_layers
        with autocast(self.rt.device):
            hidden = self.gen_inputs(z, t, s)
            for index in range(len(layers)):
                block = partial(self._block, layers, index, prefix=prefix)
                if self.config["activation_checkpointing"] and torch.is_grad_enabled():
                    hidden = checkpoint(block, hidden, use_reentrant=False)
                else:
                    hidden = block(hidden)
            clean = self.head(self.rt.decoder.norm_mot_gen(hidden), t)
        return clean.float()

    @torch.no_grad()
    def image_clean(self, anchor, t, prefix):
        """Frozen native image-only clean prediction H0(a,t;P)."""
        self.baseline_calls += 1
        return self._clean(anchor, float(t), prefix, s=None, use_lora=False)

    def state_clean(self, state, t, s, prefix):
        """Joint clean prediction H(q,t,s;P,e,c_tail)."""
        self.state_calls += 1
        return self._clean(state, float(t), prefix, s=float(s), use_lora=True)

    def state_velocity(self, state, t, s, joint_prefix, base_prediction):
        """Displacement per unit edit progress: t * [H(q,t,s) - b0].

        ``joint_prefix`` is the already-selected tuple from ``prefixes()``;
        callers must not pass the whole image/joint dictionary here.
        """
        clean = self.state_clean(state, t, s, joint_prefix)
        return (clean - base_prediction.float()) * float(t)
