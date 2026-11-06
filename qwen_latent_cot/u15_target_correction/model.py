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

    def _pixels(self, image):
        """ImageNet-normalize the protocol canvas. Never round-trip through PNG."""
        from sensenova_u1.models.neo_unify.utils import IMAGENET_MEAN, IMAGENET_STD, preprocess_pixel_values

        rt, m = self.rt, self.model
        if tuple(image.shape[-2:]) != (rt.height, rt.width):
            raise ValueError("Condition image must already use the protocol canvas")
        rgb = (image.detach().float().clamp(-1, 1) + 1) * 0.5
        mean = rgb.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = rgb.new_tensor(IMAGENET_STD).view(3, 1, 1)
        pixels, grid = preprocess_pixel_values((rgb[0] - mean) / std, patch_size=m.patch_size)
        dtype = torch.bfloat16 if rt.device.type != "cpu" else torch.float32
        return pixels.to(device=rt.device, dtype=dtype), grid.to(rt.device)

    def _query(self, instruction, grid, *, image_only):
        from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

        m, tok = self.model, self.rt.tokenizer
        m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
        prompt = "<image>" if image_only else "<image>\n" + instruction
        if image_only:
            query = m._build_t2i_query(prompt, append_text="<img>")
        else:
            query = m._build_t2i_query(
                prompt,
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text="<think>\n\n</think>\n\n<img>",
            )
        count = int(grid[0, 0] * grid[0, 1] * m.downsample_ratio**2)
        tokens = "<img>" + "<IMG_CONTEXT>" * count + "</img>"
        return query.replace("<image>", tokens, 1)

    def _it2i_inputs(self, query, vit, grid):
        from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

        m, tok = self.model, self.rt.tokenizer
        encoded = tok(query, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.rt.device)
        indexes = m.get_thw_indexes(input_ids[0], grid)
        attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
        embeds = m.language_model.get_input_embeddings()(input_ids)
        batch, length, channels = embeds.shape
        flat = input_ids.reshape(batch * length)
        selected = flat == m.img_context_token_id
        visual = vit.reshape(-1, channels)
        if int(selected.sum()) != visual.shape[0]:
            raise ValueError("Preview image tokens differ from the generation canvas")
        embeds = embeds.reshape(batch * length, channels).clone()
        embeds[selected] = visual.to(embeds.dtype)
        return embeds.reshape(batch, length, channels), indexes, attention_mask

    def _prefill(self, query, vit, grid):
        m = self.model
        embeds, indexes, mask = self._it2i_inputs(query, vit, grid)
        with autocast(self.rt.device):
            cache, _ = m._it2i_prefix_forward(embeds, indexes, mask)
            gen_index = m._build_t2i_image_indexes(
                *self.rt.grid, int(indexes[0].max()) + 1, device=self.rt.device
            )
        self.prefix_calls += 1
        return cache, gen_index, None

    @torch.no_grad()
    def _preview_vision(self, preview):
        pixels, grid = self._pixels(preview)
        expected = int(grid[0, 0] * grid[0, 1] * self.model.downsample_ratio**2)
        if expected != self.rt.tokens:
            raise ValueError("Condition image tokens differ from the generation canvas")
        with autocast(self.rt.device):
            vit = self.model.extract_feature(pixels, grid_hw=grid)
        self.prefix_vision_calls += 1
        return vit, grid

    @torch.no_grad()
    def prefixes(self, preview, edit_request, tail_condition, preserve=""):
        """Two UND prefills: image-only baseline and joint edit condition."""
        instruction = edit_prompt(edit_request, tail_condition, preserve)
        vit, grid = self._preview_vision(preview)
        caches = {}
        for name, image_only in (("image", True), ("joint", False)):
            query = self._query("" if image_only else instruction, grid, image_only=image_only)
            caches[name] = self._prefill(query, vit, grid)
        return caches

    @torch.no_grad()
    def image_prefix(self, preview):
        vit, grid = self._preview_vision(preview)
        query = self._query("", grid, image_only=True)
        return {"image": self._prefill(query, vit, grid)}

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

    def state_velocity(self, state, t, s, prefixes, base_prediction):
        """Displacement per unit edit progress: t * [H(q,t,s) - b0]."""
        clean = self.state_clean(state, t, s, prefixes["joint"])
        return (clean - base_prediction.float()) * float(t)
