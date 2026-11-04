"""Native GEN input/output transforms and temporary UND layer cache."""

from types import SimpleNamespace
import torch
from torch import nn
from .runtime import autocast


class LayerKV:
    """One layer's fresh, graph-connected native keys and values."""

    def __init__(self, index):
        self.layers = {index: SimpleNamespace(keys=None, values=None)}

    def update(self, keys, values, index, cache_kwargs=None):
        self.layers[index].keys = keys
        self.layers[index].values = values
        return keys, values


class NativeMoTField(nn.Module):
    def gen_inputs(self, z, t):
        rt, m = self.rt, self.model
        gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
        grid = torch.tensor([[gh, gw]], device=z.device)
        pixels = m.patchify(z, m.patch_size, channel_first=True)
        pixels = pixels.to(
            m.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding.weight.dtype
        )
        hidden = m.extract_feature(
            pixels.reshape(gh * gw, -1), gen_model=True, grid_hw=grid
        ).reshape(1, rt.tokens, -1)
        expanded = torch.as_tensor(t, device=z.device).reshape(()).expand(rt.tokens)
        condition = m.fm_modules["timestep_embedder"](expanded).reshape_as(hidden)
        if m.add_noise_scale_embedding:
            sigma = torch.full_like(
                expanded, rt.noise_scale() / m.noise_scale_max_value
            )
            condition = condition + m.fm_modules["noise_scale_embedder"](
                sigma
            ).reshape_as(hidden)
        return hidden + condition

    def head(self, hidden, t):
        rt, m = self.rt, self.model
        if m.use_pixel_head:
            image = hidden.reshape(1, *rt.grid, -1).permute(0, 3, 1, 2).contiguous()
            return m.fm_modules["fm_head"](image)
        if m.use_deep_fm_head:
            clean = m.fm_modules["fm_head"](
                hidden.reshape(rt.tokens, -1), t.expand(rt.tokens)
            ).reshape(1, rt.tokens, -1)
        else:
            clean = m.fm_modules["fm_head"](hidden)
        return m.unpatchify(clean, rt.patch, rt.height, rt.width)

    @torch.no_grad()
    def prefill(self, draft, prompt):
        with autocast(self.rt.device):
            embeds, indexes, mask, gen_indexes = self.prefix_inputs(draft, prompt)
            cache, _ = self.model._it2i_prefix_forward(
                embeds, indexes, {"full_attention": mask}
            )
        # Every sketch, text and generation token is visible in this branch.
        mask = torch.zeros(
            (1, 1, 1, embeds.shape[1] + self.rt.tokens), device=embeds.device
        )
        return cache, gen_indexes, mask
