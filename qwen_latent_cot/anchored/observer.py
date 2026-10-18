"""Side-branch UND read of GEN image KV. It never enters GEN attention."""
from __future__ import annotations

import torch
from torch import nn

from ..diagnostics.memory import rope_for, und_block
from .native_path import BranchContext, sample_groups


OBSERVER_SPEC = {
    "name": "verified_gen_hidden_to_und",
    "source_policy": "image_only",
    "expert": "gen",
    "prompt_kv_in_source": False,
    "writes_gen_attention": False,
    "same_layer": True,
    "rotated_source": True,
}


class VerifiedUndObserver(nn.Module):
    """Read image-only GEN KV with frozen UND blocks and return slot features.

    The source is the verified ``capture_gen(..., source_policy='image_only')``
    record: GEN expert, same-layer, rotated, image tokens only. Query slots live
    in this side branch. They are not added to the GEN token list.
    """

    def __init__(self, decoder, slots: int, source_layers: tuple[int, ...], position_start: int):
        super().__init__()
        object.__setattr__(self, "decoder", decoder)
        self.slots = slots
        self.source_layers = tuple(source_layers)
        self.position_start = position_start

    def zeros(self, samples: int, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros((samples, self.slots, like.shape[-1]))

    def read(self, hidden: torch.Tensor, captured: dict, branch: BranchContext) -> torch.Tensor:
        if captured is None:
            raise ValueError("the UND observer needs image KV captured during the body")
        if tuple(sorted(captured)) != self.source_layers:
            raise ValueError("captured GEN layers do not match the fixed observer span")
        groups = sample_groups(branch, hidden.shape[0])
        samples = len(groups)
        queries = hidden.new_zeros((samples * self.slots, hidden.shape[-1]))
        for sample, group in enumerate(groups):
            pooled = hidden[group["vae"]].mean(dim=0)
            queries[sample * self.slots:(sample + 1) * self.slots] = pooled
        positions = torch.arange(
            self.position_start, self.position_start + self.slots, device=hidden.device,
        ).repeat(samples)
        rope = rope_for(self.decoder, queries, positions)
        lengths = (self.slots,) * samples
        query_positions = set(positions.tolist())
        for index in self.source_layers:
            source = captured[index]
            if source.expert != "gen" or not source.rotated:
                raise ValueError("observer source must be rotated GEN KV")
            if tuple(source.token_roles) != ("image",) * len(source.keys):
                raise ValueError("observer source must be image tokens only")
            if not source.positions or set(source.positions) & query_positions:
                raise ValueError("GEN source positions are missing or overlap observer slots")
            if sum(source.lengths) != len(branch.packed_vae_token_indexes):
                raise ValueError("observer source length is not the image-token count")
            queries = und_block(
                self.decoder.layers[index], queries, rope, lengths, source, causal=False,
            )
        return queries.view(samples, self.slots, hidden.shape[-1])
