"""Frozen native Pre / Body / Post path. Prompt KV and token layout stay put."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import LoopConfig


def resolve_decoder(module):
    """Accept a Qwen2 model or the causal LM that owns it."""
    if hasattr(module, "layers") and hasattr(module, "forward_inference_span"):
        return module
    inner = getattr(module, "model", None)
    if inner is not None and hasattr(inner, "layers") and hasattr(inner, "forward_inference_span"):
        return inner
    raise TypeError("Anchored loop needs the BAGEL Qwen2 decoder")


@dataclass
class BranchContext:
    """One CFG branch. Caches are not shared across branches."""

    name: str
    query_lens: torch.Tensor
    packed_query_position_ids: torch.Tensor
    packed_query_indexes: torch.Tensor
    past_key_values: object
    key_values_lens: torch.Tensor
    packed_key_value_indexes: torch.Tensor
    packed_text_indexes: torch.Tensor
    packed_vae_token_indexes: torch.Tensor
    is_causal: bool = False
    flow_time: float | None = None

    def __post_init__(self):
        if self.name not in ("cond", "cfg_text", "cfg_img"):
            raise ValueError("branch name must be cond, cfg_text, or cfg_img")
        if self.past_key_values is None:
            raise ValueError("each CFG branch needs its own KV cache object")


def sample_groups(branch: BranchContext, seq_len: int) -> list[dict]:
    """Split a packed GEN query into per-sample text and image indexes."""
    text = [int(i) for i in branch.packed_text_indexes.tolist()]
    vae = [int(i) for i in branch.packed_vae_token_indexes.tolist()]
    text_set, vae_set = set(text), set(vae)
    if text_set & vae_set:
        raise ValueError("text and image tokens overlap")
    if text_set | vae_set != set(range(seq_len)):
        raise ValueError("query tokens must stay the native text and image tokens")
    groups = []
    offset = 0
    for length in branch.query_lens.tolist():
        span = range(offset, offset + int(length))
        groups.append({
            "text": [i for i in span if i in text_set],
            "vae": [i for i in span if i in vae_set],
        })
        offset += int(length)
    if offset != seq_len:
        raise ValueError("query lengths do not cover the packed sequence")
    for group in groups:
        if not group["vae"]:
            raise ValueError("every sample needs image tokens")
    return groups


def snapshot_cache(cache):
    keys, values = [], []
    for index in range(cache.num_layers):
        key, value = cache.key_cache[index], cache.value_cache[index]
        keys.append(None if key is None else key.detach().clone())
        values.append(None if value is None else value.detach().clone())
    return keys, values


def caches_match(cache, keys, values) -> bool:
    for index in range(cache.num_layers):
        key, value = cache.key_cache[index], cache.value_cache[index]
        if keys[index] is None or values[index] is None:
            if key is not None or value is not None:
                return False
            continue
        if key is None or value is None:
            return False
        if key.shape != keys[index].shape or value.shape != values[index].shape:
            return False
        if not torch.equal(key, keys[index]) or not torch.equal(value, values[index]):
            return False
    return True


class NativePathWrapper:
    """Cache one RoPE and run Pre, shared Body, and Post without writing prompt KV."""

    def __init__(self, decoder, config: LoopConfig):
        self.decoder = decoder
        self.start = config.body_start
        self.end = config.body_end
        self.verify_prompt_cache = config.verify_prompt_cache
        self.body_calls = 0

    def rope(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        cos, sin = self.decoder.rotary_emb(hidden, position_ids.unsqueeze(0))
        return cos.squeeze(0), sin.squeeze(0)

    def _run(self, hidden, branch: BranchContext, embeddings, start: int, end: int, apply_final_norm: bool):
        if start == end and not apply_final_norm:
            return hidden
        output = self.decoder.forward_inference_span(
            packed_query_sequence=hidden,
            query_lens=branch.query_lens,
            packed_query_position_ids=branch.packed_query_position_ids,
            packed_query_indexes=branch.packed_query_indexes,
            past_key_values=branch.past_key_values,
            key_values_lens=branch.key_values_lens,
            packed_key_value_indexes=branch.packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=branch.is_causal,
            mode="gen",
            packed_vae_token_indexes=branch.packed_vae_token_indexes,
            packed_text_indexes=branch.packed_text_indexes,
            layer_start=start,
            layer_end=end,
            apply_final_norm=apply_final_norm,
            position_embeddings=embeddings,
        )
        return output.packed_query_sequence

    def run_span(self, hidden, branch, embeddings, start, end, apply_final_norm, captured=None):
        """Run one span and optionally collect its full-query GEN KV.

        Full-query keeps boundary tokens and image tokens. The caller stitches
        Pre, Body, and Post to cover every layer. An empty span records nothing.
        """
        if captured is None:
            return self._run(hidden, branch, embeddings, start, end, apply_final_norm)
        from ..diagnostics.native import capture_gen
        with capture_gen(
            self.decoder,
            branch.past_key_values,
            t=branch.flow_time,
            positions=branch.packed_query_position_ids,
            detach=not torch.is_grad_enabled(),
            source_policy="full_query",
            layers=range(start, end),
        ) as span:
            output = self._run(hidden, branch, embeddings, start, end, apply_final_norm)
        captured.update(span)
        return output

    def pre(self, hidden, branch, embeddings, captured=None):
        return self.run_span(hidden, branch, embeddings, 0, self.start, False, captured)

    def body(self, hidden, branch, embeddings, captured=None):
        self.body_calls += 1
        return self.run_span(hidden, branch, embeddings, self.start, self.end, False, captured)

    def post(self, hidden, branch, embeddings, captured=None):
        return self.run_span(hidden, branch, embeddings, self.end, len(self.decoder.layers), True, captured)

    def guard_layout(self, branch: BranchContext, hidden: torch.Tensor, positions, lengths, cache_snapshot):
        if hidden.shape[0] != positions.shape[0]:
            raise RuntimeError("sequence length changed inside the native path")
        if not torch.equal(branch.packed_query_position_ids, positions):
            raise RuntimeError("native position ids changed")
        if not torch.equal(branch.query_lens, lengths):
            raise RuntimeError("native token counts changed")
        if cache_snapshot is not None and not caches_match(branch.past_key_values, *cache_snapshot):
            raise RuntimeError("prompt KV cache was written or resized")


def pack_flow_hidden(
    model,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    packed_text_ids: torch.Tensor,
    packed_text_indexes: torch.Tensor,
    packed_vae_token_indexes: torch.Tensor,
    packed_vae_position_ids: torch.Tensor,
    packed_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Pack text and latent tokens the same way as ``Bagel._forward_flow``."""
    if timestep.unique().shape[0] != 1:
        raise ValueError("one flow time per packed forward")
    text = model.language_model.model.embed_tokens(packed_text_ids)
    packed = text.new_zeros((int(packed_seqlens.sum()), model.hidden_size))
    packed = packed.index_copy(0, packed_text_indexes, text)
    latent = model.vae2llm(x_t) + model.time_embedder(timestep) + model.latent_pos_embed(packed_vae_position_ids)
    if latent.dtype != packed.dtype:
        latent = latent.to(packed.dtype)
    return packed.index_copy(0, packed_vae_token_indexes, latent)


def velocity_from_hidden(hidden: torch.Tensor, vae_indexes: torch.Tensor, head=None) -> torch.Tensor:
    projected = hidden if head is None else head(hidden)
    return projected[vae_indexes]
