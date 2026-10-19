"""Frozen copy of the trained UND observation readout.

The C2 observation run proved a readout that starts from neutral-text
embeddings, attends at every layer to full-query GEN KV, and uses the trained
UND Q/K/V/O and Q/K norms. Those norms live on a copy. The native decoder that
generates keeps its own weights. Layernorms, MLP, and token embeddings were
not trained, so the reader uses those native modules directly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn

from ..diagnostics.memory import join_prefix, project, rope_for, rotate, und_block
from ..diagnostics.visual import observation_readout_keys
from ..diagnostics.visual_record import FORMAT_VERSION, STAGES
from .native_path import BranchContext, sample_groups
from .observation_source import require_source_mode


OBSERVER_SPEC = {
    "name": "trained_observation",
    "source_policy": "full_query",
    "layers": "all",
    "query": "neutral_seed_embeddings",
    "weights": "observation_checkpoint_copy",
    "prompt_kv_in_source": False,
    "writes_gen_attention": False,
    "same_layer": True,
    "rotated_source": True,
}


class _UndReadAttention(nn.Module):
    """UND projections only. GEN experts stay on the native decoder."""

    def __init__(self, source):
        super().__init__()
        self.layer_idx = source.layer_idx
        self.num_heads = source.num_heads
        self.num_key_value_heads = source.num_key_value_heads
        self.head_dim = source.head_dim
        self.q_proj = copy.deepcopy(source.q_proj)
        self.k_proj = copy.deepcopy(source.k_proj)
        self.v_proj = copy.deepcopy(source.v_proj)
        self.o_proj = copy.deepcopy(source.o_proj)
        self.q_norm = copy.deepcopy(source.q_norm)
        self.k_norm = copy.deepcopy(source.k_norm)


class _UndReadLayer(nn.Module):
    def __init__(self, source):
        super().__init__()
        # Untrained modules stay the native objects. Registering them here would
        # try to reparent them away from the generation decoder.
        object.__setattr__(self, "input_layernorm", source.input_layernorm)
        object.__setattr__(self, "post_attention_layernorm", source.post_attention_layernorm)
        object.__setattr__(self, "mlp", source.mlp)
        self.self_attn = _UndReadAttention(source.self_attn)


@dataclass(frozen=True)
class ReaderPrefix:
    """Hidden state entering the first GEN body layer after the native Pre KV.

    Later rounds reuse it because those Pre objects do not change. ``queries``
    is detached: the prefix has no ADSL parameters. ``native_boundary`` is the
    KV the first read attended at that layer.
    """

    queries: torch.Tensor
    positions: torch.Tensor
    lengths: tuple[int, ...]
    boundary: int
    native_boundary: object
    pre_ptrs: tuple[int, ...] = ()


def _seed_queries(embed, seed_token_ids, branch: BranchContext, slots: int, position_start: int):
    if seed_token_ids is None or len(seed_token_ids) != slots:
        raise RuntimeError("observation readout needs one neutral seed token per slot")
    sequence = int(branch.query_lens.sum())
    samples = len(sample_groups(branch, sequence))
    ids = torch.tensor(seed_token_ids, device=branch.query_lens.device, dtype=torch.long)
    queries = embed(ids.repeat(samples))
    positions = torch.arange(position_start, position_start + slots, device=queries.device).repeat(samples)
    return queries, positions, (slots,) * samples, samples, sequence


def _attend(layers, rope_decoder, queries, positions, lengths, captured, sequence: int, start: int, end: int):
    """Same-layer recurrence. ``start == end`` keeps ``queries`` unchanged."""
    query_positions = set(positions.tolist())
    for index in range(start, end):
        layer = layers[index]
        source = captured[index]
        if source.layer != index or source.expert != "mixed_und_gen" or not source.rotated:
            raise ValueError("observation source must be same-layer full-query rotated GEN KV")
        if not source.token_roles or set(source.token_roles) - {"boundary", "image"} or "image" not in source.token_roles:
            raise ValueError("observation source must be the GEN query, with image tokens and no prompt KV")
        if not source.positions or set(source.positions) & query_positions:
            raise ValueError("GEN source positions are missing or overlap observer slots")
        if sum(source.lengths) != sequence:
            raise ValueError("full-query source must cover the packed GEN query")
        rope = rope_for(rope_decoder, queries, positions)
        queries = und_block(layer, queries, rope, lengths, source, causal=False)
    return queries


def _require_full_capture(captured, depth: int) -> None:
    if captured is None or tuple(sorted(captured)) != tuple(range(depth)):
        raise ValueError("trained observation reads GEN KV at every layer")


def observation_slots(
    layers,
    embed,
    rope_decoder,
    seed_token_ids,
    captured,
    branch: BranchContext,
    slots: int,
    position_start: int,
) -> torch.Tensor:
    """Run the verified C2 observation recurrence and return the final slot hidden.

    The query is the neutral seed embedding, the tensor ``neutral_seeds``
    stores before layer 0. Every UND layer attends to that layer's full-query
    GEN KV. GEN body hidden is not an input, and the GEN body span is not the
    layer subset. A later ADSL round may pass recomputed Body and Post KV
    through this same recurrence; that source is a separate combination.
    """
    depth = len(layers)
    _require_full_capture(captured, depth)
    queries, positions, lengths, samples, sequence = _seed_queries(
        embed, seed_token_ids, branch, slots, position_start,
    )
    queries = _attend(layers, rope_decoder, queries, positions, lengths, captured, sequence, 0, depth)
    return queries.view(samples, slots, queries.shape[-1])


def observation_with_prefix(
    layers,
    embed,
    rope_decoder,
    seed_token_ids,
    captured,
    branch: BranchContext,
    slots: int,
    position_start: int,
    boundary: int,
) -> tuple[torch.Tensor, ReaderPrefix]:
    """Full recurrence, plus the hidden state entering ``boundary``.

    The returned slots match ``observation_slots``. ``boundary`` is the first
    GEN body layer. Layers before it attend only to native Pre KV.
    """
    depth = len(layers)
    if not 0 <= boundary < depth:
        raise ValueError("reader prefix boundary is outside the decoder")
    _require_full_capture(captured, depth)
    queries, positions, lengths, samples, sequence = _seed_queries(
        embed, seed_token_ids, branch, slots, position_start,
    )
    entering = _attend(layers, rope_decoder, queries, positions, lengths, captured, sequence, 0, boundary)
    final = _attend(layers, rope_decoder, entering, positions, lengths, captured, sequence, boundary, depth)
    prefix = ReaderPrefix(
        entering.detach(), positions.detach(), lengths, boundary, captured[boundary],
    )
    return final.view(samples, slots, final.shape[-1]), prefix


def observation_from_prefix(
    layers,
    rope_decoder,
    prefix: ReaderPrefix,
    captured,
    branch: BranchContext,
    slots: int,
    pre_ptrs: tuple[int, ...],
) -> torch.Tensor:
    """Continue a cached prefix. Layers before ``prefix.boundary`` do not run."""
    depth = len(layers)
    if tuple(pre_ptrs) != tuple(prefix.pre_ptrs):
        raise RuntimeError(
            "anchored observation prefix KV changed; the cached reader state belongs to the native Pre KV"
        )
    _require_full_capture(captured, depth)
    sequence = int(branch.query_lens.sum())
    samples = len(sample_groups(branch, sequence))
    if prefix.queries.shape[0] != samples * slots:
        raise ValueError("cached reader state does not match this packed branch")
    final = _attend(
        layers, rope_decoder, prefix.queries, prefix.positions, prefix.lengths,
        captured, sequence, prefix.boundary, depth,
    )
    return final.view(samples, slots, final.shape[-1])


def boundary_attention_entropy(layer, queries, rope_decoder, positions, lengths, source) -> float:
    """Mean attention entropy at one reader layer. A measurement, not a verdict.

    The scores follow the same Q/K, RoPE, source join, and head grouping as
    ``und_block``. Comparing the native and stitched sources at the first body
    layer shows whether that boundary jumps. It does not show that the stitch
    matches the C2 training distribution.
    """
    with torch.no_grad():
        rope = rope_for(rope_decoder, queries, positions)
        attention = layer.self_attn
        query, key, value = project(attention, layer.input_layernorm(queries))
        query, key = rotate(query, rope), rotate(key, rope)
        key, _value, key_lengths = join_prefix(source, key, value, lengths)
        groups = query.shape[1] // key.shape[1]
        scale = query.shape[-1] ** -0.5
        query_offset = key_offset = 0
        terms = []
        for query_count, key_count in zip(lengths, key_lengths):
            left = query[query_offset:query_offset + query_count].float().transpose(0, 1)
            right = key[key_offset:key_offset + key_count].float().transpose(0, 1)
            right = right.repeat_interleave(groups, 0)
            weights = torch.softmax(torch.matmul(left, right.transpose(-1, -2)) * scale, dim=-1)
            terms.append(-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1))
            query_offset += query_count
            key_offset += key_count
        return float(torch.cat([term.reshape(-1) for term in terms]).mean())


class TrainedObservationObserver(nn.Module):
    """Independent frozen readout. ``loaded`` is false until a format-4 checkpoint is copied in."""

    def __init__(self, decoder, slots: int, position_start: int, seed_token_ids=None):
        super().__init__()
        object.__setattr__(self, "_rope_owner", decoder)
        object.__setattr__(self, "embed", decoder.embed_tokens)
        self.layers = nn.ModuleList(_UndReadLayer(layer) for layer in decoder.layers)
        self.slots = slots
        self.position_start = position_start
        self.seed_token_ids = None if seed_token_ids is None else tuple(int(i) for i in seed_token_ids)
        self.loaded = False
        self.checkpoint_path = None
        self.checkpoint_sha256 = None
        self.stage = None
        self.source_mode = None
        self.trained_sources = None
        self.parent = None
        self.freeze()

    def freeze(self) -> None:
        self.eval()
        self.requires_grad_(False)

    def zeros(self, samples: int, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros((samples, self.slots, like.shape[-1]))

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "checkpoint": self.checkpoint_path,
            "sha256": self.checkpoint_sha256,
            "stage": self.stage,
            "source_mode": self.source_mode,
            "trained_sources": None if self.trained_sources is None else list(self.trained_sources),
            "parent": self.parent,
            "seed_token_ids": None if self.seed_token_ids is None else list(self.seed_token_ids),
            "source_policy": OBSERVER_SPEC["source_policy"],
            "layers": OBSERVER_SPEC["layers"],
            "query": OBSERVER_SPEC["query"],
            "prompt_kv_in_source": False,
        }

    def load_checkpoint(self, path: str | Path, *, source_mode: str = "gen_full_query") -> None:
        """Copy trained UND projections. Native generation parameters are not written.

        ``source_mode`` has to be the source this checkpoint's training config
        actually used. A C2-feedback or C3 file whose tasks saw prompt KV is
        refused here, before any tensor is copied.
        """
        path = Path(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format_version") != FORMAT_VERSION:
            raise ValueError("observation reader requires native UND checkpoint format 4")
        if saved.get("stage") not in STAGES:
            raise ValueError("observation checkpoint stage is not C2-observe, C2-feedback, or C3-loop")
        if saved.get("topology") != "C" or saved.get("slots") != self.slots or saved.get("depth") != len(self.layers):
            raise ValueError("observation checkpoint topology, slots, or depth differ")
        seed_ids = tuple(int(i) for i in saved["seed_ids"])
        if len(seed_ids) != self.slots:
            raise ValueError("observation seed length differs from slots")
        if self.seed_token_ids is not None and self.seed_token_ids != seed_ids:
            raise ValueError("configured seed tokens differ from the observation checkpoint")
        provenance = require_source_mode(saved, source_mode)
        und = saved.get("und")
        if not isinstance(und, dict):
            raise ValueError("observation checkpoint has no und state")
        expected = observation_readout_keys(self.layers)
        if set(und) != set(expected):
            extra = sorted(set(und) - set(expected))
            missing = sorted(set(expected) - set(und))
            detail = extra[0] if extra else missing[0]
            raise ValueError("observation checkpoint does not match the trained UND readout set: " + detail)
        native_ptrs = {parameter.data_ptr() for parameter in self._rope_owner.parameters()}
        targets = []
        for key in expected:
            layer_index = int(key.split(".")[1])
            local = key[len(f"layers.{layer_index}."):]
            parameter = self.layers[layer_index].get_parameter(local)
            if parameter.data_ptr() in native_ptrs:
                raise RuntimeError("refusing to write an observation tensor into the native decoder")
            if parameter.shape != und[key].shape:
                raise ValueError(f"observation tensor {key} has the wrong shape")
            targets.append((parameter, und[key]))
        with torch.no_grad():
            for parameter, tensor in targets:
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.seed_token_ids = seed_ids
        self.loaded = True
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = digest
        self.stage = provenance["stage"]
        self.source_mode = provenance["resolved_mode"]
        self.trained_sources = tuple(provenance["trained_sources"])
        self.parent = provenance["parent"]
        self.freeze()

    def read(self, captured: dict, branch: BranchContext) -> torch.Tensor:
        return observation_slots(
            self.layers, self.embed, self._rope_owner, self.seed_token_ids, captured,
            branch, self.slots, self.position_start,
        )

    def read_with_boundary(self, captured: dict, branch: BranchContext, boundary: int):
        """First read. Slots match ``read``; the prefix enters the body layer."""
        return observation_with_prefix(
            self.layers, self.embed, self._rope_owner, self.seed_token_ids, captured,
            branch, self.slots, self.position_start, boundary,
        )

    def read_from_boundary(self, captured: dict, branch: BranchContext, prefix: ReaderPrefix, pre_ptrs):
        """Later read. Pre layers are not run."""
        return observation_from_prefix(
            self.layers, self._rope_owner, prefix, captured, branch, self.slots, tuple(pre_ptrs),
        )
