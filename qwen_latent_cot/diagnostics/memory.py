"""Independent frozen-UND Memory. Keys in MemoryBank are ALWAYS unrotated."""
from dataclasses import dataclass, replace
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class KV:
    layer: int
    keys: torch.Tensor
    values: torch.Tensor
    lengths: tuple[int, ...]
    rotated: bool = True
    source_kind: str = 'text'
    t: float | None = None
    round: int = 0
    token_roles: tuple[str, ...] = ()
    positions: tuple[int, ...] = ()
    expert: str = 'und'

    def __post_init__(self):
        if self.keys.shape != self.values.shape or self.keys.ndim != 3:
            raise ValueError('KV must have matching [tokens, heads, head_dim] shapes')
        if any(n < 0 for n in self.lengths) or sum(self.lengths) != len(self.keys):
            raise ValueError('KV lengths do not cover packed samples')
        if self.token_roles and len(self.token_roles) != len(self.keys):
            raise ValueError('token roles do not cover source')
        if self.positions and len(self.positions) != len(self.keys):
            raise ValueError('positions do not cover source')

    def detached(self):
        return replace(self, keys=self.keys.detach(), values=self.values.detach())


@dataclass(frozen=True)
class MemoryBank:
    layers: tuple[KV, ...]
    topology: str
    slots: int
    readout: str = 'post_block'

    def __post_init__(self):
        if not self.layers or self.topology not in ('L', 'C', 'fixed'):
            raise ValueError('invalid full-depth Memory bank')
        lengths = self.layers[0].lengths
        if any(n != self.slots for n in lengths):
            raise ValueError('Memory has incorrect slot count')
        for index, kv in enumerate(self.layers):
            if kv.layer != index or kv.rotated or kv.lengths != lengths:
                raise ValueError('bank must contain every same-layer UNROTATED KV')

    def swap(self, permutation):
        """Exchange whole samples at EVERY layer, preserving layout."""
        count = len(self.layers[0].lengths)
        if sorted(permutation) != list(range(count)):
            raise ValueError('swap must be a permutation of packed samples')
        out = []
        for kv in self.layers:
            keys, values = kv.keys.split(kv.lengths), kv.values.split(kv.lengths)
            out.append(KV(kv.layer, torch.cat([keys[i] for i in permutation]),
                          torch.cat([values[i] for i in permutation]), kv.lengths, False))
        return MemoryBank(tuple(out), self.topology, self.slots)


def rope_for(decoder, hidden, positions):
    cos, sin = decoder.rotary_emb(hidden, positions.unsqueeze(0))
    return cos.squeeze(0), sin.squeeze(0)


def rotate(states, rope):
    cos, sin = rope
    half = states.shape[-1] // 2
    rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return (states * cos[:, None] + rotated * sin[:, None]).to(states.dtype)


def project(attention, normalized, mode='und', text_indexes=None, image_indexes=None,
            query_adapter=None):
    """Native expert-specific Q/K/V and Q/K norms, before RoPE."""
    if mode == 'und':
        raw_q = attention.q_proj(normalized)
        if query_adapter is not None:
            raw_q = raw_q + query_adapter(normalized).to(raw_q.dtype)
        q = attention.q_norm(raw_q.view(-1, attention.num_heads, attention.head_dim))
        k = attention.k_norm(attention.k_proj(normalized).view(-1, attention.num_key_value_heads, attention.head_dim))
        v = attention.v_proj(normalized).view_as(k)
    elif mode == 'gen':
        q = normalized.new_zeros(len(normalized), attention.num_heads, attention.head_dim)
        k = normalized.new_zeros(len(normalized), attention.num_key_value_heads, attention.head_dim)
        v = torch.zeros_like(k)
        for indexes, suffix in ((text_indexes, ''), (image_indexes, '_moe_gen')):
            if indexes is None:
                raise ValueError('GEN projection requires text and image indexes')
            h = normalized[indexes]
            qi = getattr(attention, 'q_proj' + suffix)(h).view(-1, attention.num_heads, attention.head_dim)
            ki = getattr(attention, 'k_proj' + suffix)(h).view(-1, attention.num_key_value_heads, attention.head_dim)
            # Native GEN performs norms in float32 and casts AFTER RoPE.
            q = q.float(); k = k.float()
            q[indexes] = getattr(attention, 'q_norm' + suffix)(qi.float())
            k[indexes] = getattr(attention, 'k_norm' + suffix)(ki.float())
            v[indexes] = getattr(attention, 'v_proj' + suffix)(h).view_as(ki)
    else:
        raise ValueError('unknown expert')
    return q, k, v


def packed_attention(q, k, v, query_lengths, key_lengths, *, causal=False, visible=None,
                     native_unmasked=False):
    """Differentiable SDPA with bottom-right causal masks and strict isolation.

    visible is a packed key mask. False keys never enter the softmax.
    CPU accumulates in float32, matching the native CPU diagnostic backend.
    """
    if len(query_lengths) != len(key_lengths) or sum(query_lengths) != len(q) or sum(key_lengths) != len(k):
        raise ValueError('attention lengths do not cover samples')
    if k.shape != v.shape or q.shape[1] % k.shape[1]:
        raise ValueError('incompatible grouped query heads')
    if visible is not None and (visible.dtype != torch.bool or len(visible) != len(k)):
        raise ValueError('visible must be a bool mask over packed keys')
    outputs = []; qo = ko = 0
    for nq, nk in zip(query_lengths, key_lengths):
        qi = q[qo:qo+nq].transpose(0, 1)[None]
        ki = k[ko:ko+nk].transpose(0, 1)[None]
        vi = v[ko:ko+nk].transpose(0, 1)[None]
        groups = qi.shape[1] // ki.shape[1]
        ki = ki.repeat_interleave(groups, 1); vi = vi.repeat_interleave(groups, 1)
        if native_unmasked and (causal or visible is not None):
            raise ValueError('native unmasked path cannot receive an explicit mask')
        mask = None if native_unmasked else torch.ones(nq, nk, dtype=torch.bool, device=q.device)
        if causal:
            mask &= torch.arange(nk, device=q.device)[None] <= torch.arange(nq, device=q.device)[:, None] + nk-nq
        if visible is not None:
            mask &= visible[ko:ko+nk][None]
        if q.device.type == 'cpu':
            qi, ki, vi = qi.float(), ki.float(), vi.float()
        with torch.autocast('cpu', enabled=False) if q.device.type == 'cpu' else nullcontext():
            output = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, dropout_p=0.)
        outputs.append(output[0].transpose(0, 1).to(q.dtype))
        qo += nq; ko += nk
    return torch.cat(outputs)


def join_prefix(source, keys, values, lengths):
    if len(lengths) != len(source.lengths):
        raise ValueError('source/query sample counts differ')
    merged_k = []; merged_v = []
    for sk, sv, k, v in zip(source.keys.split(source.lengths), source.values.split(source.lengths),
                            keys.split(lengths), values.split(lengths)):
        merged_k.append(torch.cat((sk, k))); merged_v.append(torch.cat((sv, v)))
    return torch.cat(merged_k), torch.cat(merged_v), tuple(a+b for a, b in zip(source.lengths, lengths))


def und_block(layer, hidden, rope, lengths, source=None, *, causal=False, query_adapter=None):
    counter = getattr(layer, '_diagnostic_cost_counter', None)
    if counter is not None: counter.counts[2] += 1
    a = layer.self_attn
    q, k, v = project(a, layer.input_layernorm(hidden), query_adapter=query_adapter)
    q, k = rotate(q, rope), rotate(k, rope)
    klens = lengths
    if source is not None:
        if source.layer != a.layer_idx or not source.rotated:
            raise ValueError('writer requires same-layer native rotated source KV')
        k, v, klens = join_prefix(source, k, v, lengths)
    out = packed_attention(q, k, v, lengths, klens, causal=causal).reshape(len(hidden), -1)
    hidden = hidden + a.o_proj(out)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class LowRankResidual(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(self.down(hidden.to(self.down.weight.dtype))).to(hidden.dtype)


class MemoryWriter(nn.Module):
    """Own ONLY adapters; do not register the frozen decoder as trainable state."""
    def __init__(self, decoder, topology='L', slots=16, rank=32):
        super().__init__()
        if topology not in ('L', 'C') or slots <= 0 or rank <= 0:
            raise ValueError('invalid Memory configuration')
        self.topology, self.slots, self.rank = topology, slots, rank
        object.__setattr__(self, 'decoder', decoder)
        decoder.eval().requires_grad_(False)
        width = decoder.embed_tokens.weight.shape[1]
        self.adapters = nn.ModuleList([LowRankResidual(width, rank) for _ in decoder.layers])
        self.to(device=decoder.embed_tokens.weight.device)

    def forward(self, seeds, source, positions, source_off=False, *, source_kind='text',
                t=None, visual_adapter=None, detach_source=True, source_layers=None):
        count = len(self.decoder.layers)
        if len(seeds) != count or len(source) != count:
            raise ValueError('seeds and source must cover ALL decoder layers')
        lengths = source[0].lengths
        samples = len(lengths); slot_lengths = (self.slots,) * samples
        if len(positions) != samples * self.slots:
            raise ValueError('writer positions must cover fixed slots')
        hidden = seeds[0].detach()
        bank = []
        for index, (layer, adapter) in enumerate(zip(self.decoder.layers, self.adapters)):
            # Layer windows change token counts, never the packed sample count.
            same_layout = len(source[index].lengths) == samples
            if source[index].layer != index or not same_layout:
                raise ValueError('source must preserve layer and sample layout')
            h = seeds[index].detach() if self.topology == 'L' else hidden
            if h.shape != seeds[0].shape or len(h) != sum(slot_lengths):
                raise ValueError('invalid neutral seed shape')
            rope = rope_for(self.decoder, h, positions)
            origin = source[index].detached() if detach_source else source[index]
            if origin.positions and set(origin.positions).intersection(positions.tolist()):
                raise ValueError('GEN source positions overlap writer slots')
            query_adapter = None
            if visual_adapter is not None and source_kind == 'visual':
                if t is None or not 0 <= float(t) <= 1:
                    raise ValueError('visual writing requires actual t in [0, 1]')
                query_adapter = lambda normalized, i=index: visual_adapter(i, normalized, t)
            u = und_block(layer, h, rope, slot_lengths, None if source_off or (source_layers is not None and index not in source_layers) else origin,
                          query_adapter=query_adapter)
            hidden = u + adapter(u)
            _, k, v = project(layer.self_attn, layer.input_layernorm(hidden))
            bank.append(KV(index, k, v, slot_lengths, False))
        return MemoryBank(tuple(bank), self.topology, self.slots)

    @torch.no_grad()
    def fixed_bank(self, seeds):
        layers = []
        for i, (layer, h) in enumerate(zip(self.decoder.layers, seeds)):
            _, k, v = project(layer.self_attn, layer.input_layernorm(h))
            layers.append(KV(i, k, v, (self.slots,) * (len(h) // self.slots), False))
        return MemoryBank(tuple(layers), 'fixed', self.slots)


@torch.no_grad()
def neutral_seeds(decoder, token_ids, samples=1):
    """Causal native UND prefill of fixed content tokens; never target tokens."""
    if token_ids.ndim != 1 or not len(token_ids):
        raise ValueError('neutral initialization needs a nonempty token sequence')
    hidden = decoder.embed_tokens(token_ids.repeat(samples))
    positions = torch.arange(len(token_ids), device=hidden.device).repeat(samples)
    lengths = (len(token_ids),) * samples
    rope = rope_for(decoder, hidden, positions)
    seeds = []
    for layer in decoder.layers:
        seeds.append(hidden.detach().clone())
        hidden = und_block(layer, hidden, rope, lengths, causal=True)
    return tuple(seeds)


def consume_bank(decoder, bank, starts):
    """Apply consumer RoPE ONCE; preserve K/V gradients through the frozen decoder."""
    if len(starts) != len(bank.layers[0].lengths):
        raise ValueError('consumer positions must match samples')
    positions = torch.cat([torch.arange(s, s+bank.slots, device=bank.layers[0].keys.device) for s in starts])
    out = []
    for kv in bank.layers:
        rope = rope_for(decoder, kv.keys, positions)
        out.append(KV(kv.layer, rotate(kv.keys, rope), kv.values, kv.lengths, True))
    return tuple(out)
