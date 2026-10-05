"""Parameter-free, layer-local GEN/UND recurrence over frozen native BAGEL.

P is immutable prompt KV; M is an ephemeral UND state; G includes the native
SOI/EOI boundary tokens. All reads use the same iteration's pre-update states.
"""
from dataclasses import dataclass
from types import MethodType
import torch
from .attention import flash_attn_varlen_func

MODES = ('BASE', 'GEN_LAYERWISE', 'MEMORY_DYNAMIC', 'MEMORY_STATIC', 'MEMORY_NO_READ')


@dataclass(frozen=True)
class LoopConfig:
    mode: str = 'MEMORY_DYNAMIC'
    evaluations: int = 2
    start_layer: int = 16
    end_layer: int = 24
    memory_slots: int = 16
    progress_start: float = 0.0
    progress_end: float = 0.5

    def __post_init__(self):
        if self.mode not in MODES or self.evaluations < 1 or self.memory_slots < 0:
            raise ValueError('invalid mode, N or memory slot count')
        if not 0 <= self.start_layer < self.end_layer:
            raise ValueError('layer window must be nonempty and half-open')
        if not 0 <= self.progress_start <= self.progress_end <= 1:
            raise ValueError('sampling progress must be in [0,1]')


@dataclass
class MemorySeed:
    hidden: torch.Tensor
    rope: tuple
    lengths: list
    source_indexes: torch.Tensor


def select_content_indexes(token_ids, lengths, special_ids, maximum):
    """Distinct, deterministic uniform content positions, with no duplication."""
    indexes, counts, offset = [], [], 0
    for length in lengths:
        local = [offset+j for j in range(int(length)) if int(token_ids[offset+j]) not in special_ids]
        count = min(maximum, len(local))
        selected = [local[int(j*(len(local)-1)/max(count-1, 1))] for j in range(count)]
        indexes.extend(selected); counts.append(count); offset += int(length)
    if offset != len(token_ids):
        raise ValueError('packed token lengths do not cover input')
    return torch.tensor(indexes, device=token_ids.device, dtype=torch.long), counts


def _route(x, text, image, und, gen):
    u, g = und(x[text]), gen(x[image])
    y = u.new_empty((len(x), *u.shape[1:]))
    y[text] = u; y[image] = g
    return y


def _rotary(q, k, rope):
    cos, sin = (r.unsqueeze(1) for r in rope)
    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)
    return (q*cos+rotate(q)*sin).to(torch.bfloat16), (k*cos+rotate(k)*sin).to(torch.bfloat16)


def _project(layer, hidden, rope, text=None, image=None, query=True):
    a = layer.self_attn
    if text is None:
        h = layer.input_layernorm(hidden)
        q = a.q_norm(a.q_proj(h).view(-1, a.num_heads, a.head_dim)) if query else None
        k = a.k_norm(a.k_proj(h).view(-1, a.num_key_value_heads, a.head_dim))
        v = a.v_proj(h).view(-1, a.num_key_value_heads, a.head_dim)
    else:
        h = _route(hidden, text, image, layer.input_layernorm, layer.input_layernorm_moe_gen).to(torch.bfloat16)
        q = _route(h, text, image, a.q_proj, a.q_proj_moe_gen).view(-1, a.num_heads, a.head_dim).float()
        k = _route(h, text, image, a.k_proj, a.k_proj_moe_gen).view(-1, a.num_key_value_heads, a.head_dim).float()
        v = _route(h, text, image, a.v_proj, a.v_proj_moe_gen).view(-1, a.num_key_value_heads, a.head_dim)
        q = _route(q, text, image, a.q_norm, a.q_norm_moe_gen)
        k = _route(k, text, image, a.k_norm, a.k_norm_moe_gen)
    # No final unused writer Q projection. K rotation uses the same native RoPE.
    if q is None:
        _, k = _rotary(torch.zeros_like(k), k, rope)
    else:
        q, k = _rotary(q, k, rope)
    return q, k, v.to(torch.bfloat16)


def _finish(layer, hidden, attention, text=None, image=None):
    a = layer.self_attn
    attention = attention.reshape(-1, a.hidden_size)
    if text is None:
        out = hidden + a.o_proj(attention)
        return out + layer.mlp(layer.post_attention_layernorm(out))
    out = hidden + _route(attention, text, image, a.o_proj, a.o_proj_moe_gen)
    normed = _route(out, text, image, layer.post_attention_layernorm, layer.post_attention_layernorm_moe_gen).to(torch.bfloat16)
    return out + _route(normed, text, image, layer.mlp, layer.mlp_moe_gen)


def _overlay(prompt, memory, gen, plen, mlen, glen, read_memory):
    parts, lengths = [], []
    pi = mi = gi = 0
    for p, m, g in zip(plen, mlen, glen):
        pieces = [prompt[pi:pi+p]]
        if read_memory: pieces.append(memory[mi:mi+m])
        pieces.append(gen[gi:gi+g])
        parts.append(torch.cat(pieces)); lengths.append(p+g+(m if read_memory else 0))
        pi += p; mi += m; gi += g
    if pi != len(prompt) or mi != len(memory) or gi != len(gen):
        raise ValueError('packed KV layout mismatch')
    return torch.cat(parts), lengths


def _attend(q, k, v, qlens, klens):
    def cumulative(lens):
        return torch.tensor([0]+list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=q.device)
    return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cumulative(qlens),
        cu_seqlens_k=cumulative(klens), max_seqlen_q=max(qlens), max_seqlen_k=max(klens), causal=False)


def recurrent_layer(layer, kwargs, seed, config, diagnostics=None):
    """Two attention calls share one projection of each pre-update G/M state."""
    hidden = kwargs['packed_query_sequence']
    glen = kwargs['query_lens'].tolist()
    cache = kwargs['past_key_values']; idx = layer.self_attn.layer_idx
    pk, pv = cache.key_cache[idx], cache.value_cache[idx]
    if pk is None:
        a = layer.self_attn
        pk = hidden.new_empty((0, a.num_key_value_heads, a.head_dim))
        pv = pk
    plen = kwargs['key_values_lens'].tolist()
    text, image = kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes']
    rope = kwargs['packed_query_position_embeddings']
    memory = seed.hidden.clone() if seed is not None else hidden[:0]
    mlen = seed.lengths if seed is not None else [0]*len(glen)
    has_memory = len(memory) > 0
    step = 1/config.evaluations
    for r in range(config.evaluations):
        gq, gk, gv = _project(layer, hidden, rope, text, image)
        write = has_memory and r < config.evaluations-1
        if has_memory:
            mq, mk, mv = _project(layer, memory, seed.rope, query=write)
        else:
            mq = None; mk, mv = gk[:0], gv[:0]
        read = has_memory and r > 0 and config.mode != 'MEMORY_NO_READ'
        rk, rlen = _overlay(pk, mk, gk, plen, mlen, glen, read)
        rv, _ = _overlay(pv, mv, gv, plen, mlen, glen, read)
        if diagnostics is not None:
            from .memory_stats import memory_slot_stats, sampled_read_mass
            mass = sampled_read_mass(gq, rk, glen, rlen, plen, mlen, read)
            states = list(memory.split(mlen))
            records = [{'iteration':r, 'sample':i, 'read_mass':mass[i],
                        'update_ratio':0., **memory_slot_stats(state)} for i,state in enumerate(states)]
        gfull = _finish(layer, hidden, _attend(gq, rk, rv, glen, rlen), text, image)
        if write:
            wk, wlen = _overlay(pk, mk, gk, plen, mlen, glen, True)
            wv, _ = _overlay(pv, mv, gv, plen, mlen, glen, True)
            mfull = _finish(layer, memory, _attend(mq, wk, wv, mlen, wlen))
            candidate = memory + step*(mfull-memory)
            if diagnostics is not None:
                for record, before, after in zip(records, memory.split(mlen), candidate.split(mlen)):
                    record['candidate_update_ratio'] = float((after-before).float().norm()/before.float().norm().clamp_min(1e-12))
                    record['update_ratio'] = record['candidate_update_ratio'] if config.mode != 'MEMORY_STATIC' else 0.
            if config.mode != 'MEMORY_STATIC':
                memory = candidate
        hidden = hidden + step*(gfull-hidden)
        if diagnostics is not None: diagnostics.extend(records)
    return hidden, cache


class InternalLoopRuntime:
    """Install instance methods only; model state_dict and native weights stay intact."""
    def __init__(self, model, config=LoopConfig(), diagnostics=False):
        self.model, self.config = model, config
        self.progress = 0.0
        self.capture = None
        self.diagnostics_enabled = diagnostics
        self.diagnostics = []
        self.banks = {}
        self.originals = []
        layers = model.language_model.model.layers
        if config.end_layer > len(layers):
            raise ValueError('loop window exceeds native decoder depth')
        for index, layer in enumerate(layers):
            original = layer.forward_inference
            self.originals.append((layer, original))
            def wrapped(this, _index=index, _original=original, **kwargs):
                return self._layer(this, _index, _original, kwargs)
            layer.forward_inference = MethodType(wrapped, layer)

    def begin_prefill(self, cache, token_ids, lengths, special_ids):
        indexes, counts = select_content_indexes(token_ids, lengths, special_ids, self.config.memory_slots)
        self.capture = (cache, indexes, counts)
        self.banks[id(cache)] = (cache, {})

    def end_prefill(self):
        self.capture = None

    def _layer(self, layer, index, original, kwargs):
        cfg = self.config
        mode = kwargs.get('mode', 'und')
        selected = cfg.start_layer <= index < cfg.end_layer
        if mode == 'und' and self.capture is not None and selected and cfg.mode.startswith('MEMORY') and cfg.evaluations > 1:
            cache, ids, counts = self.capture
            if kwargs['past_key_values'] is not cache:
                raise RuntimeError('prefill branch/cache mismatch')
            hidden = kwargs['packed_query_sequence'][ids].detach().clone()
            rope = tuple(x[ids].detach().clone() for x in kwargs['packed_query_position_embeddings'])
            self.banks[id(cache)][1][index] = MemorySeed(hidden, rope, counts, ids.clone())
        active = selected and cfg.mode != 'BASE' and cfg.evaluations > 1 and cfg.progress_start <= self.progress <= cfg.progress_end
        if mode != 'gen' or not active:
            return original(**kwargs)
        if kwargs.get('update_past_key_values', True) or kwargs.get('is_causal', True):
            raise ValueError('GEN loop requires immutable prompt KV and noncausal attention')
        if cfg.mode == 'GEN_LAYERWISE':
            hidden = kwargs['packed_query_sequence']
            for _ in range(cfg.evaluations):
                full, cache = original(**{**kwargs, 'packed_query_sequence':hidden})
                hidden = hidden + (full-hidden)/cfg.evaluations
            return hidden, cache
        cache = kwargs['past_key_values']
        bank = self.banks.get(id(cache))
        if bank is None or index not in bank[1]:
            if int(kwargs['key_values_lens'].sum()) != 0:
                raise RuntimeError('Memory arm has no prompt seed for this CFG branch')
            seed = None  # Native text-removed CFG has no prompt or Memory.
        else:
            seed = bank[1][index]
        details = [] if self.diagnostics_enabled else None
        result = recurrent_layer(layer, kwargs, seed, cfg, details)
        if details is not None:
            self.diagnostics.extend({'layer':index, 'progress':self.progress,
                'branch':'conditional' if int(kwargs['key_values_lens'].sum()) else 'text_removed', **d} for d in details)
        return result

    def close(self):
        for layer, original in self.originals: layer.forward_inference = original
        self.originals.clear(); self.banks.clear(); self.capture = None
