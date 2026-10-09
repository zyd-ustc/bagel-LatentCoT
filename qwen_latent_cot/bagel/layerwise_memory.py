"""Capture every native prompt slot and construct immutable packed KV contexts."""
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary
import torch
from .modeling.bagel.qwen2_navit import NaiveCache


@dataclass
class PromptMemorySeed:
    hidden: torch.Tensor
    positions: torch.Tensor
    lengths: tuple
    source_indexes: torch.Tensor
    start_layer: int
    full_prompt: bool = False
    special_mask: torch.Tensor = None
    layer_hidden: dict = field(default_factory=dict)

@dataclass
class LayerKV:
    layer: int
    keys: torch.Tensor
    values: torch.Tensor
    lengths: tuple

    def __post_init__(self):
        if self.keys.shape != self.values.shape or sum(self.lengths) != len(self.keys):
            raise ValueError('layer KV does not match its packed sample lengths')

def _packed_indexes(query_lengths, context_lengths, device):
    query, past, offset = [], [], 0
    for q, p in zip(query_lengths, context_lengths):
        past.extend(range(offset, offset+p))
        query.extend(range(offset+p, offset+p+q))
        offset += p+q
    return (torch.tensor(query, device=device, dtype=torch.long),
            torch.tensor(past, device=device, dtype=torch.long))

def native_context(cache, layer, prompt_lengths, additions, query_lengths, device,
                   include_prompt=True):
    """New temporary cache; original P and every supplied KV remain read-only."""
    if len(prompt_lengths) != len(query_lengths):
        raise ValueError('packed prompt/query sample counts differ')
    for kv in additions:
        if kv.layer != layer:
            raise ValueError('a layer may only read KV from the same native layer')
        if len(kv.lengths) != len(prompt_lengths):
            raise ValueError('KV sample count differs from query sample count')
    # Replacing P is a read policy, never a mutation of the native prompt cache.
    retained = ([bool(include_prompt)]*len(prompt_lengths) if isinstance(include_prompt, bool)
                else list(include_prompt))
    if len(retained) != len(prompt_lengths):
        raise ValueError('prompt read policy sample counts differ')
    pk, pv = cache.key_cache[layer], cache.value_cache[layer]
    if sum(prompt_lengths) and (pk is None or len(pk) != sum(prompt_lengths)):
        raise ValueError('prompt KV lengths do not cover this layer')
    key_parts, value_parts, lengths = [], [], []
    offsets = [0]*(1+len(additions))
    for sample, plen in enumerate(prompt_lengths):
        pieces_k, pieces_v = [], []
        if plen and retained[sample]:
            pieces_k.append(pk[offsets[0]:offsets[0]+plen])
            pieces_v.append(pv[offsets[0]:offsets[0]+plen])
        offsets[0] += plen
        length = plen if retained[sample] else 0
        for j, kv in enumerate(additions, 1):
            n = kv.lengths[sample]
            pieces_k.append(kv.keys[offsets[j]:offsets[j]+n])
            pieces_v.append(kv.values[offsets[j]:offsets[j]+n])
            offsets[j] += n; length += n
        if pieces_k:
            key_parts.append(torch.cat(pieces_k)); value_parts.append(torch.cat(pieces_v))
        lengths.append(length)
    temporary = NaiveCache(cache.num_layers)
    if sum(lengths):
        temporary.key_cache[layer] = torch.cat(key_parts)
        temporary.value_cache[layer] = torch.cat(value_parts)
    query, past = _packed_indexes(query_lengths, lengths, device)
    return temporary, torch.tensor(lengths, device=device, dtype=torch.int32), query, past


class LayerwiseMemoryLoop:
    def __init__(self, decoder):
        self.decoder = decoder
        self.seeds = WeakKeyDictionary()
        self.prefill = None

    def begin_prefill(self, cache, token_ids, lengths, special_ids, config):
        counts = tuple(int(n) for n in lengths)
        if any(n < 0 for n in counts) or sum(counts) != len(token_ids):
            raise ValueError('packed prompt lengths do not cover token IDs')
        indexes = torch.arange(len(token_ids), device=token_ids.device)
        special_mask = torch.isin(token_ids, token_ids.new_tensor(sorted(special_ids)))
        self.prefill = cache, indexes, counts, special_mask

    def end_prefill(self):
        self.prefill = None

    def capture_prefill(self, original, kwargs, config):
        cache, indexes, counts, special_mask = self.prefill
        if kwargs['past_key_values'] is not cache:
            raise RuntimeError('prompt seed capture must use its own CFG branch cache')
        positions = kwargs['packed_query_position_ids'][indexes].detach().clone()
        seed = PromptMemorySeed(None, positions, counts, indexes.clone(),
                                config.start_layer, True, special_mask)
        self.seeds[cache] = seed
        handles = []
        def capture(index):
            def observed(layer, args, layer_kwargs):
                hidden = layer_kwargs['packed_query_sequence'][indexes].detach().clone()
                seed.layer_hidden[index] = hidden
                if index == config.start_layer:
                    seed.hidden = hidden
            return observed
        try:
            for index in range(config.start_layer, len(self.decoder.layers)):
                handles.append(self.decoder.layers[index].register_forward_pre_hook(capture(index), with_kwargs=True))
            return original(**kwargs)
        finally:
            for handle in handles:
                handle.remove()

    def clear(self):
        self.prefill = None
        self.seeds.clear()

    def run(self, kwargs, runtime):
        if runtime.config.memory_update in ('full_depth','full_depth_restart'):
            from .full_depth_memory import run_und_state
        else:
            from .und_state_loop import run_und_state
        return run_und_state(self, kwargs, runtime)
