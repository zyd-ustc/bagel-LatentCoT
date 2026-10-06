"""Layer-aligned UND KV feedback inside one frozen BAGEL denoiser call.

Every writer starts from the same native prompt state at the body entrance.
Writer hidden traverses the body once in native depth order; it never returns
from the body end to the entrance. Only layer-input KV carries between rounds.
GEN always restarts at its fixed body entrance. Native layers perform every
projection, attention, norm, MLP, and full residual update.
"""
from dataclasses import dataclass
from weakref import WeakKeyDictionary
import torch
from .modeling.bagel.qwen2_navit import NaiveCache, BaseNavitOutputWithPast

LAYERWISE_MODES = ('LAYERWISE_MEMORY_KV', 'LAYERWISE_KV_NO_READ',
                   'LAYERWISE_MEMORY_REPLACE', 'LAYERWISE_SEED_REPLACE')
REPLACE_MODES = ('LAYERWISE_MEMORY_REPLACE', 'LAYERWISE_SEED_REPLACE')


@dataclass
class PromptMemorySeed:
    hidden: torch.Tensor
    positions: torch.Tensor
    lengths: tuple
    source_indexes: torch.Tensor
    start_layer: int
    maximum_slots: int


@dataclass
class LayerKV:
    layer: int
    keys: torch.Tensor
    values: torch.Tensor
    lengths: tuple

    def __post_init__(self):
        if self.keys.shape != self.values.shape or sum(self.lengths) != len(self.keys):
            raise ValueError('layer KV does not match its packed sample lengths')


def select_content_indexes(token_ids, lengths, special_ids, maximum):
    """Select distinct native content positions; no random or learned slots."""
    indexes, counts, offset = [], [], 0
    for length in lengths:
        local = [offset+j for j in range(int(length)) if int(token_ids[offset+j]) not in special_ids]
        count = min(maximum, len(local))
        chosen = [local[int(j*(len(local)-1)/max(count-1, 1))] for j in range(count)]
        indexes.extend(chosen); counts.append(count); offset += int(length)
    if offset != len(token_ids):
        raise ValueError('packed prompt lengths do not cover token IDs')
    return token_ids.new_tensor(indexes, dtype=torch.long), tuple(counts)


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
    # A sample with no content seed keeps its native P, even in a mixed batch.
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
        indexes, counts = select_content_indexes(token_ids, lengths, special_ids, config.memory_slots)
        self.prefill = (cache, indexes, counts)

    def end_prefill(self):
        self.prefill = None

    def capture_prefill(self, original, kwargs, config):
        cache, indexes, counts = self.prefill
        if kwargs['past_key_values'] is not cache:
            raise RuntimeError('prompt seed capture must use its own CFG branch cache')
        positions = kwargs['packed_query_position_ids'][indexes].detach().clone()
        def capture(layer, args, layer_kwargs):
            hidden = layer_kwargs['packed_query_sequence'][indexes].detach().clone()
            self.seeds[cache] = PromptMemorySeed(hidden, positions, counts, indexes.clone(),
                                                config.start_layer, config.memory_slots)
        handle = self.decoder.layers[config.start_layer].register_forward_pre_hook(capture, with_kwargs=True)
        try:
            return original(**kwargs)
        finally:
            handle.remove()

    def clear(self):
        self.prefill = None
        self.seeds.clear()

    def run(self, kwargs, runtime):
        cfg, decoder = runtime.config, self.decoder
        cache = kwargs['past_key_values']
        seed = self.seeds.get(cache)
        if seed is None:
            if int(kwargs['key_values_lens'].sum()):
                raise RuntimeError('layerwise Memory requires native prompt prefill through this runtime')
            # Text-removed CFG has no prompt-derived Memory. Keep its native prior.
            return runtime.original(**kwargs)
        if not len(seed.hidden):
            return runtime.original(**kwargs)
        if cfg.start_layer!=seed.start_layer or cfg.memory_slots!=seed.maximum_slots:
            raise ValueError('changing Memory entrance or slot count requires a fresh prompt prefill')
        if len(seed.lengths) != len(kwargs['query_lens']):
            raise ValueError('Memory seed and GEN packed sample counts differ')
        prompt_lengths = tuple(kwargs['key_values_lens'].tolist())
        gen_lengths = tuple(kwargs['query_lens'].tolist())
        hidden = kwargs['packed_query_sequence']
        cos, sin = decoder.rotary_emb(hidden, kwargs['packed_query_position_ids'].unsqueeze(0))
        gen_rope = cos.squeeze(0), sin.squeeze(0)
        cos, sin = decoder.rotary_emb(seed.hidden, seed.positions.unsqueeze(0))
        memory_rope = cos.squeeze(0), sin.squeeze(0)
        layer_kwargs = {k:v for k,v in kwargs.items() if k not in ('packed_query_sequence','packed_query_position_ids')}
        layer_kwargs['packed_query_position_embeddings'] = gen_rope
        # Whole prefix at native depth, once; no Memory overlay here.
        for index in range(cfg.start_layer):
            hidden, _ = decoder.layers[index].forward_inference(packed_query_sequence=hidden, **layer_kwargs)
        entrance = hidden.clone()
        bank = {}
        replacement = cfg.mode in REPLACE_MODES
        dynamic = cfg.mode != 'LAYERWISE_SEED_REPLACE'
        reads_enabled = cfg.mode != 'LAYERWISE_KV_NO_READ'
        seed_bank = {i:LayerKV(i, cache.key_cache[i][seed.source_indexes],
                              cache.value_cache[i][seed.source_indexes], seed.lengths)
                     for i in range(cfg.start_layer, len(decoder.layers))} if replacement else {}
        capturing = runtime.probe_capture is not None and runtime.step_index in runtime.probe_capture.steps
        if capturing and replacement:
            raise ValueError('replacement probe export is not implemented; use a separate append probe run')
        if capturing and len(seed.lengths) != 1:
            raise ValueError('probe export is batch=1; ordinary generation supports packed batches')

        def run_native(layer_index, state, lengths, rope, additions, mode, store_input=True,
                       include_prompt=True):
            temporary, klens, query, past = native_context(cache, layer_index, prompt_lengths,
                                                           additions, lengths, hidden.device, include_prompt)
            call = dict(packed_query_sequence=state,
                        query_lens=torch.tensor(lengths, device=hidden.device, dtype=torch.int32),
                        packed_query_position_embeddings=rope, packed_query_indexes=query,
                        past_key_values=temporary, key_values_lens=klens,
                        packed_key_value_indexes=past, update_past_key_values=store_input,
                        is_causal=False, mode=mode)
            if mode == 'gen':
                call.update(packed_text_indexes=kwargs['packed_text_indexes'],
                            packed_vae_token_indexes=kwargs['packed_vae_token_indexes'])
            output, _ = decoder.layers[layer_index].forward_inference(**call)
            current = (LayerKV(layer_index, temporary.key_cache[layer_index][query],
                               temporary.value_cache[layer_index][query], tuple(lengths))
                       if store_input else None)
            return output, current

        from .memory_stats import memory_slot_stats
        for round_index in range(cfg.extra_rounds + 1):
            hidden = entrance.clone()
            gen_kv = {}
            for index in range(cfg.start_layer, cfg.end_layer):
                replacing = replacement and round_index > 0
                memory = ((seed_bank[index] if index==cfg.start_layer or not dynamic else bank[index])
                          if replacing else bank.get(index) if reads_enabled else None)
                if capturing and reads_enabled and round_index == cfg.extra_rounds and index > cfg.start_layer:
                    used = bank[index]
                    runtime.probe_capture.record(runtime.step_index, index, {
                        'dynamic_k':used.keys, 'dynamic_v':used.values,
                        'seed_k':cache.key_cache[index][seed.source_indexes],
                        'seed_v':cache.value_cache[index][seed.source_indexes],
                        'lengths':list(seed.lengths), 'source_indexes':seed.source_indexes,
                        'question_position_start':int(kwargs['packed_query_position_ids'].max())+1,
                        'read_round':round_index})
                hidden, current = run_native(index, hidden, gen_lengths, gen_rope,
                                             [memory] if memory is not None else [], 'gen',
                                             store_input=dynamic and round_index<cfg.extra_rounds,
                                             include_prompt=[n==0 for n in seed.lengths] if replacing else True)
                if round_index < cfg.extra_rounds:
                    gen_kv[index] = current
                if runtime.diagnostics_enabled:
                    runtime.diagnostics.append({'phase':'gen', 'layer':index, 'round':round_index,
                        'gen_reads_memory':memory is not None, 'memory_slots_per_sample':list(seed.lengths),
                        'prompt_read_per_sample':[n==0 for n in seed.lengths] if replacing else [True]*len(seed.lengths),
                        'memory_read_kind':('seed' if index==cfg.start_layer or not dynamic else 'dynamic') if replacing else 'append',
                        'progress':runtime.progress, 'branch':'conditional'})
            if round_index == cfg.extra_rounds:
                break  # No unused final writer or intermediate suffix/readout.
            if not dynamic:
                continue  # Static compression control: no GEN-dependent writer.
            # Fresh native-depth writer entrance every round. The old body-end
            # hidden is never recycled. Historical information enters as KV only.
            writer = seed.hidden.clone()
            next_bank = {}
            writer_end = len(decoder.layers) if replacement and round_index==cfg.extra_rounds-1 else cfg.end_layer
            for index in range(cfg.start_layer, writer_end):
                previous = bank.get(index)
                additions = ([previous] if previous is not None else [])
                if index in gen_kv:
                    additions += [gen_kv[index]]
                writer_input = writer
                writer, current = run_native(index, writer, seed.lengths, memory_rope, additions, 'und')
                if index > cfg.start_layer:
                    # At the first body layer, input KV is only the fixed seed;
                    # it has not observed GEN. Do not expose that duplicate as feedback.
                    next_bank[index] = current
                if runtime.diagnostics_enabled:
                    old = cache.value_cache[index][seed.source_indexes] if previous is None else previous.values
                    before = old.split(seed.lengths)
                    after = current.values.split(seed.lengths)
                    for sample,(a,b,h) in enumerate(zip(before,after,writer_input.split(seed.lengths))):
                        runtime.diagnostics.append({'phase':'writer','layer':index,'round':round_index,
                            'sample':sample,'progress':runtime.progress,'branch':'conditional',
                            'kv_update_ratio':float((b-a).float().norm()/a.float().norm().clamp_min(1e-12)),
                            'feedback_stored':index>cfg.start_layer, **memory_slot_stats(h)})
            bank = next_bank
        # GEN suffix and final routing run once. Append mode uses native P;
        # replacement modes use same-layer M without P (except empty seeds).
        for index in range(cfg.end_layer, len(decoder.layers)):
            if replacement:
                memory = bank[index] if dynamic else seed_bank[index]
                hidden, _ = run_native(index, hidden, gen_lengths, gen_rope, [memory], 'gen',
                                       store_input=False, include_prompt=[n==0 for n in seed.lengths])
                if runtime.diagnostics_enabled:
                    runtime.diagnostics.append({'phase':'suffix','layer':index,'round':cfg.extra_rounds,
                        'gen_reads_memory':True,'prompt_read_per_sample':[n==0 for n in seed.lengths],
                        'progress':runtime.progress,'branch':'conditional'})
            else:
                hidden, _ = decoder.layers[index].forward_inference(packed_query_sequence=hidden, **layer_kwargs)
        normalized = torch.zeros_like(hidden)
        text, image = kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes']
        normalized[text] = decoder.norm(hidden[text])
        normalized[image] = decoder.norm_moe_gen(hidden[image])
        return BaseNavitOutputWithPast(packed_query_sequence=normalized, past_key_values=cache)
