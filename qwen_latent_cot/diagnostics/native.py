"""Native text/source capture and temporary, single-call GEN consumption."""
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
import torch
from .memory import KV, consume_bank, project, rotate, rope_for, packed_attention, und_block


def device_inputs(inputs, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def empty_cache(decoder):
    from ..bagel.modeling.bagel.qwen2_navit import NaiveCache
    return NaiveCache(len(decoder.layers))


def cache_kv(cache, lengths):
    return tuple(KV(i, cache.key_cache[i], cache.value_cache[i], tuple(lengths)) for i in range(cache.num_layers))


@torch.no_grad()
def text_source(bundle, texts):
    """C0/C1 oracle only: exact native forward_cache_update_text entry."""
    model = bundle.model; decoder = model.language_model.model
    device = decoder.embed_tokens.weight.device
    inputs, lengths, positions = model.prepare_prompts([0]*len(texts), [0]*len(texts), texts,
                                                       bundle.tokenizer, bundle.token_ids)
    cache = model.forward_cache_update_text(empty_cache(decoder), **device_inputs(inputs, device))
    source_positions = tuple(inputs['packed_text_position_ids'].tolist())
    source = tuple(replace(kv, positions=source_positions, token_roles=('text',)*len(kv.keys))
                   for kv in cache_kv(cache, lengths))
    return source, inputs, positions


@contextmanager
def capture_gen(decoder, conditional_cache, *, t=None, round_index=0,
                positions=None, detach=True, source_policy='full_query'):
    """Observe native normalized GEN input KV without mutating its layout.

    Capture only the conditional branch. GEN image queries retain native GEN
    expert parameters, K norm and positions. No prompt KV enters the source.
    """
    if source_policy not in ('full_query', 'image_only'):
        raise ValueError('unknown GEN source policy')
    captured = {}; handles = []
    def observer(index):
        def hook(attention, args, kwargs):
            if kwargs.get('mode') != 'gen' or kwargs.get('past_key_values') is not conditional_cache:
                return
            _, keys, values = project(attention, kwargs['packed_query_sequence'], 'gen',
                                      kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes'])
            keys = rotate(keys, kwargs['packed_query_position_embeddings']).to(torch.bfloat16)
            lengths = tuple(int(n) for n in kwargs['query_lens'].tolist())
            roles = ['boundary'] * len(keys)
            for token in kwargs['packed_vae_token_indexes'].tolist():
                roles[token] = 'image'
            pos = tuple(int(p) for p in positions.tolist()) if positions is not None else ()
            if source_policy == 'image_only':
                selected = kwargs['packed_vae_token_indexes']
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1]+length)
                lengths = tuple(sum(a <= i < b for i in selected.tolist())
                                for a, b in zip(offsets, offsets[1:]))
                keys, values = keys[selected], values[selected]
                roles = ['image'] * len(keys)
                pos = tuple(pos[i] for i in selected.tolist()) if pos else ()
            if detach:
                keys, values = keys.detach(), values.detach()
            captured[index] = KV(index, keys, values, lengths, True, 'visual', t,
                                 round_index, tuple(roles), pos, 'gen' if source_policy == 'image_only' else 'mixed_und_gen')
        return hook
    try:
        for i, layer in enumerate(decoder.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(observer(i), with_kwargs=True))
        yield captured
        if sorted(captured) != list(range(len(decoder.layers))):
            raise RuntimeError('native GEN source capture did not cover every layer')
    finally:
        for h in handles:
            h.remove()


def caption_logits(bundle, tokens, lengths, prefix, starts):
    """Frozen decoder, causal instruction/answer queries, input gradients live.

    tokens contain instruction + shifted answer prefix. Source images/text
    never appear here. Prefix is either C0 native text KV or Memory-only KV.
    """
    decoder = bundle.model.language_model.model
    hidden = decoder.embed_tokens(tokens)
    positions = torch.cat([torch.arange(s, s+n, device=tokens.device) for s, n in zip(starts, lengths)])
    rope = rope_for(decoder, hidden, positions)
    for i, layer in enumerate(decoder.layers):
        hidden = und_block(layer, hidden, rope, lengths, prefix[i] if prefix else None, causal=True)
    return bundle.model.language_model.lm_head(decoder.norm(hidden)).float()


def teacher_forcing(bundle, instruction_ids, answer_ids, prefix=None, start=0):
    """One sample; first answer is predicted by last instruction token."""
    import torch.nn.functional as F
    if not instruction_ids or not answer_ids:
        raise ValueError('instruction and target answer must be nonempty')
    device = bundle.model.language_model.model.embed_tokens.weight.device
    query = torch.tensor(instruction_ids + answer_ids[:-1], device=device)
    logits = caption_logits(bundle, query, (len(query),), prefix, [start])
    selected = logits[len(instruction_ids)-1:]
    targets = torch.tensor(answer_ids, device=device)
    return F.cross_entropy(selected, targets), selected


@torch.no_grad()
def free_caption(bundle, instruction_ids, prefix=None, start=0, max_tokens=96):
    device = bundle.model.language_model.model.embed_tokens.weight.device
    generated = []
    for _ in range(max_tokens):
        ids = torch.tensor(instruction_ids + generated, device=device)
        token = int(caption_logits(bundle, ids, (len(ids),), prefix, [start])[-1].argmax())
        if token == bundle.token_ids['eos_token_id']:
            break
        generated.append(token)
    return bundle.tokenizer.decode(generated, skip_special_tokens=True), generated


@contextmanager
def gen_memory_call(decoder, conditional_cache, bank, prompt_lengths, prompt_positions, masked=False):
    """Keep native prompt; append 16 slots for ONE conditional decoder call.

    CFG branches remain native. G-layout reserves identical indices/positions
    while excluding Memory from softmax. No bank persists after this context.
    """
    if len(bank.layers) != len(decoder.layers):
        raise ValueError('GEN must receive full-depth Memory')
    if len(prompt_lengths) != len(bank.layers[0].lengths):
        raise ValueError('GEN packed sample counts differ')
    memory = consume_bank(decoder, bank, prompt_positions)
    original_decoder = decoder.forward_inference
    originals = []
    call_count = 0

    def decoder_call(self, *args, **kwargs):
        nonlocal call_count
        if args:
            raise ValueError('diagnostic GEN entry requires keyword arguments')
        if kwargs.get('mode') == 'gen' and kwargs.get('past_key_values') is conditional_cache:
            call_count += 1
            if call_count > 1:
                raise RuntimeError('Memory is limited to one denoiser conditional call')
            kwargs = dict(kwargs)
            qlens = kwargs['query_lens'].tolist()
            if tuple(kwargs['key_values_lens'].tolist()) != tuple(prompt_lengths):
                raise ValueError('conditional prompt layout changed')
            chunks = kwargs['packed_query_position_ids'].split(qlens)
            kwargs['packed_query_position_ids'] = torch.cat([p+bank.slots for p in chunks])
        return original_decoder(**kwargs)

    def attention_call(index, original):
        def call(self, *args, **kwargs):
            if args:
                raise ValueError('diagnostic attention requires keyword arguments')
            if kwargs.get('mode') != 'gen' or kwargs.get('past_key_values') is not conditional_cache:
                return original(**kwargs)
            if kwargs.get('update_past_key_values', True):
                raise ValueError('GEN Memory consumption cannot persist its KV')
            if masked:
                # All Memory keys are excluded. Preserve the 16-position shift
                # in decoder_call, but use the exact native unmasked kernel on
                # prompt/self keys. An extra all-true mask selects a different
                # Ascend SDPA reduction and can accumulate BF16 drift.
                return original(**kwargs)
            h = kwargs['packed_query_sequence']
            ti, ii = kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes']
            q, k, v = project(self, h, 'gen', ti, ii)
            rope = kwargs['packed_query_position_embeddings']
            q, k, v = rotate(q, rope).to(torch.bfloat16), rotate(k, rope).to(torch.bfloat16), v.to(torch.bfloat16)
            qlens = tuple(kwargs['query_lens'].tolist())
            pk = conditional_cache.key_cache[index]; pv = conditional_cache.value_cache[index]
            mk, mv = memory[index].keys, memory[index].values
            if sum(prompt_lengths) != len(pk):
                raise ValueError('prompt cache lengths do not match')
            all_k = []; all_v = []; visible = []; klens = []
            for prompt_k, prompt_v, mem_k, mem_v, self_k, self_v in zip(
                pk.split(prompt_lengths), pv.split(prompt_lengths), mk.split(memory[index].lengths),
                mv.split(memory[index].lengths), k.split(qlens), v.split(qlens)):
                all_k.append(torch.cat((prompt_k, mem_k, self_k)))
                all_v.append(torch.cat((prompt_v, mem_v, self_v)))
                mask = torch.ones(len(all_k[-1]), device=h.device, dtype=torch.bool)
                if masked:
                    mask[len(prompt_k):len(prompt_k)+bank.slots] = False
                visible.append(mask); klens.append(len(mask))
            result = packed_attention(q, torch.cat(all_k), torch.cat(all_v), qlens, klens,
                                      native_unmasked=True).reshape(len(h), -1)
            out = torch.zeros_like(h)
            out[ti] = self.o_proj(result[ti]); out[ii] = self.o_proj_moe_gen(result[ii])
            return out, conditional_cache
        return call
    try:
        decoder.forward_inference = MethodType(decoder_call, decoder)
        for i, layer in enumerate(decoder.layers):
            attn = layer.self_attn
            originals.append((attn, attn.forward_inference))
            attn.forward_inference = MethodType(attention_call(i, attn.forward_inference), attn)
        yield
        if call_count != 1:
            raise RuntimeError('Memory must be consumed exactly once')
    finally:
        decoder.forward_inference = original_decoder
        for attn, original in originals:
            attn.forward_inference = original
