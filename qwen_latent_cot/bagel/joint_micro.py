"""Same-layer GEN/UND micro steps using both states from the start of each step.

Full prompt capacity and native positions are retained. P is immutable; GEN
reads M, while UND reads P + G + live self. The two branch evaluations use
independent temporary caches. States continue across layers, never back to the
window entrance. Window layers use K steps of size 1/K; suffix layers use one
native step. Memory is local to this denoiser call.
"""
import torch
from .layerwise_memory import LayerKV, native_context
from .native_gen import project_gen
from .native_und import project_und
from .modeling.bagel.qwen2_navit import BaseNavitOutputWithPast


def residual_step(before, native_output, steps):
    # K=1 must preserve the native branch output exactly, including BF16 rounding.
    return native_output if steps == 1 else before + (native_output - before) / steps


def run_joint_micro(loop, kwargs, runtime):
    cfg, decoder = runtime.config, loop.decoder
    cache = kwargs['past_key_values']
    seed = loop.seeds.get(cache)
    if seed is None:
        if int(kwargs['key_values_lens'].sum()):
            raise RuntimeError('joint micro loop requires native full-prompt prefill')
        return runtime.original(**kwargs)  # Native text-removed CFG bypass.
    prompt_lengths = tuple(kwargs['key_values_lens'].tolist())
    gen_lengths = tuple(kwargs['query_lens'].tolist())
    if (not seed.full_prompt or seed.start_layer != cfg.start_layer
            or seed.lengths != prompt_lengths or len(prompt_lengths) != len(gen_lengths)):
        raise ValueError('joint Memory seed differs from the prompt layout; prefill again')
    if any(i not in seed.layer_hidden for i in range(cfg.start_layer, len(decoder.layers))):
        raise ValueError('joint loop requires native hidden anchors at every used layer')
    if not len(seed.hidden):
        return runtime.original(**kwargs)
    if runtime.increment_observer is not None or runtime.kv_observer is not None:
        raise ValueError('joint micro loops use joint_observer; legacy round observers do not apply')

    gen = kwargs['packed_query_sequence']
    cos, sin = decoder.rotary_emb(gen, kwargs['packed_query_position_ids'].unsqueeze(0))
    gen_rope = cos.squeeze(0), sin.squeeze(0)
    cos, sin = decoder.rotary_emb(seed.hidden, seed.positions.unsqueeze(0))
    memory_rope = cos.squeeze(0), sin.squeeze(0)
    text, image = kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes']
    layer_kwargs = {k: v for k, v in kwargs.items()
                    if k not in ('packed_query_sequence', 'packed_query_position_ids')}
    layer_kwargs['packed_query_position_embeddings'] = gen_rope
    for index in range(cfg.start_layer):
        gen, _ = decoder.layers[index].forward_inference(packed_query_sequence=gen, **layer_kwargs)
    memory = seed.layer_hidden[cfg.start_layer].clone()
    special = seed.special_mask

    def evaluate(index, state, lengths, rope, additions, mode, include_prompt):
        temporary, klens, query, past = native_context(
            cache, index, prompt_lengths, additions, lengths, gen.device, include_prompt)
        call = dict(packed_query_sequence=state,
            query_lens=torch.tensor(lengths, device=gen.device, dtype=torch.int32),
            packed_query_position_embeddings=rope, packed_query_indexes=query,
            past_key_values=temporary, key_values_lens=klens, packed_key_value_indexes=past,
            update_past_key_values=False, is_causal=False, mode=mode)
        if mode == 'gen':
            call.update(packed_text_indexes=text, packed_vae_token_indexes=image)
        return decoder.layers[index].forward_inference(**call)[0]

    for index in range(cfg.start_layer, len(decoder.layers)):
        layer = decoder.layers[index]
        steps = cfg.micro_steps if index < cfg.end_layer else 1
        phase = 'body' if index < cfg.end_layer else 'suffix'
        reference = seed.layer_hidden[index]
        memory = torch.where(special[:, None], reference, memory)
        native = LayerKV(index, cache.key_cache[index][seed.source_indexes],
                         cache.value_cache[index][seed.source_indexes], seed.lengths)
        for micro in range(steps):
            # Jacobi update: all reads refer to these two inputs, not to either output.
            before_gen, before_memory = gen, memory
            gk, gv = project_gen(layer, before_gen, gen_rope, text, image)
            _, mk, mv = project_und(layer, before_memory, memory_rope)
            gen_kv = LayerKV(index, gk, gv, gen_lengths)
            memory_kv = LayerKV(index,
                torch.where(special[:, None, None], native.keys, mk),
                torch.where(special[:, None, None], native.values, mv), seed.lengths)
            gen_native = evaluate(index, before_gen, gen_lengths, gen_rope,
                                  [memory_kv], 'gen', False)
            memory_native = evaluate(index, before_memory, seed.lengths, memory_rope,
                                     [gen_kv], 'und', True)
            gen = residual_step(before_gen, gen_native, steps)
            memory = torch.where(special[:, None], reference,
                                 residual_step(before_memory, memory_native, steps))
            if runtime.joint_observer is not None:
                runtime.joint_observer(layer=index, phase=phase, micro=micro, micro_steps=steps,
                    step_scale=1 / steps, gen_before=before_gen, memory_before=before_memory,
                    gen_native=gen_native, memory_native=memory_native,
                    gen_after=gen, memory_after=memory, gen_kv=gen_kv, memory_kv=memory_kv,
                    native_prompt_kv=native, special_mask=special,
                    image_indexes=image, text_indexes=text)
            if runtime.diagnostics_enabled:
                from .memory_stats import memory_slot_stats
                runtime.diagnostics.append(dict(phase=phase, layer=index, micro=micro,
                    micro_steps=steps, step_scale=1 / steps, topology='joint_micro',
                    read_state='start_of_micro', progress=runtime.progress,
                    memory_lengths=list(seed.lengths), gen_reads_prompt=False,
                    und_reads_prompt=True, und_reads_gen=True,
                    gen_update_ratio=float((gen-before_gen).float().norm()
                        / before_gen.float().norm().clamp_min(1e-12)),
                    memory_update_ratio=float((memory-before_memory).float().norm()
                        / before_memory.float().norm().clamp_min(1e-12)),
                    **memory_slot_stats(memory)))

    normalized = torch.zeros_like(gen)
    normalized[text] = decoder.norm(gen[text])
    normalized[image] = decoder.norm_moe_gen(gen[image])
    return BaseNavitOutputWithPast(packed_query_sequence=normalized, past_key_values=cache)
