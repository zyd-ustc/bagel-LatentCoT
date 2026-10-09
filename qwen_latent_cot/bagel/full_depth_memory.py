"""Continuous full-depth UND Memory, with final hidden recycled between rounds.

GEN computation and conditioning layout remain identical to the legacy path.
Only body layers have current GEN KV. All UND layers read fixed native P and
live self KV. Body KV uses updated hidden; suffix KV is the native block input.
"""
import torch
from .layerwise_memory import LayerKV,native_context
from .native_und import project_und
from .modeling.bagel.qwen2_navit import BaseNavitOutputWithPast


def run_und_state(loop,kwargs,runtime):
    cfg,decoder=runtime.config,loop.decoder
    cache=kwargs['past_key_values'];seed=loop.seeds.get(cache)
    if seed is None:
        if int(kwargs['key_values_lens'].sum()):
            raise RuntimeError('UND state requires native full-prompt prefill')
        return runtime.original(**kwargs)  # Text-removed CFG keeps the native path.
    prompt_lengths=tuple(kwargs['key_values_lens'].tolist())
    gen_lengths=tuple(kwargs['query_lens'].tolist())
    if (not seed.full_prompt or seed.start_layer!=cfg.start_layer
            or seed.lengths!=prompt_lengths or len(seed.lengths)!=len(gen_lengths)):
        raise ValueError('UND state seed differs from native prompt layout; prefill again')
    layers=range(cfg.start_layer,len(decoder.layers))
    if any(i not in seed.layer_hidden for i in layers):
        raise ValueError('UND state requires native hidden seeds at every layer')
    if not len(seed.hidden):return runtime.original(**kwargs)
    hidden=kwargs['packed_query_sequence']
    cos,sin=decoder.rotary_emb(hidden,kwargs['packed_query_position_ids'].unsqueeze(0))
    gen_rope=cos.squeeze(0),sin.squeeze(0)
    cos,sin=decoder.rotary_emb(seed.hidden,seed.positions.unsqueeze(0))
    memory_rope=cos.squeeze(0),sin.squeeze(0)
    layer_kwargs={k:v for k,v in kwargs.items() if k not in ('packed_query_sequence','packed_query_position_ids')}
    layer_kwargs['packed_query_position_embeddings']=gen_rope
    for index in range(cfg.start_layer):
        hidden,_=decoder.layers[index].forward_inference(packed_query_sequence=hidden,**layer_kwargs)
    entrance=hidden.clone()
    native={i:LayerKV(i,cache.key_cache[i][seed.source_indexes],cache.value_cache[i][seed.source_indexes],seed.lengths)
            for i in layers}
    # These are local to this call. Initial prefill tensors remain immutable.
    writer=seed.layer_hidden[cfg.start_layer].clone()
    bank=dict(native)

    def pin_hidden(index,state):
        return torch.where(seed.special_mask[:,None],seed.layer_hidden[index],state)

    def pin_kv(index,keys,values):
        mask=seed.special_mask[:,None,None];reference=native[index]
        return LayerKV(index,torch.where(mask,reference.keys,keys),torch.where(mask,reference.values,values),seed.lengths)

    def native_layer(index,state,lengths,rope,additions,mode,store=False,include_prompt=True):
        temporary,klens,query,past=native_context(cache,index,prompt_lengths,additions,lengths,hidden.device,include_prompt)
        call=dict(packed_query_sequence=state,query_lens=torch.tensor(lengths,device=hidden.device,dtype=torch.int32),
            packed_query_position_embeddings=rope,packed_query_indexes=query,past_key_values=temporary,
            key_values_lens=klens,packed_key_value_indexes=past,update_past_key_values=store,is_causal=False,mode=mode)
        if mode=='gen':
            call.update(packed_text_indexes=kwargs['packed_text_indexes'],packed_vae_token_indexes=kwargs['packed_vae_token_indexes'])
        output,_=decoder.layers[index].forward_inference(**call)
        kv=(LayerKV(index,temporary.key_cache[index][query],temporary.value_cache[index][query],tuple(lengths))
            if store else None)
        return output,kv

    def observe_read(index,phase):
        if runtime.kv_observer is not None:
            runtime.kv_observer(event='final_read',layer=index,phase=phase,depth=cfg.extra_rounds,
                current=bank[index],reference=native[index],special_mask=seed.special_mask)

    for round_index in range(cfg.extra_rounds+1):
        hidden=entrance.clone();gen_kv={}
        for index in range(cfg.start_layer,cfg.end_layer):
            replacing=round_index>0
            if replacing and round_index==cfg.extra_rounds:observe_read(index,'body')
            hidden,current=native_layer(index,hidden,gen_lengths,gen_rope,
                [bank[index]] if replacing else [],'gen',store=round_index<cfg.extra_rounds,
                include_prompt=not replacing)
            if current is not None:gen_kv[index]=current
            if runtime.diagnostics_enabled:
                runtime.diagnostics.append(dict(phase='gen',layer=index,round=round_index,
                    gen_reads_memory=replacing,memory_slots_per_sample=list(seed.lengths),
                    prompt_read_per_sample=[not replacing]*len(seed.lengths),
                    memory_read_kind='continuous_full_depth_und_state' if replacing else 'native_prompt',
                    progress=runtime.progress,branch='conditional'))
        if round_index==cfg.extra_rounds:break
        for index in layers:
            before=pin_hidden(index,writer);old_kv=bank[index]
            additions=[gen_kv[index]] if index in gen_kv else []
            suffix=index>=cfg.end_layer
            writer,current=native_layer(index,before,seed.lengths,memory_rope,additions,'und',store=suffix)
            writer=pin_hidden(index,writer)
            if suffix:
                # Preserve legacy suffix: KV is written by native attention
                # from the layer input, while its output continues the writer.
                bank[index]=pin_kv(index,current.keys,current.values)
            else:
                _,keys,values=project_und(decoder.layers[index],writer,memory_rope)
                bank[index]=pin_kv(index,keys,values)
            if runtime.kv_observer is not None:
                runtime.kv_observer(event='writer_update',layer=index,phase='body',depth=cfg.extra_rounds,
                    from_round=round_index,to_round=round_index+1,current=bank[index],reference=old_kv,
                    special_mask=seed.special_mask,hidden_before=before,hidden_after=writer)
            if runtime.diagnostics_enabled:
                from .memory_stats import memory_slot_stats
                for sample,(a,b) in enumerate(zip(before.split(seed.lengths),writer.split(seed.lengths))):
                    runtime.diagnostics.append(dict(phase='writer',layer=index,round=round_index,sample=sample,
                        progress=runtime.progress,branch='conditional',feedback_stored=index in gen_kv,
                        memory_update='full_depth',hidden_update_ratio=float((b-a).float().norm()/a.float().norm().clamp_min(1e-12)),
                        **memory_slot_stats(b)))
    for index in range(cfg.end_layer,len(decoder.layers)):
        observe_read(index,'suffix')
        hidden,_=native_layer(index,hidden,gen_lengths,gen_rope,[bank[index]],'gen',include_prompt=False)
    normalized=torch.zeros_like(hidden)
    text,image=kwargs['packed_text_indexes'],kwargs['packed_vae_token_indexes']
    normalized[text]=decoder.norm(hidden[text]);normalized[image]=decoder.norm_moe_gen(hidden[image])
    return BaseNavitOutputWithPast(packed_query_sequence=normalized,past_key_values=cache)
