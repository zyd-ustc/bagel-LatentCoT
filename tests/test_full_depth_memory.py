"""Numerical contracts for continuous UND traversal and recurrent feedback."""
from dataclasses import replace
import torch
import pytest
from helpers import prepared
from qwen_latent_cot.bagel.native_und import project_und


@pytest.mark.parametrize('memory_update',['full_depth','full_depth_restart'])
@pytest.mark.parametrize('rounds',[1,2,4])
def test_all_28_und_layers_are_connected_with_configured_round_entrance(rounds,memory_update):
    model,kwargs,runtime=prepared(rounds=rounds,start=0,end=8,depth=28,special_tokens=True)
    runtime.config=replace(runtime.config,memory_update=memory_update)
    decoder=runtime.decoder;seed=runtime.layerwise.seeds[kwargs['past_key_values']]
    originals=[];calls=[];events=[]
    runtime.kv_observer=lambda **event: events.append(event)
    for index,layer in enumerate(decoder.layers):
        original=layer.forward_inference;originals.append(original)
        def observed(*,i=index,fn=original,**kw):
            before=kw['packed_query_sequence'].clone()
            result=fn(**kw)
            calls.append((i,kw['mode'],before,result[0].clone(),kw['key_values_lens'].tolist()))
            return result
        layer.forward_inference=observed
    try:
        result=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(result).all()
        und=[c for c in calls if c[1]=='und']
        assert [c[0] for c in und]==list(range(28))*rounds
        content=~seed.special_mask
        assert torch.equal(und[0][2],seed.layer_hidden[0])
        for previous,current in zip(und,und[1:]):
            if current[0]==0 and memory_update=='full_depth_restart':
                assert torch.equal(current[2],seed.layer_hidden[0])
            else:
                assert torch.equal(current[2][content],previous[3][content])
            assert torch.equal(current[2][seed.special_mask],seed.layer_hidden[current[0]][seed.special_mask])
        assert all(c[4]==[6,10] for c in und if c[0]<8)
        assert all(c[4]==list(seed.lengths) for c in und if c[0]>=8)
        # key_values_lens counts past KV; native attention adds live self KV.
        gen=[c for c in calls if c[1]=='gen']
        entry=[c[2] for c in gen if c[0]==0]
        assert len(entry)==rounds+1 and all(torch.equal(entry[0],x) for x in entry)
        assert all(sum(c[0]==i for c in gen)==(rounds+1 if i<8 else 1) for i in range(28))
        cos,sin=decoder.rotary_emb(seed.hidden,seed.positions.unsqueeze(0));rope=cos.squeeze(0),sin.squeeze(0)
        writes=[e for e in events if e['event']=='writer_update']
        assert len(writes)==rounds*28
        for e in writes:
            state=e['hidden_after'] if e['layer']<8 else e['hidden_before']
            _,k,v=project_und(decoder.layers[e['layer']],state,rope)
            assert torch.equal(k[content],e['current'].keys[content])
            assert torch.equal(v[content],e['current'].values[content])
        for e in (x for x in events if x['event']=='final_read'):
            latest=next(x for x in reversed(writes) if x['layer']==e['layer'])
            assert torch.equal(latest['current'].keys,e['current'].keys)
            assert torch.equal(latest['current'].values,e['current'].values)
    finally:
        for layer,original in zip(decoder.layers,originals):layer.forward_inference=original
        runtime.close()


@pytest.mark.parametrize("memory_update",["full_depth","full_depth_restart"])
@pytest.mark.parametrize("device_type",["cpu","npu"])
def test_full_depth_repeatability_cache_isolation_and_native_r0(device_type,memory_update):
    from test_npu_runtime import npu_device
    device="cpu" if device_type=="cpu" else npu_device()
    model,kwargs,runtime=prepared(rounds=2,start=0,end=8,depth=28,special_tokens=True,device=device)
    runtime.config=replace(runtime.config,memory_update=memory_update)
    decoder=runtime.decoder;cache=kwargs['past_key_values']
    keys={i:k.clone() for i,k in cache.key_cache.items()}
    values={i:v.clone() for i,v in cache.value_cache.items()}
    try:
        native=runtime.original(**kwargs).packed_query_sequence
        a=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(a,decoder.forward_inference(**kwargs).packed_query_sequence)
        changed=kwargs['packed_query_sequence'].clone();changed[4:]+=torch.linspace(-3,3,changed.shape[-1],device=changed.device)
        b=decoder.forward_inference(**{**kwargs,'packed_query_sequence':changed}).packed_query_sequence
        assert torch.equal(a[:4],b[:4]) and not torch.equal(a[4:],b[4:])
        runtime.config=replace(runtime.config,memory_update='legacy_layerwise')
        legacy=decoder.forward_inference(**kwargs).packed_query_sequence
        assert not torch.equal(a,legacy)
        runtime.config=replace(runtime.config,memory_update=memory_update,extra_rounds=0)
        assert torch.equal(native,decoder.forward_inference(**kwargs).packed_query_sequence)
        assert all(torch.equal(cache.key_cache[i],k) and torch.equal(cache.value_cache[i],values[i]) for i,k in keys.items())
    finally:runtime.close()


def test_restarted_r1_is_exactly_the_existing_r1():
    _,kwargs,runtime=prepared(rounds=1,start=0,end=8,depth=28,special_tokens=True)
    events=[]
    runtime.kv_observer=lambda **event: events.append(event)
    try:
        runtime.config=replace(runtime.config,memory_update='full_depth')
        a=runtime.decoder.forward_inference(**kwargs).packed_query_sequence
        first=list(events);events.clear()
        runtime.config=replace(runtime.config,memory_update='full_depth_restart')
        b=runtime.decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(a,b)
        assert len(first)==len(events)
        for x,y in zip(first,events):
            assert torch.equal(x['current'].keys,y['current'].keys)
            assert torch.equal(x['current'].values,y['current'].values)
    finally:runtime.close()


def test_restarted_second_writer_uses_first_memory_increment_through_gen_kv():
    # Replace ONLY second-round stored GEN KV by round0 KV. This numerical
    # diagnostic isolates the feedback consumed by the second UND writer.
    _,kwargs,runtime=prepared(rounds=2,start=0,end=8,depth=28,special_tokens=True)
    runtime.config=replace(runtime.config,memory_update='full_depth_restart')
    decoder=runtime.decoder
    def run(ablate):
        gen0={};counts={};writes=[];originals=[];changed=[]
        runtime.kv_observer=lambda **e: writes.append(e) if e['event']=='writer_update' else None
        for index,layer in enumerate(decoder.layers):
            original=layer.forward_inference;originals.append(original)
            def observed(*,i=index,fn=original,**kw):
                result=fn(**kw)
                if kw['mode']=='gen' and kw['update_past_key_values']:
                    count=counts.get(i,0);counts[i]=count+1
                    cache=kw['past_key_values'];query=kw['packed_query_indexes']
                    kv=(cache.key_cache[i][query].clone(),cache.value_cache[i][query].clone())
                    if count==0:gen0[i]=kv
                    else:
                        changed.append(not torch.equal(kv[0],gen0[i][0]) or not torch.equal(kv[1],gen0[i][1]))
                        if ablate:
                            cache.key_cache[i][query]=gen0[i][0]
                            cache.value_cache[i][query]=gen0[i][1]
                return result
            layer.forward_inference=observed
        try:
            decoder.forward_inference(**kwargs)
            assert any(changed), 'M1 must change the GEN KV fed back to writer2'
            return writes
        finally:
            for layer,original in zip(decoder.layers,originals):layer.forward_inference=original
    try:
        normal=run(False);without_feedback=run(True)
        for a,b in zip(normal[:28],without_feedback[:28]):
            assert torch.equal(a['hidden_after'],b['hidden_after'])
        seed=runtime.layerwise.seeds[kwargs['past_key_values']]
        assert torch.equal(normal[28]['hidden_before'],seed.layer_hidden[0])
        assert torch.equal(without_feedback[28]['hidden_before'],seed.layer_hidden[0])
        assert any(not torch.equal(a['hidden_after'],b['hidden_after'])
                   for a,b in zip(normal[28:36],without_feedback[28:36]))
    finally:runtime.close()
