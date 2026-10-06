from dataclasses import replace
import pytest
import torch
from test_internal_loop import fixture
from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime
from qwen_latent_cot.bagel.layerwise_memory import LayerKV, native_context, select_content_indexes
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache


def prepared(batch=2, empty_first=False, rounds=1, mode='LAYERWISE_MEMORY_KV', device='cpu', slots=3, start=1, special_tokens=False):
    model, kwargs = fixture(batch=batch,device=device)
    decoder = model.language_model.model
    runtime = InternalLoopRuntime(model, LoopConfig(mode=mode, extra_rounds=rounds,
        start_layer=start, end_layer=3, memory_slots=slots), diagnostics=True)
    lengths = kwargs['key_values_lens'].tolist()
    tokens = torch.tensor([5,6,7,8,9] if batch==2 else [5,6,7],device=device)
    if empty_first: tokens[:lengths[0]]=0
    if special_tokens:
        offset=0
        for n in lengths:
            tokens[offset]=0;tokens[offset+n-1]=1;offset+=n
    cache = NaiveCache(4)
    positions = torch.tensor([j for n in lengths for j in range(n)],device=device)
    runtime.begin_prefill(cache, tokens, lengths, {0,1})
    try:
        decoder.forward_inference(packed_query_sequence=decoder.embed_tokens(tokens),
            query_lens=torch.tensor(lengths,dtype=torch.int32,device=device), packed_query_position_ids=positions,
            packed_query_indexes=torch.arange(len(tokens),device=device), past_key_values=cache,
            key_values_lens=torch.zeros(batch,dtype=torch.int32,device=device),
            packed_key_value_indexes=torch.empty(0,dtype=torch.long,device=device),
            update_past_key_values=True, is_causal=True, mode='und')
    finally: runtime.end_prefill()
    return model, {**kwargs,'past_key_values':cache}, runtime


@pytest.mark.parametrize('rounds',[1,2,3])
@pytest.mark.parametrize('slots',[0,1])
def test_full_static_replacement_exact_native_without_token_compression(rounds,slots):
    model,kwargs,runtime=prepared(mode='LAYERWISE_FULL_SEED_REPLACE',rounds=rounds,
                                  slots=slots,start=0,special_tokens=True)
    try:
        decoder=model.language_model.model;cache=kwargs['past_key_values']
        seed=runtime.layerwise.seeds[cache]
        assert seed.full_prompt and seed.lengths==(2,3)
        assert seed.source_indexes.tolist()==list(range(5))
        assert seed.positions.tolist()==[0,1,0,1,2]
        assert seed.special_mask.tolist()==[True,True,True,False,True]
        weights={k:v.clone() for k,v in model.language_model.state_dict().items()}
        native=runtime.original(**kwargs).packed_query_sequence
        before={i:(cache.key_cache[i],cache.key_cache[i].clone(),cache.value_cache[i].clone()) for i in range(4)}
        actual=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(actual,native)
        for i,(obj,k,v) in before.items():
            assert cache.key_cache[i] is obj and torch.equal(obj,k) and torch.equal(cache.value_cache[i],v)
        assert all(torch.equal(v,weights[k]) for k,v in model.language_model.state_dict().items())
        assert not any(row['phase']=='writer' for row in runtime.diagnostics)
        reads=[row for row in runtime.diagnostics if row['phase']=='gen' and row['round']>0]
        assert reads and all(row['memory_slots_per_sample']==[2,3] for row in reads)
        assert all(row['prompt_read_per_sample']==[False,False] for row in reads)
    finally:runtime.close()


@pytest.mark.parametrize('rounds',[1,2,3])
def test_full_dynamic_same_length_special_kv_pinned_and_content_changes(rounds):
    model,kwargs,runtime=prepared(mode='LAYERWISE_FULL_MEMORY_REPLACE',rounds=rounds,
                                  slots=0,start=0,special_tokens=True)
    decoder=model.language_model.model;cache=kwargs['past_key_values'];seed=runtime.layerwise.seeds[cache]
    originals=[layer.forward_inference for layer in decoder.layers];reads=[]
    def wrap(i):
        def observed(**kw):
            context=kw['past_key_values'].key_cache[i]
            if kw['mode']=='gen':
                reads.append((i,context.clone(),kw['past_key_values'].value_cache[i].clone(),kw['key_values_lens'].tolist()))
            return originals[i](**kw)
        return observed
    for i,layer in enumerate(decoder.layers):layer.forward_inference=wrap(i)
    try:
        native=runtime.original(**kwargs).packed_query_sequence;reads.clear()
        output=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(output).all() and not torch.equal(output,native)
        assert torch.equal(output[:4],native[:4])  # The special-only prompt stays native.
        body=[r for r in reads if r[0]==2];suffix=[r for r in reads if r[0]==3]
        assert len(body)==rounds+1 and len(suffix)==1
        for i,k,v,lengths in body[1:]+suffix:
            assert lengths==[2,3] and len(k)==5  # No P+M concatenation on GEN reads.
            assert torch.equal(k[seed.special_mask],cache.key_cache[i][seed.special_mask])
            assert torch.equal(v[seed.special_mask],cache.value_cache[i][seed.special_mask])
            assert not torch.equal(k[~seed.special_mask],cache.key_cache[i][~seed.special_mask])
        assert torch.equal(output,decoder.forward_inference(**kwargs).packed_query_sequence)
        runtime.config=replace(runtime.config,extra_rounds=0)
        assert torch.equal(decoder.forward_inference(**kwargs).packed_query_sequence,native)
    finally:
        for layer,fn in zip(decoder.layers,originals):layer.forward_inference=fn
        runtime.close()


def test_full_dynamic_sample_isolation_null_cfg_and_current_gen_dependence():
    model,kwargs,runtime=prepared(mode='LAYERWISE_FULL_MEMORY_REPLACE',rounds=3,slots=0,start=0)
    try:
        decoder=model.language_model.model
        a=decoder.forward_inference(**kwargs).packed_query_sequence
        changed=kwargs['packed_query_sequence'].clone();changed[4:]+=torch.linspace(-3,3,changed.shape[-1])
        b=decoder.forward_inference(**{**kwargs,'packed_query_sequence':changed}).packed_query_sequence
        assert torch.equal(a[:4],b[:4]) and not torch.equal(a[4:],b[4:])
        null={**kwargs,'past_key_values':NaiveCache(4),'key_values_lens':torch.zeros(2,dtype=torch.int32),
              'packed_key_value_indexes':torch.empty(0,dtype=torch.long),'packed_query_indexes':torch.arange(11)}
        assert torch.equal(decoder.forward_inference(**null).packed_query_sequence,
                           runtime.original(**null).packed_query_sequence)
        runtime.config=replace(runtime.config,mode='LAYERWISE_MEMORY_REPLACE',memory_slots=3)
        with pytest.raises(ValueError,match='seed policy'):decoder.forward_inference(**kwargs)
    finally:runtime.close()


@pytest.mark.parametrize('mode',['LAYERWISE_FULL_MEMORY_REPLACE','LAYERWISE_FULL_SEED_REPLACE'])
def test_cuda_full_prompt_replacement_contract(mode):
    if not torch.cuda.is_available():pytest.skip('user-run full replacement CUDA contract')
    model,kwargs,runtime=prepared(mode=mode,device='cuda',rounds=3,slots=0,start=0,special_tokens=True)
    try:
        native=runtime.original(**kwargs).packed_query_sequence
        output=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(output).all()
        if mode=='LAYERWISE_FULL_SEED_REPLACE':assert torch.equal(output,native)
        else:assert torch.equal(output[:4],native[:4]) and not torch.equal(output,native)
    finally:runtime.close()


def test_distinct_content_positions_and_layer_alignment_validation():
    indexes, lengths = select_content_indexes(torch.tensor([0,5,6,7,1,0,8,1]), [5,3], {0,1}, 2)
    assert indexes.tolist()==[1,3,6] and lengths==(2,1)
    cache=NaiveCache(4)
    kv=LayerKV(2,torch.randn(3,2,8),torch.randn(3,2,8),(2,1))
    with pytest.raises(ValueError,match='same native layer'):
        native_context(cache,1,[0,0],[kv],[4,7],'cpu')


def test_seed_is_native_layer_entrance_with_original_content_positions():
    from qwen_latent_cot.bagel.memory_stats import project_memory
    model,kwargs,runtime=prepared()
    try:
        cache=kwargs['past_key_values'];seed=runtime.layerwise.seeds[cache]
        assert seed.positions.tolist()==[0,1,0,1,2]
        decoder=model.language_model.model
        cos,sin=decoder.rotary_emb(seed.hidden,seed.positions.unsqueeze(0))
        _,k,v=project_memory(decoder.layers[1],seed.hidden,(cos.squeeze(0),sin.squeeze(0)))
        assert torch.equal(k,cache.key_cache[1][seed.source_indexes])
        assert torch.equal(v,cache.value_cache[1][seed.source_indexes])
        runtime.progress=1.1
        assert torch.equal(decoder.forward_inference(**kwargs).packed_query_sequence,
                           runtime.original(**kwargs).packed_query_sequence)
    finally:runtime.close()


@pytest.mark.parametrize('rounds',[1,2,4])
def test_native_no_read_exact_packed_cache_immutable_and_effect(rounds):
    model,kwargs,runtime=prepared(rounds=rounds)
    try:
        decoder=model.language_model.model
        expected=runtime.original(**kwargs).packed_query_sequence
        weights={k:v.clone() for k,v in model.language_model.state_dict().items()}
        cache=kwargs['past_key_values']
        copies={i:(k,k.clone(),cache.value_cache[i],cache.value_cache[i].clone()) for i,k in cache.key_cache.items()}
        actual=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(actual).all() and not torch.equal(actual,expected)
        for cfg in [replace(runtime.config,mode='LAYERWISE_KV_NO_READ'),
                    replace(runtime.config,extra_rounds=0),replace(runtime.config,memory_slots=0)]:
            runtime.config=cfg
            assert torch.equal(decoder.forward_inference(**kwargs).packed_query_sequence,expected)
        for i,(k,kcopy,v,vcopy) in copies.items():
            assert cache.key_cache[i] is k and cache.value_cache[i] is v
            assert torch.equal(k,kcopy) and torch.equal(v,vcopy)
        assert all(torch.equal(v,weights[k]) for k,v in model.language_model.state_dict().items())
    finally: runtime.close()


def test_fixed_entrances_actual_native_input_kv_same_layer_and_suffix_once():
    model,kwargs,runtime=prepared(rounds=2)
    calls=[];decoder=model.language_model.model
    originals=[layer.forward_inference for layer in decoder.layers]
    def wrap(i):
        def observed(**kw):
            before=kw['packed_query_sequence'].clone()
            context=kw['past_key_values'].key_cache[i]
            context=None if context is None else context.clone()
            out,cache=originals[i](**kw)
            current=None
            if kw['update_past_key_values']:
                current=cache.key_cache[i][kw['packed_query_indexes']].clone()
            calls.append((i,kw['mode'],before,context,current,kw['key_values_lens'].tolist()))
            return out,cache
        return observed
    for i,layer in enumerate(decoder.layers):layer.forward_inference=wrap(i)
    try:
        decoder.forward_inference(**kwargs)
        gen1=[c for c in calls if c[0]==1 and c[1]=='gen']
        und1=[c for c in calls if c[0]==1 and c[1]=='und']
        assert len(gen1)==3 and len(und1)==2
        assert all(torch.equal(c[2],gen1[0][2]) for c in gen1)
        assert all(torch.equal(c[2],und1[0][2]) for c in und1)
        assert all(c[5]==[2,3] for c in gen1)  # first layer has no grounded feedback
        assert len([c for c in calls if c[0]==0])==1
        assert len([c for c in calls if c[0]==3])==1
        writes=[c for c in calls if c[0]==2 and c[1]=='und']
        reads=[c for c in calls if c[0]==2 and c[1]=='gen']
        for write,read in zip(writes,reads[1:]):
            # Per sample original P followed by exactly the preceding native writer input KV.
            assert read[5]==[4,6]
            assert torch.equal(read[3][2:4],write[4][:2])
            assert torch.equal(read[3][7:10],write[4][2:])
        assert [c for c in calls if c[0]==3][0][5]==[2,3]
    finally:
        for layer,original in zip(decoder.layers,originals):layer.forward_inference=original
        runtime.close()


@pytest.mark.parametrize('empty_first',[False,True])
@pytest.mark.parametrize('mode',['LAYERWISE_MEMORY_KV','LAYERWISE_MEMORY_REPLACE','LAYERWISE_SEED_REPLACE'])
def test_sample_isolation_local_round_state_and_null_cfg(empty_first,mode):
    model,kwargs,runtime=prepared(empty_first=empty_first,mode=mode)
    try:
        decoder=model.language_model.model
        a=decoder.forward_inference(**kwargs).packed_query_sequence
        hidden=kwargs['packed_query_sequence'].clone();hidden[4:]+=3
        b=decoder.forward_inference(**{**kwargs,'packed_query_sequence':hidden}).packed_query_sequence
        assert torch.equal(a[:4],b[:4]) and not torch.equal(a[4:],b[4:])
        assert torch.equal(a,decoder.forward_inference(**kwargs).packed_query_sequence)
        if empty_first:
            assert torch.equal(a[:4],runtime.original(**kwargs).packed_query_sequence[:4])
        null={**kwargs,'past_key_values':NaiveCache(4),'key_values_lens':torch.zeros(2,dtype=torch.int32),
              'packed_key_value_indexes':torch.empty(0,dtype=torch.long),
              'packed_query_indexes':torch.arange(11)}
        assert torch.equal(decoder.forward_inference(**null).packed_query_sequence,
                           runtime.original(**null).packed_query_sequence)
        runtime.clear_prompt_state()
        with pytest.raises(RuntimeError,match='prefill'):
            decoder.forward_inference(**kwargs)
    finally:runtime.close()


def test_probe_exports_exact_read_bank_without_affecting_math():
    from qwen_latent_cot.bagel.memory_probe import ProbeCapture
    model,kwargs,runtime=prepared(batch=1)
    try:
        decoder=model.language_model.model
        expected=decoder.forward_inference(**kwargs).packed_query_sequence
        runtime.probe_capture=ProbeCapture([0])
        actual=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(actual,expected)
        assert set(runtime.probe_capture.layers[0])=={2}
        state=runtime.probe_capture.layers[0][2]
        assert not torch.equal(state['dynamic_k'],state['seed_k'])
        first=state['dynamic_k'].clone();seed=state['seed_k'].clone()
        changed=kwargs['packed_query_sequence'].clone()
        changed[kwargs['packed_vae_token_indexes']]+=2
        decoder.forward_inference(**{**kwargs,'packed_query_sequence':changed})
        updated=runtime.probe_capture.layers[0][2]
        assert torch.equal(updated['seed_k'],seed)
        assert not torch.equal(updated['dynamic_k'],first)  # writer depends on current GEN, not prompt alone
    finally:runtime.close()


@pytest.mark.parametrize('empty_first',[False,True])
def test_cuda_native_packed_writer_contract(empty_first):
    if not torch.cuda.is_available():pytest.skip('user-run CUDA contract')
    model,kwargs,runtime=prepared(device='cuda',empty_first=empty_first)
    try:
        native=runtime.original(**kwargs).packed_query_sequence
        actual=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(actual).all()
        if empty_first:assert torch.equal(actual[:4],native[:4])
        runtime.config=replace(runtime.config,mode='LAYERWISE_KV_NO_READ')
        assert torch.equal(model.language_model.model.forward_inference(**kwargs).packed_query_sequence,native)
    finally:runtime.close()


@pytest.mark.parametrize('rounds',[1,2])
def test_cpu_gradient_reaches_writer_seed_through_native_layer_kv(rounds):
    # This tests representation differentiability, not a supported training API.
    # Native weights stay frozen; no optimizer, update or training checkpoint.
    model,kwargs,runtime=prepared(batch=1,rounds=rounds)
    try:
        seed=runtime.layerwise.seeds[kwargs['past_key_values']]
        seed.hidden.requires_grad_(True)
        output=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        head=torch.linspace(-1,1,output.shape[-1])
        (output[kwargs['packed_vae_token_indexes']].float()*head).sum().backward()
        assert seed.hidden.grad is not None
        assert torch.isfinite(seed.hidden.grad).all() and seed.hidden.grad.float().norm()>0
        assert all(parameter.grad is None for parameter in model.language_model.parameters())
    finally:runtime.close()


@pytest.mark.parametrize('rounds',[1,2])
@pytest.mark.parametrize('mode',['LAYERWISE_MEMORY_REPLACE','LAYERWISE_SEED_REPLACE'])
def test_replacement_removes_prompt_reads_and_uses_exact_same_layer_bank(mode,rounds):
    model,kwargs,runtime=prepared(mode=mode,rounds=rounds,slots=1)
    decoder=model.language_model.model;cache=kwargs['past_key_values']
    seed=runtime.layerwise.seeds[cache]
    original=[layer.forward_inference for layer in decoder.layers]
    weights={k:v.clone() for k,v in model.language_model.state_dict().items()}
    prompt={i:(cache.key_cache[i],cache.key_cache[i].clone(),cache.value_cache[i].clone()) for i in range(4)}
    calls=[]
    def wrap(i):
        def observed(**kw):
            context=kw['past_key_values'].key_cache[i].clone()
            out,temporary=original[i](**kw)
            current=temporary.key_cache[i][kw['packed_query_indexes']].clone() if kw['update_past_key_values'] else None
            calls.append((i,kw['mode'],context,current,kw['key_values_lens'].tolist(),
                          kw['packed_query_position_embeddings']))
            return out,temporary
        return observed
    for i,layer in enumerate(decoder.layers):layer.forward_inference=wrap(i)
    try:
        result=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(result).all()
        for i in [1,2]:
            gens=[c for c in calls if c[0]==i and c[1]=='gen']
            assert len(gens)==rounds+1 and gens[0][4]==[2,3]
            for c in gens[1:]:
                assert c[4]==[1,1]  # Only K slots, no original P, no bank accumulation.
                assert all(torch.equal(a,b) for a,b in zip(c[5],gens[0][5]))
                if mode=='LAYERWISE_SEED_REPLACE' or i==1:
                    assert torch.equal(c[2],cache.key_cache[i][seed.source_indexes])
            if mode=='LAYERWISE_MEMORY_REPLACE' and i==2:
                writes=[c for c in calls if c[0]==i and c[1]=='und']
                assert len(writes)==rounds
                for writer,gen in zip(writes,gens[1:]):
                    assert torch.equal(gen[2],writer[3])
                assert writes[0][4]==[6,10]  # P + current GEN; writer still sees P.
                if rounds==2:assert writes[1][4]==[7,11]  # P + previous M + current GEN.
        suffix=[c for c in calls if c[0]==3 and c[1]=='gen']
        assert len(suffix)==1 and suffix[0][4]==[1,1]
        suffix_writes=[c for c in calls if c[0]==3 and c[1]=='und']
        if mode=='LAYERWISE_MEMORY_REPLACE':
            assert len(suffix_writes)==1 and suffix_writes[0][4]==[2,3]
            assert torch.equal(suffix[0][2],suffix_writes[0][3])
        else:
            assert not any(c[1]=='und' for c in calls)
            assert torch.equal(suffix[0][2],cache.key_cache[3][seed.source_indexes])
        assert len([c for c in calls if c[0]==0])==1
        for i,(obj,k,v) in prompt.items():
            assert cache.key_cache[i] is obj and torch.equal(obj,k) and torch.equal(cache.value_cache[i],v)
        assert all(torch.equal(v,weights[k]) for k,v in model.language_model.state_dict().items())
        runtime.config=replace(runtime.config,extra_rounds=0)
        assert torch.equal(decoder.forward_inference(**kwargs).packed_query_sequence,
                           runtime.original(**kwargs).packed_query_sequence)
    finally:
        for layer,fn in zip(decoder.layers,original):layer.forward_inference=fn
        runtime.close()


@pytest.mark.parametrize('mode',['LAYERWISE_MEMORY_REPLACE','LAYERWISE_SEED_REPLACE'])
def test_replacement_empty_sample_fallback_and_local_state(mode):
    model,kwargs,runtime=prepared(mode=mode,rounds=2,empty_first=True,slots=1)
    try:
        decoder=model.language_model.model
        a=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(a[:4],runtime.original(**kwargs).packed_query_sequence[:4])
        changed=kwargs['packed_query_sequence'].clone();changed[4:]+=3
        b=decoder.forward_inference(**{**kwargs,'packed_query_sequence':changed}).packed_query_sequence
        assert torch.equal(a[:4],b[:4]) and not torch.equal(a[4:],b[4:])
        assert torch.equal(a,decoder.forward_inference(**kwargs).packed_query_sequence)
    finally:runtime.close()


def test_replacement_feedback_dependence_and_cpu_gradient():
    model,kwargs,runtime=prepared(batch=1,mode='LAYERWISE_MEMORY_REPLACE',rounds=2,slots=1)
    try:
        decoder=model.language_model.model;seed=runtime.layerwise.seeds[kwargs['past_key_values']]
        seed.hidden.requires_grad_(True)
        result=decoder.forward_inference(**kwargs).packed_query_sequence
        (result[kwargs['packed_vae_token_indexes']].float()*torch.linspace(-1,1,result.shape[-1])).sum().backward()
        assert torch.isfinite(seed.hidden.grad).all() and seed.hidden.grad.norm()>0
        assert all(p.grad is None for p in model.language_model.parameters())
        runtime.config=replace(runtime.config,mode='LAYERWISE_SEED_REPLACE')
        static=decoder.forward_inference(**kwargs).packed_query_sequence
        with torch.no_grad():seed.hidden.add_(torch.linspace(-2,2,seed.hidden.shape[-1]))
        assert torch.equal(static,decoder.forward_inference(**kwargs).packed_query_sequence)
        runtime.config=replace(runtime.config,mode='LAYERWISE_MEMORY_REPLACE')
        dynamic=decoder.forward_inference(**kwargs).packed_query_sequence
        assert not torch.equal(result,dynamic) and not torch.equal(dynamic,static)
    finally:runtime.close()


@pytest.mark.parametrize('start',[0,1])
def test_replacement_body_and_suffix_bank_depend_on_current_gen(start):
    model,kwargs,runtime=prepared(batch=1,mode='LAYERWISE_MEMORY_REPLACE',slots=1,start=start)
    decoder=model.language_model.model;captured={}
    originals=[layer.forward_inference for layer in decoder.layers]
    def wrap(i):
        def observed(**kw):
            out,cache=originals[i](**kw)
            if kw['mode']=='und':captured[i]=cache.key_cache[i][kw['packed_query_indexes']].clone()
            return out,cache
        return observed
    for i,layer in enumerate(decoder.layers):layer.forward_inference=wrap(i)
    try:
        decoder.forward_inference(**kwargs)
        initial={i:t.clone() for i,t in captured.items()}
        changed=kwargs['packed_query_sequence'].clone()
        changed[kwargs['packed_vae_token_indexes']]+=torch.linspace(-3,3,changed.shape[-1])
        decoder.forward_inference(**{**kwargs,'packed_query_sequence':changed})
        assert torch.equal(initial[start],captured[start])  # Native fixed entrance.
        assert not torch.equal(initial[2],captured[2])
        assert not torch.equal(initial[3],captured[3])  # Feedback propagates through UND to suffix.
    finally:
        for layer,fn in zip(decoder.layers,originals):layer.forward_inference=fn
        runtime.close()
