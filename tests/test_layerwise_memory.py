from dataclasses import replace
import pytest
import torch
from test_internal_loop import fixture
from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime
from qwen_latent_cot.bagel.layerwise_memory import LayerKV, native_context, select_content_indexes
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache


def prepared(batch=2, empty_first=False, rounds=1, mode='LAYERWISE_MEMORY_KV', device='cpu'):
    model, kwargs = fixture(batch=batch,device=device)
    decoder = model.language_model.model
    runtime = InternalLoopRuntime(model, LoopConfig(mode=mode, extra_rounds=rounds,
        start_layer=1, end_layer=3, memory_slots=3), diagnostics=True)
    lengths = kwargs['key_values_lens'].tolist()
    tokens = torch.tensor([5,6,7,8,9] if batch==2 else [5,6,7],device=device)
    if empty_first: tokens[:lengths[0]]=0
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
def test_sample_isolation_local_round_state_and_null_cfg(empty_first):
    model,kwargs,runtime=prepared(empty_first=empty_first)
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
