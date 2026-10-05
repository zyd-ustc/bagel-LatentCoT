from dataclasses import replace
from types import SimpleNamespace
import torch
import pytest
from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime, MemorySeed, recurrent_layer, select_content_indexes
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2MoTDecoderLayer, NaiveCache


def fixture(device='cpu'):
    torch.manual_seed(123)
    cfg=Qwen2Config(hidden_size=32,intermediate_size=64,num_attention_heads=4,
        num_key_value_heads=2,num_hidden_layers=1,layer_module='Qwen2MoTDecoderLayer')
    layer=Qwen2MoTDecoderLayer(cfg,0).eval().requires_grad_(False).to(device=device,dtype=torch.bfloat16)
    cache=NaiveCache(1)
    cache.key_cache[0]=torch.randn(5,2,8,device=device,dtype=torch.bfloat16)
    cache.value_cache[0]=torch.randn(5,2,8,device=device,dtype=torch.bfloat16)
    h=torch.randn(9,32,device=device,dtype=torch.bfloat16)
    theta=torch.randn(9,4,device=device).repeat(1,2)
    kwargs=dict(packed_query_sequence=h,query_lens=torch.tensor([4,5],device=device,dtype=torch.int32),
        packed_query_position_embeddings=(theta.cos(),theta.sin()),
        packed_query_indexes=torch.tensor([2,3,4,5,9,10,11,12,13],device=device),
        past_key_values=cache,key_values_lens=torch.tensor([2,3],device=device,dtype=torch.int32),
        packed_key_value_indexes=torch.tensor([0,1,6,7,8],device=device),
        update_past_key_values=False,is_causal=False,mode='gen',
        packed_text_indexes=torch.tensor([0,3,4,8],device=device),
        packed_vae_token_indexes=torch.tensor([1,2,5,6,7],device=device))
    angles=torch.randn(3,4,device=device).repeat(1,2)
    seed=MemorySeed(torch.randn(3,32,device=device,dtype=torch.bfloat16),
        (angles.cos(),angles.sin()),[1,2],torch.tensor([1,3,4],device=device))
    return layer, kwargs, seed


def test_selection_excludes_special_and_does_not_duplicate():
    ids, lens=select_content_indexes(torch.tensor([0,5,6,7,1,0,8,1,0,1]),[5,3,2],{0,1},16)
    assert ids.tolist()==[1,2,3,6] and lens==[3,1,0]
    ids,lens=select_content_indexes(torch.arange(8),[8],set(),3)
    assert ids.tolist()==[0,3,7]


@pytest.mark.parametrize('mode',['BASE','MEMORY_DYNAMIC'])
def test_n1_calls_native_exactly(mode):
    layer, kwargs,_=fixture(); original=layer.forward_inference
    expected,_=original(**kwargs)
    model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    runtime=InternalLoopRuntime(model,LoopConfig(mode=mode,evaluations=1,start_layer=0,end_layer=1))
    got,_=layer.forward_inference(**kwargs)
    assert torch.equal(got,expected)
    runtime.close(); assert layer.forward_inference==original


@pytest.mark.parametrize('depth',[2,3,4])
def test_no_read_matches_gen_and_prompt_cache_is_immutable(depth):
    layer,kwargs,seed=fixture()
    cache=kwargs['past_key_values']; pk=cache.key_cache[0]; pv=cache.value_cache[0]
    before_k,before_v=pk.clone(),pv.clone()
    original=layer.forward_inference
    expected=kwargs['packed_query_sequence']
    for _ in range(depth):
        full,_=original(**{**kwargs,'packed_query_sequence':expected})
        expected=expected+(full-expected)/depth
    got,_=recurrent_layer(layer,kwargs,seed,LoopConfig(mode='MEMORY_NO_READ',evaluations=depth,start_layer=0,end_layer=1))
    assert torch.equal(got,expected), (got-expected).abs().max().item()
    assert cache.key_cache[0] is pk and cache.value_cache[0] is pv
    assert torch.equal(pk,before_k) and torch.equal(pv,before_v)


def test_dynamic_has_delayed_read_and_uses_updated_memory():
    layer,kwargs,seed=fixture()
    cfg=LoopConfig(start_layer=0,end_layer=1)
    dynamic,_=recurrent_layer(layer,kwargs,seed,cfg)
    static,_=recurrent_layer(layer,kwargs,seed,replace(cfg,mode='MEMORY_STATIC'))
    noread,_=recurrent_layer(layer,kwargs,seed,replace(cfg,mode='MEMORY_NO_READ'))
    assert not torch.equal(dynamic,static) and not torch.equal(dynamic,noread)
    seed_before=seed.hidden.clone()
    r1,_=recurrent_layer(layer,kwargs,seed,replace(cfg,evaluations=1))
    native,_=layer.forward_inference(**kwargs)
    h = kwargs['packed_query_sequence']
    assert torch.equal(r1,h+(native-h))  # Same native F; residual arithmetic is BF16.
    assert torch.equal(seed.hidden,seed_before)


def test_packed_samples_do_not_cross_talk():
    layer,kwargs,seed=fixture();cfg=LoopConfig(start_layer=0,end_layer=1)
    a,_=recurrent_layer(layer,kwargs,seed,cfg)
    changed=seed.hidden.clone();changed[1:]+=10
    b,_=recurrent_layer(layer,kwargs,replace(seed,hidden=changed),cfg)
    assert torch.equal(a[:4],b[:4])
    assert not torch.equal(a[4:],b[4:])


def test_writer_has_no_unused_final_q_or_mlp():
    layer,kwargs,seed=fixture();counts={'q':0,'mlp':0}
    handles=[layer.self_attn.q_proj.register_forward_hook(lambda m,i,o: counts.__setitem__('q',counts['q']+int(len(i[0])==3))),
             layer.mlp.register_forward_hook(lambda m,i,o: counts.__setitem__('mlp',counts['mlp']+int(len(i[0])==3)))]
    recurrent_layer(layer,kwargs,seed,LoopConfig(evaluations=3,start_layer=0,end_layer=1))
    for h in handles:h.remove()
    assert counts=={'q':2,'mlp':2}


def test_runtime_does_not_add_parameters_and_captures_native_positions():
    layer,kwargs,_=fixture();keys=set(layer.state_dict())
    model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    runtime=InternalLoopRuntime(model,LoopConfig(start_layer=0,end_layer=1))
    cache=NaiveCache(1);tokens=torch.tensor([0,4,5,1]);runtime.begin_prefill(cache,tokens,[4],{0,1})
    und={**kwargs,'mode':'und','packed_query_sequence':kwargs['packed_query_sequence'][:4],
        'query_lens':torch.tensor([4],dtype=torch.int32),'past_key_values':cache,
        'key_values_lens':torch.tensor([0],dtype=torch.int32),
        'packed_query_indexes':torch.arange(4),'packed_key_value_indexes':torch.tensor([],dtype=torch.long),
        'packed_query_position_embeddings':tuple(r[:4] for r in kwargs['packed_query_position_embeddings']),
        'update_past_key_values':True,'is_causal':True}
    layer.forward_inference(**und);runtime.end_prefill()
    stored=runtime.banks[id(cache)][1][0]
    assert torch.equal(stored.hidden,und['packed_query_sequence'][[1,2]])
    assert torch.equal(stored.rope[0],und['packed_query_position_embeddings'][0][[1,2]])
    assert keys==set(layer.state_dict());runtime.close()


def test_progress_schedule_repeats_only_selected_layers():
    from torch import nn
    calls=[]
    class Layer(nn.Module):
        def __init__(self,index):super().__init__();self.index=index
        def forward_inference(self,**kw):
            calls.append(self.index)
            return kw['packed_query_sequence']+1,kw['past_key_values']
    layers=[Layer(i) for i in range(3)]
    model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    runtime=InternalLoopRuntime(model,LoopConfig(mode='GEN_LAYERWISE',start_layer=1,end_layer=2,evaluations=3))
    kwargs={'mode':'gen','packed_query_sequence':torch.zeros(1),'past_key_values':None,
            'update_past_key_values':False,'is_causal':False}
    for layer in layers:layer.forward_inference(**kwargs)
    assert calls==[0,1,1,1,2]
    calls.clear();runtime.progress=.51
    for layer in layers:layer.forward_inference(**kwargs)
    assert calls==[0,1,2]
    runtime.close()


def test_nonempty_cfg_branch_cannot_read_another_branch_seed():
    layer,kwargs,seed=fixture()
    model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    runtime=InternalLoopRuntime(model,LoopConfig(start_layer=0,end_layer=1))
    foreign=NaiveCache(1)
    runtime.banks[id(foreign)]=(foreign,{0:seed})
    with pytest.raises(RuntimeError,match='CFG branch'):layer.forward_inference(**kwargs)
    runtime.close()


def test_optional_diagnostics_do_not_change_generation_math():
    layer,kwargs,seed=fixture();cfg=LoopConfig(start_layer=0,end_layer=1)
    a,_=recurrent_layer(layer,kwargs,seed,cfg);details=[]
    b,_=recurrent_layer(layer,kwargs,seed,cfg,details)
    assert torch.equal(a,b)
    assert len(details)==4
    assert all(d['read_mass']==0 for d in details if d['iteration']==0)
    assert all(d['read_mass']>0 for d in details if d['iteration']==1)
    assert all(d['update_ratio']>0 for d in details if d['iteration']==0)


def test_slot_stats_detect_symmetry():
    from qwen_latent_cot.bagel.memory_stats import memory_slot_stats
    identical=memory_slot_stats(torch.ones(8,16))
    independent=memory_slot_stats(torch.eye(8,16))
    assert identical['effective_rank']<1.001 and identical['slot_std']==0
    assert identical['mean_pairwise_cosine']==1
    assert independent['effective_rank']>7.99
    assert independent['mean_pairwise_cosine']==0


def test_mixed_empty_memory_slots_keep_packed_isolation():
    layer,kwargs,seed=fixture();seed=replace(seed,hidden=seed.hidden[1:],rope=tuple(r[1:] for r in seed.rope),lengths=[0,2])
    dynamic,_=recurrent_layer(layer,kwargs,seed,LoopConfig(start_layer=0,end_layer=1))
    no_read,_=recurrent_layer(layer,kwargs,seed,LoopConfig(mode='MEMORY_NO_READ',start_layer=0,end_layer=1))
    assert torch.equal(dynamic[:4],no_read[:4])
    assert not torch.equal(dynamic[4:],no_read[4:])


def test_attention_bottom_right_mask_against_float64_reference():
    from qwen_latent_cot.bagel.attention import flash_attn_varlen_func
    torch.manual_seed(42)
    q=torch.randn(2,4,8,dtype=torch.float64);k=torch.randn(5,2,8,dtype=torch.float64);v=torch.randn_like(k)
    out=flash_attn_varlen_func(q,k,v,torch.tensor([0,2],dtype=torch.int32),torch.tensor([0,5],dtype=torch.int32),2,5,causal=True)
    kk=k.repeat_interleave(2,dim=1).transpose(0,1);vv=v.repeat_interleave(2,dim=1).transpose(0,1)
    logits=q.transpose(0,1)@kk.transpose(-1,-2)/(8**.5)
    allowed=torch.arange(5)[None,:]<=torch.arange(2)[:,None]+3
    expected=(logits.masked_fill(~allowed,float('-inf')).softmax(-1)@vv).transpose(0,1)
    assert torch.allclose(out,expected,atol=2e-7,rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA/FlashAttention contract')
@pytest.mark.parametrize('mlen',[(1,2),(0,2)])
def test_cuda_flash_attention_parity_and_empty_query_segments(mlen):
    layer,kwargs,seed=fixture('cuda')
    if mlen[0]==0:seed=replace(seed,hidden=seed.hidden[1:],rope=tuple(r[1:] for r in seed.rope),lengths=list(mlen))
    cfg=LoopConfig(mode='MEMORY_NO_READ',start_layer=0,end_layer=1)
    h=kwargs['packed_query_sequence']
    for _ in range(2):
        full,_=layer.forward_inference(**{**kwargs,'packed_query_sequence':h})
        h=h+(full-h)/2
    actual,_=recurrent_layer(layer,kwargs,seed,cfg)
    assert torch.equal(h,actual)
    details=[]
    dynamic,_=recurrent_layer(layer,kwargs,seed,replace(cfg,mode='MEMORY_DYNAMIC'),details)
    assert torch.isfinite(dynamic).all()
    if not mlen[0]:assert torch.equal(actual[:4],dynamic[:4])
