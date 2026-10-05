from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import torch
import pytest
from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime, initial_memory, memory_layout
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM, NaiveCache
from oracles.legacy_memory_kernels import legacy_forward_inference
from oracles.parent_runner import legacy_kwargs


def fixture(device='cpu', batch=2):
    torch.manual_seed(123)
    cfg = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=4,
        layer_module='Qwen2MoTDecoderLayer', pad_token_id=0)
    llm = Qwen2ForCausalLM(cfg).eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    model = SimpleNamespace(language_model=llm)
    cache = NaiveCache(4)
    glen, plen = ([4, 7], [2, 3]) if batch == 2 else ([6], [3])
    for index in range(4):
        cache.key_cache[index] = torch.randn(sum(plen), 2, 8, device=device, dtype=torch.bfloat16)
        cache.value_cache[index] = torch.randn_like(cache.key_cache[index])
    positions, text, image, queries, cached = [], [], [], [], []
    qoff = merged = 0
    for g, p in zip(glen, plen):
        positions.extend(range(p, p+g))
        text.extend([qoff, qoff+g-1]); image.extend(range(qoff+1, qoff+g-1))
        cached.extend(range(merged, merged+p)); queries.extend(range(merged+p, merged+p+g))
        qoff += g; merged += p+g
    ids = lambda x: torch.tensor(x, device=device, dtype=torch.long)
    kwargs = dict(packed_query_sequence=torch.randn(sum(glen), 32, device=device, dtype=torch.bfloat16),
        query_lens=ids(glen).int(), packed_query_position_ids=ids(positions),
        packed_query_indexes=ids(queries), past_key_values=cache, key_values_lens=ids(plen).int(),
        packed_key_value_indexes=ids(cached), update_past_key_values=False, is_causal=False,
        mode='gen', packed_text_indexes=ids(text), packed_vae_token_indexes=ids(image))
    return model, kwargs


def run(model, kwargs, cfg, **options):
    runtime = InternalLoopRuntime(model, cfg, **options)
    try:
        result = model.language_model.model.forward_inference(**kwargs)
        return result.packed_query_sequence, list(runtime.diagnostics)
    finally:
        runtime.close()


@pytest.mark.parametrize('rounds', [0, 1, 2, 4])
def test_no_read_matches_native_and_prompt_cache_is_immutable(rounds):
    model, kwargs = fixture()
    expected = model.language_model.model.forward_inference(**kwargs).packed_query_sequence
    cache = kwargs['past_key_values']
    before = {i:(k, k.clone(), cache.value_cache[i], cache.value_cache[i].clone()) for i,k in cache.key_cache.items()}
    actual, _ = run(model, kwargs, LoopConfig(mode='MEMORY_NO_READ', extra_rounds=rounds, start_layer=1, end_layer=3))
    assert torch.equal(actual, expected), (actual-expected).abs().max().item()
    for i,(k,kcopy,v,vcopy) in before.items():
        assert cache.key_cache[i] is k and cache.value_cache[i] is v
        assert torch.equal(k,kcopy) and torch.equal(v,vcopy)


@pytest.mark.parametrize('mode,rounds,slots', [('BASE', 2, 8), ('MEMORY_LOOP',0,8), ('MEMORY_LOOP',2,0)])
def test_native_bypass_exact_and_runtime_adds_no_parameters(mode,rounds,slots):
    model, kwargs = fixture(); decoder = model.language_model.model
    original = decoder.forward_inference
    weights = {k:v.clone() for k,v in model.language_model.state_dict().items()}
    expected = original(**kwargs).packed_query_sequence
    actual, _ = run(model, kwargs, LoopConfig(mode=mode,extra_rounds=rounds,memory_slots=slots,end_layer=3))
    assert torch.equal(actual,expected) and decoder.forward_inference == original
    assert set(weights) == set(model.language_model.state_dict())
    assert all(torch.equal(v,weights[k]) for k,v in model.language_model.state_dict().items())




@pytest.mark.parametrize('batch', [1,2])
@pytest.mark.parametrize('rounds', [1,2,4])
def test_frozen_parent_hidden_and_velocity_parity(batch,rounds):
    model,kwargs=fixture(batch=batch); decoder=model.language_model.model
    oldkw,native=legacy_kwargs(kwargs)
    expected=legacy_forward_inference(decoder,**oldkw,memory_loop_repeat=rounds+1,
        memory_loop_start=1,memory_loop_end=3,block_gen_reads_memory=True).packed_query_sequence[native]
    actual,_=run(model,kwargs,LoopConfig(extra_rounds=rounds,start_layer=1,end_layer=3,memory_slots=3))
    assert torch.equal(actual,expected), (actual-expected).abs().max().item()
    torch.manual_seed(41)
    head=torch.nn.Linear(32,7).to(device=actual.device,dtype=actual.dtype).requires_grad_(False)
    assert torch.equal(head(actual[kwargs['packed_vae_token_indexes']]),head(expected[kwargs['packed_vae_token_indexes']]))


def test_body_resets_gen_and_boundary_and_only_recycles_memory():
    model,kwargs=fixture();decoder=model.language_model.model
    calls=[]; original=decoder.layers[1].forward_inference
    def observe(**kw):
        calls.append(kw['packed_query_sequence'].clone())
        return original(**kw)
    decoder.layers[1].forward_inference=observe
    cfg=LoopConfig(extra_rounds=2,start_layer=1,end_layer=3,memory_slots=3)
    run(model,kwargs,cfg)
    layout=memory_layout(kwargs,3)
    assert len(calls)==3
    assert all(torch.equal(state[layout.native_indexes],calls[0][layout.native_indexes]) for state in calls)
    assert not torch.equal(calls[0][layout.memory_indexes],calls[1][layout.memory_indexes])
    assert not torch.equal(calls[1][layout.memory_indexes],calls[2][layout.memory_indexes])


def test_prefix_suffix_execute_once_body_executes_r_plus_one():
    model,kwargs=fixture(); counts=[0]*4
    for i,layer in enumerate(model.language_model.model.layers):
        original=layer.forward_inference
        def wrapped(_i=i,_original=original,**kw):
            counts[_i]+=1;return _original(**kw)
        layer.forward_inference=wrapped
    run(model,kwargs,LoopConfig(extra_rounds=3,start_layer=1,end_layer=3))
    assert counts==[1,4,4,1]


def test_memory_is_ephemeral_and_cfg_branches_do_not_share_state():
    model,kwargs=fixture();cfg=LoopConfig(end_layer=3)
    runtime=InternalLoopRuntime(model,cfg)
    try:
        a=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        changed={**kwargs,'packed_query_sequence':kwargs['packed_query_sequence']+2}
        model.language_model.model.forward_inference(**changed)
        b=model.language_model.model.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(a,b)
        assert not hasattr(runtime,'banks')
    finally:runtime.close()


def test_packed_samples_do_not_cross_talk():
    model,kwargs=fixture();cfg=LoopConfig(end_layer=3)
    a,_=run(model,kwargs,cfg)
    h=kwargs['packed_query_sequence'].clone();h[4:]+=2
    b,_=run(model,{**kwargs,'packed_query_sequence':h},cfg)
    assert torch.equal(a[:4],b[:4]) and not torch.equal(a[4:],b[4:])


def test_positions_boundary_initialization_and_slot_noise():
    model,kwargs=fixture();layout=memory_layout(kwargs,8)
    assert torch.equal(layout.positions[layout.native_indexes],kwargs['packed_query_position_ids'])
    assert layout.positions[layout.memory_indexes].tolist()==[2]*8+[3]*8
    memory=initial_memory(kwargs['packed_query_sequence'],[4,7],8)
    assert not torch.equal(memory[:8],memory[8:])
    # Low-scale real embedding-like boundary states retain old 1e-4 slot noise.
    memory=initial_memory(kwargs['packed_query_sequence']*.01,[4,7],8)
    assert torch.unique(memory[:8],dim=0).shape[0]==8


def test_progress_bypass_and_exception_restore_attention():
    model,kwargs=fixture();decoder=model.language_model.model
    native=decoder.forward_inference(**kwargs).packed_query_sequence
    runtime=InternalLoopRuntime(model,LoopConfig(end_layer=3,progress_end=.5))
    original=decoder.layers[0].self_attn.forward_inference
    try:
        runtime.progress=.51
        assert torch.equal(decoder.forward_inference(**kwargs).packed_query_sequence,native)
        runtime.progress=0
        with pytest.raises(ValueError,match='immutable'):
            decoder.forward_inference(**{**kwargs,'update_past_key_values':True})
        assert decoder.layers[0].self_attn.forward_inference==original
        with pytest.raises(ValueError,match='already installed'):
            InternalLoopRuntime(model,LoopConfig(end_layer=3,progress_end=.5))
    finally:runtime.close()


def test_diagnostics_and_capture_do_not_change_generation():
    from qwen_latent_cot.bagel.memory_probe import ProbeCapture
    model,kwargs=fixture(batch=1);cfg=LoopConfig(start_layer=1,end_layer=3)
    expected,_=run(model,kwargs,cfg)
    capture=ProbeCapture([0]);actual,details=run(model,kwargs,cfg,diagnostics=True,probe_capture=capture)
    assert torch.equal(actual,expected) and set(capture.layers[0])=={1,2}
    assert len(details)==6
    assert all(not d['gen_reads_memory'] for d in details if d['phase']=='prefix' or d['round']==0)
    assert all(d['gen_reads_memory'] for d in details if d['round']==1 or d['phase']=='suffix')
    assert not torch.equal(capture.layers[0][1]['dynamic_k'],capture.layers[0][1]['seed_k'])



def test_strict_read_blocks_memory_including_boundary_relay():
    from types import MethodType
    from qwen_latent_cot.bagel.memory_attention import blocked_memory_attention
    model,kwargs=fixture();decoder=model.language_model.model
    expanded,_=legacy_kwargs(kwargs)
    memory=expanded.pop('packed_memory_token_indexes')
    layer=decoder.layers[0]
    cos,sin=decoder.rotary_emb(expanded['packed_query_sequence'],expanded.pop('packed_query_position_ids').unsqueeze(0))
    expanded['packed_query_position_embeddings']=(cos.squeeze(0),sin.squeeze(0))
    original=layer.self_attn.forward_inference
    def masked(this,**kw):
        return blocked_memory_attention(this,**kw,packed_memory_token_indexes=memory,block_gen_reads_memory=True)
    layer.self_attn.forward_inference=MethodType(masked,layer.self_attn)
    try:
        a,_=layer.forward_inference(**expanded)
        changed=expanded['packed_query_sequence'].clone();changed[memory]+=10
        b,_=layer.forward_inference(**{**expanded,'packed_query_sequence':changed})
        nonmemory=torch.ones(len(a),dtype=torch.bool,device=a.device);nonmemory[memory]=False
        assert torch.equal(a[nonmemory],b[nonmemory])
        assert not torch.equal(a[memory],b[memory])
    finally:layer.self_attn.forward_inference=original


def test_exception_inside_masked_layer_restores_native_attention():
    model,kwargs=fixture();decoder=model.language_model.model
    layer=decoder.layers[0];original=layer.self_attn.forward_inference
    def fail(*args,**kwargs):raise RuntimeError('injected MLP failure')
    layer.mlp_moe_gen.forward=fail
    runtime=InternalLoopRuntime(model,LoopConfig(end_layer=3))
    try:
        with pytest.raises(RuntimeError,match='injected'):
            decoder.forward_inference(**kwargs)
        assert layer.self_attn.forward_inference==original
    finally:runtime.close()

def test_slot_stats_detect_symmetry():
    from qwen_latent_cot.bagel.memory_stats import memory_slot_stats
    identical=memory_slot_stats(torch.ones(8,16));independent=memory_slot_stats(torch.eye(8,16))
    assert identical['effective_rank']<1.001 and identical['slot_std']==0
    assert identical['mean_pairwise_cosine']==1
    assert independent['effective_rank']>7.99 and independent['mean_pairwise_cosine']==0


def test_legacy_oracle_hash_is_frozen():
    root=Path(__file__).resolve().parents[1]
    ledger=json.loads((root/'docs/LEGACY_MEMORY_SOURCE.json').read_text())
    assert hashlib.sha256((root/'tests/oracles/legacy_memory_kernels.py').read_bytes()).hexdigest()==ledger['frozen_oracle_sha256']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='user-run CUDA contract')
def test_cuda_legacy_velocity_parity():
    model,kwargs=fixture('cuda',batch=1);decoder=model.language_model.model
    oldkw,native=legacy_kwargs(kwargs)
    expected=legacy_forward_inference(decoder,**oldkw,memory_loop_repeat=2,memory_loop_start=1,
        memory_loop_end=3,block_gen_reads_memory=True).packed_query_sequence[native]
    actual,_=run(model,kwargs,LoopConfig(start_layer=1,end_layer=3,memory_slots=3))
    assert torch.equal(actual,expected),(actual-expected).abs().max().item()
