from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from helpers import prepared
from qwen_latent_cot.bagel.native_und import project_und
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache


def setup(rounds=2,start=0,device='cpu'):
    return prepared(mode='LAYERWISE_UND_STATE_REPLACE',rounds=rounds,
                    start=start,special_tokens=True,device=device)


def test_native_seeds_at_every_layer_and_same_layer_projection():
    model,kwargs,runtime=setup();cache=kwargs['past_key_values'];seed=runtime.layerwise.seeds[cache]
    try:
        assert set(seed.layer_hidden)==set(range(4))
        cos,sin=model.language_model.model.rotary_emb(seed.hidden,seed.positions.unsqueeze(0))
        rope=cos.squeeze(0),sin.squeeze(0)
        for index,h in seed.layer_hidden.items():
            _,k,v=project_und(model.language_model.model.layers[index],h,rope)
            assert torch.equal(k,cache.key_cache[index]) and torch.equal(v,cache.value_cache[index])
    finally:runtime.close()

def test_carries_each_layer_output_hidden_and_projects_that_updated_hidden():
    rounds=2
    model,kwargs,runtime=setup(rounds);decoder=model.language_model.model
    seed=runtime.layerwise.seeds[kwargs['past_key_values']];events=[]
    def observe(**event):
        events.append({k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in event.items()})
    runtime.kv_observer=observe
    cos,sin=decoder.rotary_emb(seed.hidden,seed.positions.unsqueeze(0));rope=cos.squeeze(0),sin.squeeze(0)
    try:
        output=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.isfinite(output).all()
        for index in range(3):
            writes=[e for e in events if e['event']=='writer_update' and e['phase']=='body' and e['layer']==index]
            assert len(writes)==rounds and torch.equal(writes[0]['hidden_before'],seed.layer_hidden[index])
            for previous,current in zip(writes,writes[1:]):
                assert torch.equal(previous['hidden_after'],current['hidden_before'])
            for event in writes:
                assert torch.equal(event['hidden_after'][seed.special_mask],seed.layer_hidden[index][seed.special_mask])
                _,k,v=project_und(decoder.layers[index],event['hidden_after'],rope)
                content=~seed.special_mask
                assert torch.equal(k[content],event['current'].keys[content])
                assert torch.equal(v[content],event['current'].values[content])
            final=next(e for e in events if e['event']=='final_read' and e['layer']==index)
            assert torch.equal(final['current'].keys,writes[-1]['current'].keys)
        first=[e for e in events if e['event']=='writer_update' and e['layer']==0]
        assert all(not torch.equal(e['current'].keys[~seed.special_mask],e['reference'].keys[~seed.special_mask]) for e in first)
        suffix_writes=[e for e in events if e['event']=='writer_update' and e['phase']=='suffix']
        assert len(suffix_writes)==1
        suffix=next(e for e in events if e['event']=='final_read' and e['phase']=='suffix')
        assert torch.equal(suffix['current'].keys,suffix_writes[0]['current'].keys)
        assert torch.equal(suffix['current'].values,suffix_writes[0]['current'].values)
        assert not torch.equal(suffix['current'].keys[~seed.special_mask],suffix['reference'].keys[~seed.special_mask])
    finally:runtime.close()

@pytest.mark.parametrize('start',[0,1])
def test_gen_entrance_resets_full_capacity_and_suffix_runs_once(start):
    model,kwargs,runtime=setup(start=start);decoder=model.language_model.model;originals=[];calls=[]
    for index,layer in enumerate(decoder.layers):
        original=layer.forward_inference;originals.append(original)
        def observed(*,i=index,fn=original,**kw):
            calls.append((i,kw['mode'],kw['packed_query_sequence'].clone(),kw['key_values_lens'].tolist()))
            return fn(**kw)
        layer.forward_inference=observed
    try:
        decoder.forward_inference(**kwargs)
        entry=[c[2] for c in calls if c[0]==start and c[1]=='gen']
        assert len(entry)==3 and all(torch.equal(entry[0],h) for h in entry[1:])
        for index in range(4):
            assert len([c for c in calls if c[0]==index and c[1]=='gen'])==(3 if start<=index<3 else 1)
            assert len([c for c in calls if c[0]==index and c[1]=='und'])==(2 if start<=index<3 else 1 if index>=3 else 0)
        assert all(c[3]==[2,3] for c in calls if c[1]=='gen')
        # Writer has P + current GEN + live UND query; no old M KV addition.
        assert all(c[3]==[6,10] for c in calls if c[1]=='und' and c[0]<3)
    finally:
        for layer,original in zip(decoder.layers,originals):layer.forward_inference=original
        runtime.close()

def test_call_and_sample_isolation_native_bypass_seed_cache_and_weights_immutable():
    model,kwargs,runtime=setup();decoder=model.language_model.model;cache=kwargs['past_key_values']
    seed=runtime.layerwise.seeds[cache];seeds={i:h.clone() for i,h in seed.layer_hidden.items()}
    keys={i:k.clone() for i,k in cache.key_cache.items()};values={i:v.clone() for i,v in cache.value_cache.items()}
    weights={name:w.clone() for name,w in model.language_model.state_dict().items()}
    try:
        native=runtime.original(**kwargs).packed_query_sequence
        a=decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(a,decoder.forward_inference(**kwargs).packed_query_sequence)
        changed=kwargs['packed_query_sequence'].clone();changed[4:]+=torch.linspace(-3,3,changed.shape[-1])
        b=decoder.forward_inference(**{**kwargs,'packed_query_sequence':changed}).packed_query_sequence
        assert torch.equal(a[:4],b[:4]) and torch.equal(a[:4],native[:4]) and not torch.equal(a[4:],b[4:])
        null={**kwargs,'past_key_values':NaiveCache(4),'key_values_lens':torch.zeros(2,dtype=torch.int32),
              'packed_key_value_indexes':torch.empty(0,dtype=torch.long),'packed_query_indexes':torch.arange(11)}
        assert torch.equal(decoder.forward_inference(**null).packed_query_sequence,runtime.original(**null).packed_query_sequence)
        runtime.config=replace(runtime.config,extra_rounds=0)
        assert torch.equal(decoder.forward_inference(**kwargs).packed_query_sequence,native)
        assert all(torch.equal(seed.layer_hidden[i],h) for i,h in seeds.items())
        assert all(torch.equal(cache.key_cache[i],k) and torch.equal(cache.value_cache[i],values[i]) for i,k in keys.items())
        assert all(torch.equal(model.language_model.state_dict()[name],w) for name,w in weights.items())
    finally:runtime.close()

def test_real_weight_validator_orchestration_with_cpu_decoder():
    from contextlib import nullcontext
    from types import MethodType
    import importlib.util
    model,kwargs,runtime=setup();cache=kwargs['past_key_values'];seed=runtime.layerwise.seeds[cache]
    def flow(this,*,x_t,cfg_text_scale=1.,**unused):
        return runtime.decoder.forward_inference(**{**kwargs,'packed_query_sequence':x_t}).packed_query_sequence*cfg_text_scale
    model._forward_flow=MethodType(flow,model)
    class Generator:
        prompt_lengths=(2,3)
        def autocast(self):return nullcontext()
        def prepare(self,*args):
            runtime.layerwise.seeds[cache]=seed
            return dict(packed_init_noises=kwargs['packed_query_sequence'].clone(),past_key_values=cache),['a','b']
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/compare_windows.py'
    spec=importlib.util.spec_from_file_location('state_e0',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        result=module.validate(SimpleNamespace(model=model),Generator(),runtime,(2,))
        assert result['passed'] and result['conditional_effect_observed']
        assert set(result['native_hidden_seed_kv_parity'])=={'0','1','2','3'}
    finally:runtime.close()

@pytest.mark.parametrize('start,end',[(0,8),(4,12),(8,16),(12,20),(16,24),(20,28)])
def test_exact_frozen_target_hidden_and_velocity_parity_for_six_windows(start,end):
    from oracles.und_state import run_und_state
    model,kwargs,runtime=prepared(rounds=2,start=start,end=end,depth=28,special_tokens=True)
    try:
        # Frozen selected runner from before cleanup; no second production path.
        old_runtime=SimpleNamespace(**vars(runtime),probe_capture=None)
        expected=run_und_state(runtime.layerwise,kwargs,old_runtime).packed_query_sequence
        actual=runtime.decoder.forward_inference(**kwargs).packed_query_sequence
        assert torch.equal(actual,expected)
        torch.manual_seed(41)
        head=torch.nn.Linear(32,7).to(dtype=actual.dtype).requires_grad_(False)
        image=kwargs['packed_vae_token_indexes']
        assert torch.equal(head(actual[image]),head(expected[image]))
    finally:runtime.close()
