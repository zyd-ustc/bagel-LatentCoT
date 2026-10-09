from dataclasses import replace
from contextlib import nullcontext
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

@pytest.mark.parametrize("memory_update",["legacy_layerwise","full_depth"])
def test_real_weight_validator_orchestration_with_cpu_decoder(memory_update):
    from types import MethodType
    import importlib.util
    model,kwargs,runtime=setup();cache=kwargs['past_key_values'];seed=runtime.layerwise.seeds[cache]
    runtime.config=replace(runtime.config,memory_update=memory_update)
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
        for first,last in ((0.,1.),(0.,4/48),(29/48,1.)):
            runtime.config=replace(runtime.config,progress_start=first,progress_end=last)
            runtime.progress=.25
            result=module.validate(SimpleNamespace(model=model),Generator(),runtime,(2,))
            assert result['passed'] and result['conditional_effect_observed']
            assert set(result['native_hidden_seed_kv_parity'])=={'0','1','2','3'}
            assert runtime.progress==.25
            if first>0 or last<1:
                assert all(v['outside_window']['equal'] for v in result['timesteps'].values())
    finally:runtime.close()

@pytest.mark.parametrize('rounds',[1,2,3,4])
def test_exact_frozen_target_hidden_and_velocity_parity_for_selected_window(rounds):
    start,end=0,8
    from oracles.und_state import run_und_state
    model,kwargs,runtime=prepared(rounds=rounds,start=start,end=end,depth=28,special_tokens=True)
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


@pytest.mark.parametrize('name',[f'EARLY_{n}_R{r}' for n in (10,20) for r in (1,2,3,4)])
def test_time_window_boundaries_match_full_loop_or_native_numerically(name):
    import json
    from pathlib import Path
    from qwen_latent_cot.evaluation.windows import validate_config,arm_configs,window_metadata
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/window_comparison.json').read_text())
    validate_config(config,28)
    cfg=arm_configs(config)[name];metadata=window_metadata(config)[name]
    model,kwargs,runtime=setup(rounds=cfg['extra_rounds'])
    try:
        native=runtime.original(**kwargs).packed_query_sequence
        full=runtime.decoder.forward_inference(**kwargs).packed_query_sequence
        runtime.config=replace(runtime.config,progress_start=cfg['progress_start'],progress_end=cfg['progress_end'])
        active=[step for step in range(49) if cfg['progress_start']<=step/48<=cfg['progress_end']]
        assert active==metadata['loop_step_indexes'] and len(active)==metadata['loop_calls']
        first,last=active[0],active[-1]
        probes={0,first,last,48}
        if first>0:probes.add(first-1)
        if last<48:probes.add(last+1)
        for step in sorted(probes):
            runtime.progress=step/48;runtime.step_index=step
            actual=runtime.decoder.forward_inference(**kwargs).packed_query_sequence
            assert torch.equal(actual,full if step in active else native)
            torch.manual_seed(41)
            head=torch.nn.Linear(32,7).to(dtype=actual.dtype).requires_grad_(False)
            indexes=kwargs['packed_vae_token_indexes']
            assert torch.equal(head(actual[indexes]),head((full if step in active else native)[indexes]))
    finally:runtime.close()


def test_native_feedback_full_context_and_cfg_isolation():
    """Exercise native API ordering, full feedback, CFG contexts and paired noise."""
    from types import SimpleNamespace
    from PIL import Image
    from qwen_latent_cot.bagel.feedback import NativeFeedback, parse_feedback
    from qwen_latent_cot.bagel.inferencer import T2IGenerator
    import hashlib
    class Model(torch.nn.Module):
        latent_downsample=16
        patch_latent_dim=4
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1))
        def prepare_prompts(self,lens,ropes,texts,*args):
            return {'text':texts[0]},[lens[0]+len(texts[0])+2],[ropes[0]+len(texts[0])+2]
        def forward_cache_update_text(self,cache,text):
            cache.trace=getattr(cache,'trace',[])+[('text',text)];return cache
        def prepare_vae_images(self,lens,ropes,*args):return {},[lens[0]+7],[ropes[0]+1]
        def forward_cache_update_vae(self,vae,cache):cache.trace=[('vae','all')];return cache
        def prepare_vit_images(self,lens,ropes,*args):return {},[lens[0]+11],[ropes[0]+1]
        def forward_cache_update_vit(self,cache):cache.trace=getattr(cache,'trace',[])+[('vit','all')];return cache
        def prepare_vae_latent(self,lens,ropes,*args):return {'packed_init_noises':torch.empty(1)}
        def prepare_vae_latent_cfg(self,lens,ropes,*args):return {'cfg_key_values_lens':torch.tensor(lens)}
    model=Model();bundle=SimpleNamespace(model=model,vae=None,tokenizer=SimpleNamespace(encode=lambda s:list(s)),token_ids={})
    engine=NativeFeedback.__new__(NativeFeedback);engine.model=model;engine.bundle=bundle
    engine.decoder=T2IGenerator(bundle);engine.device=torch.device('cpu')
    engine.vae_transform=SimpleNamespace(resize_transform=lambda image:image);engine.vit_transform=None
    instruction='full feedback: '+('preserve every correct object; '*40)
    flow,noise_hash,meta=engine.prepare_edit(Image.new('RGB',(512,512)),instruction,19)
    assert flow['past_key_values'].trace==[('vae','all'),('vit','all'),('text',instruction)]
    assert flow['cfg_text_past_key_values'].trace==[('vae','all'),('vit','all')]
    assert flow['cfg_img_past_key_values'].trace==[('text',instruction)]
    assert meta['instruction_token_ids']==list(instruction)
    assert meta['conditional_lengths']==[18+len(instruction)+2]
    noise=torch.randn(1024,4,generator=torch.Generator().manual_seed(19),dtype=torch.float32)
    assert noise_hash==hashlib.sha256(noise.numpy().tobytes()).hexdigest()
    assert torch.equal(flow['packed_init_noises'],noise)
    assert parse_feedback('{"observed":"two cubes","discrepancies":[],"preserve":["both cubes"],"uncertain":[],"edit":"preserve"}')['edit']=='preserve'
    with pytest.raises(ValueError):parse_feedback('{"edit":"invented fallback"}')


def test_full_und_observation_matches_native_prefill_and_static_read_control():
    from copy import deepcopy
    from PIL import Image
    from types import MethodType
    from helpers import fixture
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel
    from qwen_latent_cot.bagel.observation_memory import ObservationConditions,memory_context
    model,_=fixture(batch=1,depth=4)
    class Wrapper(torch.nn.Module):
        prepare_prompts=Bagel.prepare_prompts
        forward_cache_update_text=Bagel.forward_cache_update_text
        prepare_start_tokens=Bagel.prepare_start_tokens
        generate_text=Bagel.generate_text
        use_moe=True
        def __init__(self,llm):
            super().__init__();self.language_model=llm;self.config=SimpleNamespace(llm_config=llm.config)
        def prepare_vae_latent(self,*args):return {'packed_init_noises':torch.zeros(4,2)}
        def prepare_vae_latent_cfg(self,lens,ropes,*args):return {'cfg_key_values_lens':torch.tensor(lens)}
    wrapper=Wrapper(model.language_model)
    wrapper.language_model.model.enable_taylorseer=False
    tokenizer=SimpleNamespace(encode=lambda text:[5,6,7],decode=lambda ids:' '.join(map(str,ids)))
    bundle=SimpleNamespace(model=wrapper,vae=None,tokenizer=tokenizer,token_ids={'bos_token_id':0,'eos_token_id':1})
    engine=ObservationConditions(bundle)
    torch.manual_seed(14)
    prefix=NaiveCache(4)
    hidden=torch.randn(4,32,dtype=torch.bfloat16)
    wrapper.language_model.forward_inference(packed_query_sequence=hidden,query_lens=torch.tensor([4],dtype=torch.int32),
        packed_query_position_ids=torch.zeros(4,dtype=torch.long),packed_query_indexes=torch.arange(4),
        past_key_values=prefix,key_values_lens=torch.zeros(1,dtype=torch.int32),packed_key_value_indexes=torch.empty(0,dtype=torch.long),
        update_past_key_values=True,is_causal=False,mode='und')
    def image(this,image,context,vae=True):
        return dict(past_key_values=deepcopy(prefix),kv_lens=[4],ropes=[2])
    engine.image=MethodType(image,engine)
    image_input=Image.new('RGB',(512,512))
    observed,context,visual,meta=engine.prepare_conditions(image_input,'prompt',True)
    static,static_context,_,static_meta=engine.prepare_conditions(image_input,'prompt',False)
    reference=engine.text('prompt',image(engine,image_input,engine.context()))
    assert meta['text_ids']==static_meta['text_ids']==[0,5,6,7,1]
    assert meta['text_positions']==static_meta['text_positions']==list(range(2,7))
    assert meta['memory_length']==5 and meta['conditional_lengths']==[9]
    for i in range(4):
        assert torch.equal(context['past_key_values'].key_cache[i],reference['past_key_values'].key_cache[i])
        assert torch.equal(context['past_key_values'].key_cache[i][:4],static_context['past_key_values'].key_cache[i][:4])
    assert any(not torch.equal(context['past_key_values'].key_cache[i][4:],static_context['past_key_values'].key_cache[i][4:]) for i in range(1,4))
    static_memory=memory_context(static_context,4)
    # Alter image evidence: observed Memory changes, static Memory does not.
    for i in range(4):prefix.value_cache[i]+=2
    _,context2,_,_=engine.prepare_conditions(image_input,'prompt',True)
    _,static2,_,_=engine.prepare_conditions(image_input,'prompt',False)
    assert any(not torch.equal(context['past_key_values'].key_cache[i][4:],context2['past_key_values'].key_cache[i][4:]) for i in range(1,4))
    assert all(torch.equal(static_memory['past_key_values'].key_cache[i],static2['past_key_values'].key_cache[i][4:]) for i in range(4))
    before={i:(k.clone(),context['past_key_values'].value_cache[i].clone()) for i,k in context['past_key_values'].key_cache.items()}
    engine.answer_context(context,'How many objects?',2)
    assert all(torch.equal(context['past_key_values'].key_cache[i],k) and torch.equal(context['past_key_values'].value_cache[i],v) for i,(k,v) in before.items())


@pytest.mark.parametrize('step_index,duration',[(9,1),(19,1),(9,5),(9,10),(9,20)])
def test_observation_fixed_cache_window_at_current_noise_time_and_restores_hook(tmp_path,step_index,duration):
    from PIL import Image
    from qwen_latent_cot.bagel.observation_memory import ObservationMemoryGenerator,tensor_hash
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.calls=[]
            self.language_model=SimpleNamespace(model=SimpleNamespace())
        def _forward_flow(self,x_t,timestep,past_key_values,**kwargs):
            self.calls.append((x_t.clone(),timestep.clone(),past_key_values))
            return torch.full_like(x_t,.2 if past_key_values=='native' else .3)
        def generate_image(self,packed_init_noises,num_timesteps,**kwargs):
            x=packed_init_noises;ts=torch.linspace(1,0,num_timesteps);ts=3*ts/(1+2*ts)
            for i,t in enumerate(ts[:-1]):
                v=self._forward_flow(x_t=x,timestep=torch.full((len(x),),t),past_key_values='native')
                x=x-v*(ts[i]-ts[i+1])
            self.final_latent=x;self.schedule=ts
            return [x]
    model=Model();original=model._forward_flow
    gen=ObservationMemoryGenerator.__new__(ObservationMemoryGenerator)
    gen.model=model;gen.device=torch.device('cpu');gen.runtime=None;gen.observation_step=step_index;gen.observe=True
    gen.end_step=step_index+duration
    gen.questions=['How many objects?'];gen.probe_max_tokens=2;gen.trace_dir=tmp_path;gen.edit_sampling={'cfg_text_scale':3.,'cfg_img_scale':1.5,'cfg_interval':[.4,1.]}
    writer_calls=[]
    def conditions(*args):
        writer_calls.append(args)
        cache=NaiveCache(1);cache.key_cache[0]=torch.ones(3,1,2);cache.value_cache[0]=torch.ones(3,1,2)
        return {'past_key_values':'updated'},dict(past_key_values=cache),{},dict(visual_prefix_length=1)
    gen.engine=SimpleNamespace(prepare_conditions=conditions,probe=lambda *args:{'fake_probe':'never reaches GEN'})
    gen.prepare=lambda *args:({'packed_init_noises':torch.ones(2,3)},['common_noise'])
    gen.decode=lambda *args:Image.new('RGB',(512,512),'gray')
    images,hashes=gen.generate(['prompt'],[(512,512)],[0],timestep_shift=3.)
    assert len(model.calls)==49+duration and hashes==['common_noise'] and len(gen.events)==duration
    assert len(writer_calls)==1 and [e['updates'] for e in gen.events]==[1]+[0]*(duration-1)
    assert sum(bool(e['probes']) for e in gen.events)==1 and gen.events[-1]['held_memory_unchanged']
    updated=[c for c in model.calls if c[2]=='updated']
    assert len(updated)==duration
    assert torch.equal(torch.tensor([float(c[1][0]) for c in updated]),model.schedule[step_index:gen.end_step])
    expected=.8-.1*(model.schedule[step_index]-model.schedule[gen.end_step])
    assert torch.allclose(model.final_latent,torch.full_like(model.final_latent,expected),atol=1e-6)
    a,b=model.calls[step_index:step_index+2]
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    assert a[2]=='native' and b[2]=='updated'
    assert gen.events[0]['step_index']==step_index and gen.events[0]['updates']==1
    assert gen.events[0]['x_t_sha256']==tensor_hash(a[0])
    assert model._forward_flow==original and (tmp_path/'state.pt').exists()
    from copy import deepcopy
    from qwen_latent_cot.evaluation.io import validate_observation_record
    record=dict(arm='test_observed',valid_file=True,observation_step=step_index,
        conditioning_end_step=gen.end_step,observation_events=deepcopy(gen.events))
    validate_observation_record(record)
    missing=deepcopy(record);missing['observation_events'].pop()
    with pytest.raises(ValueError,match='coverage'):validate_observation_record(missing)
    mutable=deepcopy(record);mutable['observation_events'][-1]['held_memory_unchanged']=False
    with pytest.raises(ValueError,match='immutability'):validate_observation_record(mutable)
    if duration>1:
        repeated=deepcopy(record);repeated['observation_events'][1]['updates']=1
        with pytest.raises(ValueError,match='count'):validate_observation_record(repeated)
    # Failure must also restore the original denoiser method.
    gen.decode=lambda *args:(_ for _ in ()).throw(RuntimeError('diagnostic failure'))
    with pytest.raises(RuntimeError,match='diagnostic failure'):gen.generate(['prompt'],[(512,512)],[0])
    assert model._forward_flow==original


def test_duration_arm_coverage_legacy_control_and_comparisons():
    import json
    from pathlib import Path
    from qwen_latent_cot.evaluation.windows import validate_config,arm_configs,window_metadata,comparison_pairs,reference_arm
    root=Path(__file__).resolve().parents[1]
    c=validate_config(json.loads((root/'configs/observation_duration_comparison.json').read_text()),28)
    arms=arm_configs(c);windows=window_metadata(c)
    assert len(arms)==10 and c['expected_prompts']==32
    old=arm_configs(json.loads((root/'configs/loop_layer_npu_legacy_pilot.json').read_text()))['EARLY_20_R2']
    assert arms['LEGACY_EARLY_20_R2']==old
    assert windows['LEGACY_EARLY_20_R2']['loop_step_indexes']==list(range(20))
    for n in (1,5,10,20):
        for kind in ('STATIC','OBSERVED'):
            name=f'STEP_09_L{n:02d}_{kind}'
            assert windows[name]['loop_step_indexes']==list(range(9,9+n))
            assert windows[name]['memory_writer_calls']==1
    assert windows['STEP_09_L20_OBSERVED']['covered_delta_t']>=windows['LEGACY_EARLY_20_R2']['covered_delta_t']
    assert len(comparison_pairs(c))==11 and reference_arm(c)=='LEGACY_EARLY_20_R2'
    assert all(a in arms and b in arms for a,b in comparison_pairs(c))
