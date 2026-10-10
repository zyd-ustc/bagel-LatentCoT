"""Diagnostic contracts: observations preserve native results and branch isolation."""
import json
from dataclasses import replace
from types import MethodType
from types import SimpleNamespace
import pytest
import torch
from helpers import prepared
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache
from qwen_latent_cot.evaluation.increments import diagnose_fixed_state


def diagnostic_fixture(device='cpu'):
    model,kwargs,runtime=prepared(rounds=4,start=0,end=8,depth=28,special_tokens=True,device=device)
    runtime.config=replace(runtime.config,memory_update='full_depth_restart')
    model.llm2vae=torch.nn.Linear(32,32,bias=False).to(device=device,dtype=torch.bfloat16).eval()
    image=kwargs['packed_vae_token_indexes'];noise=kwargs['packed_query_sequence'][image].clone()
    null=dict(kwargs,past_key_values=NaiveCache(28),key_values_lens=torch.zeros(2,device=device,dtype=torch.int32),
        packed_key_value_indexes=torch.empty(0,device=device,dtype=torch.long),packed_query_indexes=torch.arange(11,device=device))
    def flow(this,**kw):
        sequence=kwargs['packed_query_sequence'].clone();sequence[image]=kw['x_t']
        conditional=runtime.decoder.forward_inference(**dict(kwargs,packed_query_sequence=sequence)).packed_query_sequence
        v=model.llm2vae(conditional)[image]
        if kw['cfg_text_scale']>1:
            unconditional=runtime.decoder.forward_inference(**dict(null,packed_query_sequence=sequence)).packed_query_sequence
            u=model.llm2vae(unconditional)[image]
            guided=u+kw['cfg_text_scale']*(v-u)
            return guided*(v.norm()/(guided.norm()+1e-8)).clamp(0,1)
        return v
    model._forward_flow=MethodType(flow,model)
    inputs=dict(x_t=noise,timestep=torch.full((len(noise),),.9,device=device),past_key_values=kwargs['past_key_values'],
        packed_vae_token_indexes=image,cfg_text_scale=4.)
    return model,runtime,inputs


@pytest.mark.parametrize('device_type',['cpu','npu'])
def test_fixed_state_observer_parity_complete_layers_and_velocity(tmp_path,device_type):
    from test_npu_runtime import npu_device
    device='cpu' if device_type=='cpu' else npu_device()
    model,runtime,inputs=diagnostic_fixture(device)
    try:
        before=model._forward_flow(**inputs)
        result,case=diagnose_fixed_state(model,runtime,inputs,tmp_path,0,.01)
        assert torch.equal(before,result) and all(case['contracts'].values())
        assert runtime.increment_observer is None and runtime.config.extra_rounds==4
        assert not model.llm2vae._forward_hooks
        rows=[json.loads(line) for line in (tmp_path/'layers.jsonl').read_text().splitlines()]
        velocity=[json.loads(line) for line in (tmp_path/'velocity.jsonl').read_text().splitlines()]
        assert all(r['finite'] for r in rows+velocity)
        for depth in range(5):
            assert {r['layer'] for r in rows if r['series']=='final_depth' and r['round']==depth and r['component']=='gen_output_hidden'}==set(range(28))
        locked=[r for r in rows if r['series']=='within_deepest' and r['layer']==0
                and r['comparison']=='adjacent_round' and r['component'] in ('memory_output_hidden','gen_input_K','gen_input_V')]
        assert locked and all(r['equal'] for r in locked)
        null=[r for r in velocity if r['component']=='null_text' and r['comparison']=='adjacent_round']
        assert len(null)==4 and all(r['equal'] for r in null)
        assert {r['round'] for r in velocity if r['component']=='conditional'}==set(range(5))
        assert len([r for r in velocity if r['component']=='cfg_post_renorm' and r['comparison']=='adjacent_round'])==4
    finally:runtime.close()


def test_diagnostic_exception_restores_layers_hooks_and_config(tmp_path):
    model,runtime,inputs=diagnostic_fixture();cfg=runtime.config
    layers=[layer.forward_inference for layer in runtime.decoder.layers]
    def fail(*args):raise RuntimeError('preview failure')
    try:
        with pytest.raises(RuntimeError,match='preview failure'):
            diagnose_fixed_state(model,runtime,inputs,tmp_path,0,.01,save_previews=fail)
        assert runtime.config==cfg and runtime.increment_observer is None
        assert not model.llm2vae._forward_hooks
        assert all(layer.forward_inference==old for layer,old in zip(runtime.decoder.layers,layers))
        assert torch.isfinite(model._forward_flow(**inputs)).all()
    finally:runtime.close()


def test_report_checks_complete_rounds_and_emits_portable_html(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/diagnose_increments.py'
    spec=importlib.util.spec_from_file_location('increment_cli',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,runtime,inputs=diagnostic_fixture()
    directory=tmp_path/'workers/worker_0/prompt_00000_s0';case_dir=directory/'step_00'
    try:_,case=diagnose_fixed_state(model,runtime,inputs,case_dir,0,.01)
    finally:runtime.close()
    plan=dict(config=dict(steps=[0],seeds=[0],max_rounds=4),prompt_ids=['p0'],source_sha256='test',comparison_scope='same x_t/t')
    monkeypatch.setattr(module,'read_plan',lambda path:plan)
    plan_path=tmp_path/'plan.json';plan_path.write_text('{}')
    (directory.parent/'run.json').write_text(json.dumps(dict(plan_sha256=module.sha256(plan_path))))
    trajectory=dict(prompt_id='p0',seed=0,index=0,prompt='test',path=str(directory),denoiser_steps=49,cases=[case])
    done=directory.parent/'completed.json';done.write_text(json.dumps([trajectory]))
    module.report(SimpleNamespace(plan=str(plan_path),output_dir=str(tmp_path)))
    assert (tmp_path/'diagnostics.html').is_file() and (tmp_path/'layers_summary.csv').is_file()
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['status']=='complete' and not summary['semantic_gain_verified']
    rows=(case_dir/'layers.jsonl').read_text().splitlines()
    (case_dir/'layers.jsonl').write_text('\n'.join(rows[:-1])+'\n')
    with pytest.raises(ValueError,match='truncated'):module.report(SimpleNamespace(plan=str(plan_path),output_dir=str(tmp_path)))


def test_native_sampler_advances_once_and_diagnostic_replay_preserves_trajectory(tmp_path):
    from contextlib import nullcontext
    from PIL import Image
    from qwen_latent_cot.evaluation.increments import sample_with_diagnostics
    model,runtime,inputs=diagnostic_fixture()
    runtime.config=replace(runtime.config,progress_end=19/48)
    original=model._forward_flow
    def generate(this,packed_init_noises,**unused):
        x=packed_init_noises.clone();ts=torch.linspace(1,0,50);ts=3*ts/(1+2*ts)
        for i,t in enumerate(ts[:-1]):
            runtime.progress=i/48
            x=x-model._forward_flow(**dict(inputs,x_t=x,timestep=torch.full((len(x),),t)))*(ts[i]-ts[i+1])
        return [x]
    model.generate_image=MethodType(generate,model)
    generator=SimpleNamespace(runtime=runtime,model=model,device=torch.device('cpu'),
        prepare=lambda *args:({'packed_init_noises':inputs['x_t'].clone()},['noise']),
        autocast=lambda:nullcontext(),decode=lambda *args:Image.new('RGB',(32,32),'gray'))
    try:
        expected=model.generate_image(packed_init_noises=inputs['x_t'])[0]
        observed=[]
        def decode(latent,shape):
            observed.append(latent.clone());return Image.new('RGB',shape[::-1],'gray')
        generator.decode=decode
        result=sample_with_diagnostics(generator,tmp_path,'test',(32,32),0,steps=[9],previews=False)
        assert result['denoiser_steps']==49 and len(result['cases'])==1
        assert torch.equal(observed[-1],expected)
        assert model._forward_flow==original and not runtime.layerwise.seeds
    finally:runtime.close()
