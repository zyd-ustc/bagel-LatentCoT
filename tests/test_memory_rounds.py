from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import pytest
import torch
from helpers import prepared
from qwen_latent_cot.evaluation.memory_rounds import MemoryRoundCapture,kv_metrics,tensor_metrics,velocity_comparisons


def test_content_metrics_do_not_hide_change_with_fixed_special_slots():
    a=torch.ones(3,2,4);b=a.clone();b[1]+=2
    rows=kv_metrics((a,a),(b,b),torch.tensor([True,False,True]))
    by={(r['subset'],r['component']):r for r in rows}
    assert by['special','K']['equal'] and by['special','V']['equal']
    assert by['content','K']['relative_l2']==2
    assert by['all','K']['relative_l2']==pytest.approx(2/math.sqrt(3))
    assert not tensor_metrics(torch.ones(1),torch.tensor([float('nan')]))['finite']


def test_observer_preserves_output_and_captures_actual_final_reads_at_all_depths():
    model,kwargs,runtime=prepared(batch=1,mode='LAYERWISE_UND_STATE_REPLACE',rounds=3,
        start=0,special_tokens=True)
    decoder=model.language_model.model;config=runtime.config;capture=MemoryRoundCapture(3)
    try:
        for depth in (1,2,3):
            runtime.config=replace(config,extra_rounds=depth)
            runtime.kv_observer=None
            expected=decoder.forward_inference(**kwargs).packed_query_sequence
            runtime.kv_observer=capture
            actual=decoder.forward_inference(**kwargs).packed_query_sequence
            assert torch.equal(actual,expected)
        rows=capture.comparisons([1,2,3])
        assert all(r['finite'] for r in rows)
        assert all(r['equal'] for r in rows if r['subset']=='special')
        assert any(not r['equal'] for r in rows if r['scope']=='independent_depth_final_read' and r['layer']==0 and r['subset']=='content')
        assert set(capture.reads[1])==set(range(4))
        assert not any(r['phase']=='suffix' for r in capture.writer_rows)
        assert any(not r['equal'] for r in rows if r['subset']=='content' and r['from_round']==1)
        # The body writer prefix is the same in independent R1/R2 and within R3.
        for previous,depth in ((0,1),(1,2),(2,3)):
            for layer in (0,1,2):
                for component in ('K','V'):
                    matched=[r for r in rows if r['from_round']==previous and r['to_round']==depth
                             and r['layer']==layer and r['component']==component and r['subset']=='content']
                    assert len(matched)==2
                    assert matched[0]['relative_l2']==matched[1]['relative_l2']
        capture.reads[2].pop(3)
        with pytest.raises(ValueError,match='incomplete'):capture.comparisons([1,2,3])
    finally:runtime.close()


def test_velocity_reports_increment_separately_from_change_vs_base():
    values={i:torch.full((2,3),v) for i,v in enumerate((1.,2.,2.5,2.5))}
    rows=velocity_comparisons(values,'conditional')
    by={(r['from_round'],r['to_round']):r for r in rows}
    assert by[1,2]['delta_over_first_effect']==pytest.approx(.5)
    assert by[2,3]['equal'] and by[2,3]['relative_l2']==0
    assert by[0,3]['relative_l2']==pytest.approx(1.5)


def test_summary_uses_sum_of_squared_layer_update_norms():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate/merge_memory_rounds.py'
    spec=importlib.util.spec_from_file_location('round_merge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for step in (0,24):
        memory=[]
        for layer in (0,1,2):
            for to_round,norm in ((1,4.),(2,2.),(3,1.)):
                for component in ('K','V'):
                    memory.append(dict(scope='within_deepest_call_writer',phase='body',layer=layer,
                        subset='content',component=component,from_round=to_round-1,to_round=to_round,
                        relative_l2=norm,delta_norm=norm,cosine=.8,equal=False))
        records.append(dict(memory_mode='LAYERWISE_UND_STATE_REPLACE',step=step,memory=memory,velocity=[]))
    summary=module.summarize(records)
    assert len(summary['writer_update_contraction'])==8
    assert all(r['ratio']['mean']==pytest.approx(.5) for r in summary['writer_update_contraction'])


def test_bfloat16_velocity_denominator_matches_float_delta():
    values={0:torch.tensor([1.,10.],dtype=torch.bfloat16),
            1:torch.tensor([4.03125,13.0625],dtype=torch.bfloat16)}
    values[2]=values[1];values[3]=values[1]
    first=velocity_comparisons(values,'conditional')[0]
    assert first['delta_over_first_effect']==1.


def test_worker_keeps_base_trajectory_and_merge_rejects_missing_probe(tmp_path,monkeypatch):
    """Exercise the real diagnostic orchestration using only a tiny CPU decoder."""
    from contextlib import nullcontext
    import json
    import sys
    from types import MethodType,SimpleNamespace
    from qwen_latent_cot.bagel import backbone,inferencer
    model,kwargs,old_runtime=prepared(batch=1,mode='LAYERWISE_UND_STATE_REPLACE',rounds=3,
        start=0,special_tokens=True)
    cache=kwargs['past_key_values'];seed=old_runtime.layerwise.seeds[cache]
    old_runtime.close()
    base_decoder=model.language_model.model.forward_inference
    calls=[]
    def flow(this,*,x_t,cfg_text_scale=1.,**unused):
        return model.language_model.model.forward_inference(**{**kwargs,'packed_query_sequence':x_t}).packed_query_sequence*cfg_text_scale
    def generate(this,*,packed_init_noises,num_timesteps,cfg_text_scale,**unused):
        x=packed_init_noises.clone()
        for step in range(num_timesteps-1):
            expected=base_decoder(**{**kwargs,'packed_query_sequence':x}).packed_query_sequence*cfg_text_scale
            prediction=this._forward_flow(x_t=x,timestep=torch.full((len(x),),1-step*.25),
                cfg_text_scale=cfg_text_scale,past_key_values=cache)
            assert torch.equal(prediction,expected)  # Dynamic probes never drive the trajectory.
            calls.append(step);x=x-prediction*.01
        return [x]
    model._forward_flow=MethodType(flow,model);model.generate_image=MethodType(generate,model)
    original_flow=model._forward_flow
    class Generator:
        def __init__(self,bundle,runtime):self.runtime=runtime;self.prompt_lengths=(3,)
        def autocast(self):return nullcontext()
        def prepare(self,prompts,shapes,seeds):
            self.runtime.layerwise.seeds[cache]=seed
            return dict(packed_init_noises=kwargs['packed_query_sequence'].clone(),past_key_values=cache),['test_noise']
    monkeypatch.setattr(backbone,'load_native',lambda *a:SimpleNamespace(model=model))
    monkeypatch.setattr(inferencer,'T2IGenerator',Generator)
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:None)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('round_worker',root/'scripts/evaluate/diagnose_memory_rounds.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    prompts=tmp_path/'prompts.jsonl';prompts.write_text(json.dumps(dict(prompt='A red cube.',prompt_id='a'))+'\n')
    run=tmp_path/'run';out=run/'worker_0';run.mkdir()
    arguments=['diagnose','--model-path',str(tmp_path),'--prompts',str(prompts),'--output-dir',str(out),
        '--prompt-count','1','--num-shards','1','--device','cpu','--num-timesteps','4',
        '--probe-steps','0,1,2','--end-layer','3','--plan',str(run/'plan.json')]
    args=worker.parser().parse_args(arguments[1:]);_,_,plan=worker.inputs(args)
    plan['model_sha256']={'tiny_cpu_fixture':'fixture'}
    (run/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(sys,'argv',arguments);worker.main()
    assert calls==[0,1,2] and model._forward_flow==original_flow
    spec=importlib.util.spec_from_file_location('round_merge_cli',root/'scripts/evaluate/merge_memory_rounds.py')
    merge=importlib.util.module_from_spec(spec);spec.loader.exec_module(merge)
    monkeypatch.setattr(sys,'argv',['merge','--run-dir',str(run)]);merge.main()
    summary=json.loads((run/'summary.json').read_text())
    assert summary['probes']==3 and summary['hidden']
    measured=[json.loads(s) for s in (out/'samples.jsonl').read_text().splitlines()]
    assert len(measured)==3 and {r['step'] for r in measured}=={0,1,2}
    assert all(all(r['sanity'].values()) for r in measured)
    rows=(out/'samples.jsonl').read_text().splitlines()
    (out/'samples.jsonl').write_text('\n'.join(rows[:-1])+'\n')
    with pytest.raises(ValueError,match='incomplete shard'):merge.main()
