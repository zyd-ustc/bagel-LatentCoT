"""Data legality, family isolation, bounded sampling and native task training."""
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from test_memory_topology import lm,runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.diagnostics.corpus import build,Corpus,json_records,normalized_records
from qwen_latent_cot.diagnostics.curriculum import NoisePolicy,CoverageSampler
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.visual_objectives import training_objective


def image(tmp,name,color):
    path=tmp/name;Image.new('RGB',(32,32),color).save(path);return str(path)


def tiny_corpus(tmp):
    a=image(tmp,'a.png','red');b=image(tmp,'b.png','blue');c=image(tmp,'c.png','green')
    rows=[dict(id='edit',image=[a],image_gen=b,input_caption='a red cup',output_caption='a blue cup',
               conversations=[dict(from_='human',value='')])]
    rows[0]['conversations']=[{'from':'human','value':'<image> Edit the image: Make the cup blue.'},{'from':'gpt','value':'[TASK2]'}]
    edit=tmp/'edit.json';edit.write_text(json.dumps(rows))
    val=tmp/'val.jsonl';val.write_text(json.dumps(dict(sample_id='val',image=c,caption='a green cup',prompt='a green cup',split='val'))+'\n')
    root=tmp/'corpus';build(dict(sources=[dict(name='edit',format='unimoe',path=str(edit),split='train'),dict(name='val',format='paired',path=str(val),inherit_split=True)]),root,tokenizer=lambda s:s.split(),shard_size=1)
    return Corpus(root)


def test_endpoint_roles_have_distinct_flow_legality_and_shared_episode(tmp_path):
    corpus=tiny_corpus(tmp_path);views=corpus.groups('train',{'endpoint'})
    rows=[views[0][i] for i in range(len(views[0]))];assert len(rows)==2
    repair,hold=sorted(rows,key=lambda r:r['feedback']['satisfied'])
    assert not repair['flow_supervision'] and hold['flow_supervision']
    assert repair['prompt']==hold['prompt']=='a blue cup' and repair['pair_id']==hold['pair_id']
    assert repair['annotation']['origin']=='source_annotation'
    cfg=json.loads(Path('configs/visual_feedback.json').read_text());sampler=CoverageSampler(corpus,dict(cfg,task_mix={'endpoint':1}),'C2-feedback')
    x,t,seed=sampler.next(1);y,u,other=sampler.next(1)
    assert x['feedback']['satisfied']!=y['feedback']['satisfied'] and (t,seed)==(u,other)
    saved=sampler.state_dict();expected=[sampler.next(2) for _ in range(12)]
    resumed=CoverageSampler(corpus,dict(cfg,task_mix={'endpoint':1}),'C2-feedback');resumed.load_state_dict(saved)
    assert expected==[resumed.next(2) for _ in range(12)]
    assert 'order' not in saved;corpus.close()


def test_placeholder_not_a_family_and_heldout_alias_overrides_train(tmp_path):
    blank=image(tmp_path,'blank.png','black');a=image(tmp_path,'a.png','red');b=image(tmp_path,'b.png','blue')
    rows=[dict(id=str(i),image=[blank],image_gen=p,conversations=[{'from':'human','value':'Generate an image: a cup'},{'from':'gpt','value':'[TASK0]'}]) for i,p in enumerate((a,b))]
    data=tmp_path/'raw.json';data.write_text(json.dumps(rows))
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(dict(sample_id='held',image=a,caption='a cup',split='test'))+'\n')
    root=tmp_path/'corpus';build(dict(sources=[dict(name='raw',format='unimoe',path=str(data),split='train'),dict(name='held',format='paired',path=str(held),diagnostic=True,inherit_split=True)]),root)
    corpus=Corpus(root);train=corpus.groups('train',{'generation'});test=corpus.groups('test',{'generation'})
    assert len(train[0])==len(test[0])==1
    assert train[0][0]['family_id']!=test[0][0]['family_id'];corpus.close()


def test_streamed_array_crosses_chunk_boundary_and_rejects_truncation(tmp_path):
    path=tmp_path/'large.json';rows=[dict(text='中'*(1024*1024+13)),dict(text='end')]
    path.write_text(json.dumps(rows,ensure_ascii=False));assert list(json_records(path))==rows
    path.write_text('[{"a":1}');
    with pytest.raises(ValueError):list(json_records(path))


def test_high_noise_masks_and_curriculum_calibration():
    cfg=json.loads(Path('configs/visual_feedback.json').read_text());policy=NoisePolicy(cfg)
    repair=dict(task='feedback',flow_supervision=False,annotation={'scope':'endpoint'},feedback={'correction':'make blue','satisfied':False})
    hold=dict(repair,flow_supervision=True,feedback={'correction':'','satisfied':True})
    assert policy.masks(repair,.2,teacher_available=True).asdict()==dict(sem=True,teacher=True,flow=False)
    assert not any(policy.masks(repair,1,teacher_available=True).asdict().values())
    assert policy.masks(hold,1,teacher_available=True).asdict()==dict(sem=False,teacher=False,flow=True)
    assert not policy.masks(repair,.2,pass_index=1).sem
    actual=dict(repair,state='trajectory',annotation={'scope':'state'})
    assert policy.masks(actual,.95).sem and not policy.masks(actual,1).sem
    cfg['noise_curriculum'].append(dict(start_step=2,observe_max_t=.5,endpoint_max_t=.5))
    assert NoisePolicy(cfg,2).observe_max==.25


def tiny_config(runtime,stage):
    cfg=json.loads(Path('configs/visual_memory_pilot.json' if stage=='C2-observe' else 'configs/visual_feedback.json').read_text())
    cfg.update({k:runtime.config[k] for k in ('resolution','rank','visual_rank','caption_instruction')})
    cfg.update(stage=stage,source_layers=[0,1,2],steps=1,effective_batch=2,eval_every=1,sampler_steps=4,
               max_caption_tokens=2,output_token_budgets=[2],noise_seeds=[0],validation_max_rows=2,
               task_mix={'generation':1},loss_weights=dict(sem=1 if stage=='C2-observe' else 0,teacher=0,flow=0 if stage=='C2-observe' else 1))
    runtime.config=cfg;return cfg


def test_native_condition_prefix_keeps_prompt_separate_and_input_gradients(visual_runtime,tmp_path):
    runtime,row=visual_runtime;tiny_config(runtime,'C2-feedback')
    condition=image(tmp_path,'condition.png','blue');row=dict(row,condition_images=[condition])
    state=runtime.training_state(row,.9,3)
    assert state.lengths[0]>state.positions[0]
    assert state.prompt_kv[0].lengths[0]==len(state.prompt_kv[0].positions)
    assert 'cfg_img_past_key_values' in state.flow
    student=VisualStudent(runtime);values,report=training_objective(student,row,state,'C2-feedback')
    values['flow'].backward()
    assert student.visual.layers[0].up.weight.grad.abs().max()>0
    assert all(p.grad is None for p in runtime.model.parameters())


@pytest.mark.parametrize('stage',['C2-observe','C2-feedback'])
def test_indexed_training_resume_matches_uninterrupted(visual_runtime,tmp_path,stage):
    from qwen_latent_cot.diagnostics.visual_training import train
    runtime,_=visual_runtime;cfg=tiny_config(runtime,stage);corpus=tiny_corpus(tmp_path)
    torch.manual_seed(41);initial=VisualStudent(runtime);weights={k:v.clone() for k,v in initial.state_dict().items()}
    output=tmp_path/'first';output.mkdir();train(initial,corpus,stage,output,skip_validation=True)
    assert (output/'last.pt').exists()
    cfg['steps']=2;resumed=tmp_path/'resume';resumed.mkdir()
    train(initial,corpus,stage,resumed,resume=str(output/'last.pt'),skip_validation=True)
    wanted={k:v.clone() for k,v in initial.state_dict().items()}
    initial.load_state_dict(weights);torch.manual_seed(41)
    complete=tmp_path/'complete';complete.mkdir();train(initial,corpus,stage,complete,skip_validation=True)
    for k,v in initial.state_dict().items():torch.testing.assert_close(v,wanted[k],rtol=0,atol=0)
    logs=[json.loads(s) for s in (complete/'training.jsonl').read_text().splitlines()]
    resumed_log=json.loads((resumed/'training.jsonl').read_text())
    assert logs[1]['sampled_states']==resumed_log['sampled_states']
    assert all(p.grad is None for p in runtime.model.parameters());corpus.close()


def test_repair_has_no_reconstruction_fallback(visual_runtime):
    runtime,row=visual_runtime;tiny_config(runtime,'C2-feedback');runtime.config['loss_weights']=dict(sem=1,teacher=0,flow=1)
    row=dict(row,task='feedback',flow_supervision=False,feedback=dict(observation='',gap='',correction='make blue',satisfied=False,origin='source_annotation'))
    state=runtime.training_state(row,.2,3)
    with pytest.raises(ValueError):state.flow_target()
    student=VisualStudent(runtime);values,_=training_objective(student,row,state,'C2-feedback')
    assert float(values['flow'])==0 and values['sem'].requires_grad
    runtime.config['feedback_updates']=2
    with pytest.raises(ValueError,match='online feedback'):training_objective(student,row,state,'C3-loop')


def test_pair_draws_do_not_distort_task_quota(tmp_path):
    from collections import Counter
    corpus=tiny_corpus(tmp_path);cfg=json.loads(Path('configs/visual_feedback.json').read_text())
    sampler=CoverageSampler(corpus,cfg,'C2-feedback');counts=Counter()
    for _ in range(1000):
        row,_,_=sampler.next(1);counts['endpoint' if row['task']=='feedback' else 'conditional' if row['condition_images'] else row['task']]+=1
    for task,weight in sampler.weights.items():assert abs(counts[task]/1000-weight)<.004
    corpus.close()


def test_long_caption_keeps_generation_and_short_hold_label(tmp_path):
    a=image(tmp_path,'a.png','red');b=image(tmp_path,'b.png','blue');goal=' '.join(['blue']*600)
    data=tmp_path/'long.json';data.write_text(json.dumps([dict(id='long',image=[a],image_gen=b,input_caption='red cup',output_caption=goal,
        conversations=[{'from':'human','value':'Edit the image: make it blue'},{'from':'gpt','value':'[TASK2]'}])]))
    root=tmp_path/'corpus';report=build(dict(sources=[dict(name='long',format='unimoe',path=str(data),split='train')]),root,tokenizer=lambda s:s.split())
    corpus=Corpus(root);assert report['counts']['context_ineligible:observe']==1
    assert len(corpus.groups('train',{'generation'})[0])==1
    endpoint=corpus.groups('train',{'endpoint'})[0];assert len(endpoint)==2
    hold=next(endpoint[i] for i in range(2) if endpoint[i]['feedback']['satisfied'])
    assert hold['prompt']==goal and hold['caption']=='No change is needed.';corpus.close()


def test_pure_noise_trajectory_stays_out_of_train_views(tmp_path):
    state=tmp_path/'state.pt';torch.save({},state)
    data=tmp_path/'trajectory.jsonl';data.write_text(json.dumps(dict(id='noise',state=str(state),t=1,noise_seed=3,prompt='blue cup',
         feedback=dict(observation='',gap='',correction='make blue',satisfied=False,origin='bagel_auto')))+'\n')
    root=tmp_path/'corpus';report=build(dict(sources=[dict(name='trajectory',format='trajectory',path=str(data),split='train')]),root)
    assert report['counts']['pure_noise_diagnostic']==1 and report['views']==[]


def test_modified_asset_is_detected_before_training(visual_runtime,tmp_path):
    runtime,_=visual_runtime;tiny_config(runtime,'C2-feedback');corpus=tiny_corpus(tmp_path)
    row=corpus.groups('train',{'generation'})[0][0]
    Image.new('RGB',(32,32),'yellow').save(row['image'])
    with pytest.raises(ValueError,match='asset changed'):runtime.training_state(row,.9,3)
    corpus.close()


def test_feedback_validation_uses_native_cfg_suffix_without_relabeling(visual_runtime,tmp_path):
    from qwen_latent_cot.diagnostics.task_evaluation import evaluate_tasks
    runtime,row=visual_runtime;cfg=tiny_config(runtime,'C2-feedback')
    cfg['loss_weights']=dict(sem=1,teacher=0,flow=0)
    row=dict(row,task='feedback',flow_supervision=False,feedback=dict(observation='',gap='',correction='make blue',satisfied=False,origin='source_annotation'))
    output=tmp_path/'validation';output.mkdir();records=evaluate_tasks(VisualStudent(runtime),[row],output,NoisePolicy(cfg),'C2-feedback')
    record=records[0];assert record['state']['t']<=.25 and record['image_state']['t']>.25
    assert record['supervision']['masks']['sem']
    assert {'base','memory','source_off'}==set(record['images'])
    assert all((output/name).is_file() for name in record['images'].values())


def test_c3_online_endpoint_sampling_uses_calibrated_native_nodes(tmp_path):
    import random
    cfg=json.loads(Path('configs/memory_loop_r2.json').read_text());policy=NoisePolicy(cfg)
    row=dict(task='feedback');rng=random.Random(0)
    for _ in range(20):
        t=policy.time(row,rng);assert 0<t<=.25 and t in policy.native[:-1]
