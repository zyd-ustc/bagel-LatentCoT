"""Feedback supervision on real tiny native BAGEL, including input gradients."""
import json
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from test_memory_topology import lm, runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.visual_objectives import training_objective
from qwen_latent_cot.diagnostics.flow_training import conditional_gen, TrainingState
from qwen_latent_cot.diagnostics.feedback_source import prompt_source, compose_source
from qwen_latent_cot.diagnostics.explicit_feedback import Feedback
from qwen_latent_cot.diagnostics.data import visual_label
from qwen_latent_cot.diagnostics.record import FrozenGuard


def configure(runtime, stage='C2-feedback'):
    cfg = json.loads(Path('configs/visual_feedback.json').read_text())
    cfg.update({k:runtime.config[k] for k in ('resolution','rank','caption_instruction','visual_rank',
                                           'response_layers','response_query_count')})
    cfg.update(stage=stage, source_layers=[0], steps=1, effective_batch=2, eval_every=1,
               sampler_steps=4, max_caption_tokens=2, train_t=[.2], eval_t=[.6], noise_seeds=[0])
    runtime.config = cfg
    return cfg


def feedback_row(row, *, satisfied=False):
    return visual_label(dict(row, task='feedback', flow_supervision=satisfied,annotation={'origin':'source_annotation','scope':'state' if row.get('state') else 'endpoint'}, label_kind='feedback', feedback=dict(
        observation='one red cup', gap='count and color',
        correction='' if satisfied else 'two blue cups', satisfied=satisfied)))


def test_teacher_velocity_gradients_reach_student_and_preserve_native(visual_runtime):
    runtime, row = visual_runtime; cfg = configure(runtime)
    cfg['loss_weights'].update(sem=0., flow=0., teacher=1.)
    student = VisualStudent(runtime,runtime.writer)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, student.text_teacher)
    row = feedback_row(row); state = runtime.training_state(row, .2, 3)
    values, info = training_objective(student, row, state, 'C2-feedback')
    assert values['teacher'].requires_grad and info['passes'] == 2
    values['teacher'].backward()
    assert student.visual.layers[0].up.weight.grad.abs().max() > 0
    assert student.writer.adapters[-1].up.weight.grad.abs().max() > 0
    assert all(p.grad is None for p in student.text_teacher.parameters())
    torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.001).step()
    guard.check()


def test_prompt_visible_outside_window_and_off_keeps_prompt(visual_runtime):
    runtime, row = visual_runtime; configure(runtime); student = VisualStudent(runtime,runtime.writer)
    state = runtime.training_state(row, .2, 2)
    p = prompt_source(state.flow, state.lengths, state.positions)
    g = conditional_gen(runtime, state).source
    off = student.write(g, state.t, prompt=p, source_off=True)
    # Compare against a text-only source processed by the same visual queries.
    expected = student.writer(runtime.seeds, p, torch.arange(2048,2064),
                             source_kind='visual', t=state.t, visual_adapter=student.visual)
    for a,b in zip(off.layers,expected.layers):
        torch.testing.assert_close(a.keys,b.keys,rtol=0,atol=0)
    changed = tuple(replace(kv, values=kv.values+3) if kv.layer==2 else kv for kv in p)
    a = student.write(g,state.t,prompt=p); b = student.write(g,state.t,prompt=changed)
    assert not torch.equal(a.layers[2].values,b.layers[2].values)
    composed = compose_source(p,g,[0])
    assert composed[2] is p[2] and composed[0].lengths[0] > p[0].lengths[0]


def test_saved_trajectory_has_teacher_target_and_no_fake_flow(visual_runtime, tmp_path):
    runtime,row = visual_runtime; configure(runtime); student = VisualStudent(runtime,runtime.writer)
    from qwen_latent_cot.diagnostics.feedback_dataset import collect_feedback_states
    runtime.text_teacher_hash='test'
    exported=collect_feedback_states(runtime,[feedback_row(dict(row,prompt='target',noise_seed=3))],tmp_path/'states',[1])
    row=dict(exported[0],state=str(tmp_path/'states'/exported[0]['state']),task='feedback',
             annotation={'origin':'manual_checked','scope':'state'},flow_supervision=False)
    row.pop('teacher')
    row=visual_label(row)
    t=row['t']
    state = runtime.training_state(row,t,3)
    values,info = training_objective(student,row,state,'C2-feedback')
    assert state.x0 is None and state.epsilon is None
    assert values['teacher'].requires_grad and float(values['flow']) == 0
    with pytest.raises(ValueError,match='no paired'):
        state.flow_target()
    # A reload/resave cycle keeps the native grid.
    torch.save(dict(state.metadata,x_t=state.x_t),tmp_path/'resaved.pt')
    runtime.state(dict(row,state=str(tmp_path/'resaved.pt')),t,3)


def test_loop_uses_fresh_feedback_for_later_round(visual_runtime):
    runtime,row = visual_runtime; cfg=configure(runtime,'C3-loop'); cfg['feedback_updates']=2
    student=VisualStudent(runtime,runtime.writer); row=feedback_row(row); state=runtime.training_state(row,.2,3)
    with pytest.raises(ValueError,match='online feedback'):
        training_objective(student,row,state,'C3-loop')
    calls=[]
    def provider(actual,previous_bank):
        calls.append((actual,previous_bank))
        return Feedback('two red cups','color','two green cups',origin='bagel_auto')
    values,info=training_objective(student,row,state,'C3-loop',feedback_provider=provider)
    assert len(calls)==1 and calls[0][0] is state and calls[0][1].slots==16
    assert [r['correction'] for r in info['rounds']]==['two blue cups','two green cups']
    sum(values.values()).backward()
    assert info['passes']==3 and info['feedback_updates']==2


def test_satisfied_target_is_masked_layout_not_visible_zero_KV(visual_runtime, monkeypatch):
    runtime,row=visual_runtime;cfg=configure(runtime);cfg['loss_weights'].update(sem=0.,flow=0.)
    student=VisualStudent(runtime,runtime.writer);row=feedback_row(row,satisfied=True)
    state=runtime.training_state(row,.2,0)
    import qwen_latent_cot.diagnostics.visual_objectives as module
    original=module.conditional_gen; calls=[]
    def trace(runtime,actual,bank=None,**kwargs):
        calls.append((actual,bank));return original(runtime,actual,bank,**kwargs)
    monkeypatch.setattr(module,'conditional_gen',trace)
    values,_=training_objective(student,row,state,'C2-feedback')
    actual,bank=calls[-1]
    assert bank is None
    torch.testing.assert_close(actual.flow['packed_position_ids'],state.flow['packed_position_ids']+16)
    values['teacher'].backward()


def test_c1_feedback_compression_uses_full_native_text_velocity(visual_runtime):
    from qwen_latent_cot.diagnostics.caption import prepare_source, correction_velocity_loss
    runtime,row=visual_runtime;configure(runtime);runtime.writer.requires_grad_(True)
    row=feedback_row(row)
    source,_=prepare_source(runtime,row,'C1')
    bank=runtime.memory(source)
    loss=correction_velocity_loss(runtime,row,bank)
    loss.backward()
    assert runtime.writer.adapters[-1].up.weight.grad.abs().max()>0
    assert all(p.grad is None for p in runtime.model.parameters())


def test_e1_export_and_cached_teacher_equal_live_teacher(visual_runtime,tmp_path):
    from qwen_latent_cot.diagnostics.feedback_dataset import collect_feedback_states
    from qwen_latent_cot.diagnostics.data import load_dataset
    runtime,row=visual_runtime;configure(runtime);runtime.text_teacher_hash='test'
    row=feedback_row(row);output=tmp_path/'export'
    exported=collect_feedback_states(runtime,[dict(row,noise_seed=3)],output,[0,1])
    rows=load_dataset(output/'feedback.jsonl',visual=True)
    assert len(rows)==2 and all(r['family_id']==row['family_id'] for r in rows)
    student=VisualStudent(runtime,runtime.writer);student.text_teacher_hash='test'
    row=dict(rows[1],task='feedback',flow_supervision=False,annotation={'origin':'manual_checked','scope':'state'});state=runtime.training_state(row,row['t'],3)
    cached,info=training_objective(student,row,state,'C2-feedback')
    live,other=training_objective(student,{k:v for k,v in row.items() if k!='teacher'},state,'C2-feedback')
    torch.testing.assert_close(cached['teacher'],live['teacher'],rtol=0,atol=0)
    assert info['gen_calls']==other['gen_calls']-1
    cached['teacher'].backward()
    student.text_teacher_hash='different'
    with pytest.raises(ValueError,match='cached teacher'):
        training_objective(student,row,state,'C2-feedback')


def test_c1_feedback_warm_start_trains_and_validates(visual_runtime,tmp_path):
    from qwen_latent_cot.diagnostics.caption import train
    from qwen_latent_cot.diagnostics.record import save_checkpoint
    runtime,row=visual_runtime;cfg=configure(runtime)
    cfg.update(c1_feedback=True,feedback_velocity_weight=1.)
    checkpoint=tmp_path/'c1.pt';save_checkpoint(checkpoint,runtime,0,'C1')
    rows=[feedback_row(row),feedback_row(dict(row,sample_id='val',family_id='val',split='val'))]
    output=tmp_path/'warm';output.mkdir()
    train(runtime,rows,'C1',output,str(checkpoint))
    assert (output/'last.pt').exists() and (output/'validation_000001/captions.jsonl').exists()
    assert json.loads((output/'selection.json').read_text())['fact_accuracy'] is None
    assert json.loads((output/'training_summary.json').read_text())['best_validation_fact_accuracy'] is None
    assert all(p.grad is None for p in runtime.model.parameters())


def test_explicit_weight_init_preserves_selected_c1(visual_runtime,tmp_path):
    from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint,initialize_student,load_visual_checkpoint
    from qwen_latent_cot.diagnostics.record import save_checkpoint
    runtime,row=visual_runtime;configure(runtime,'C2-observe')
    old=VisualStudent(runtime,runtime.writer);old.text_teacher_hash='old'
    with torch.no_grad():old.visual.layers[0].up.weight.fill_(.125)
    path=tmp_path/'observe.pt';save_visual_checkpoint(path,old,1,'C2-observe')
    with torch.no_grad():runtime.writer.adapters[0].up.weight.fill_(.5)
    c1=tmp_path/'c1.pt';save_checkpoint(c1,runtime,1,'C1')
    new=initialize_student(runtime,c1,path,teacher_path=c1)
    torch.testing.assert_close(new.writer.adapters[0].up.weight,torch.full_like(new.writer.adapters[0].up.weight,.5))
    torch.testing.assert_close(new.visual.layers[0].up.weight,old.visual.layers[0].up.weight)
    with pytest.raises(ValueError,match='provenance differs'):
        load_visual_checkpoint(path,new)
