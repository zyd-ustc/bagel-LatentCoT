"""Stage A information isolation, applied residual and real optimizer step."""
import json
from pathlib import Path
import pytest
import torch
from test_memory_topology import lm, runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.interaction.stage_a import split_prompt, prompt_units, training_state, losses, memory_controls
from qwen_latent_cot.interaction.student import InteractionStudent
from qwen_latent_cot.diagnostics.flow_training import conditional_gen


@pytest.fixture
def stage_student(visual_runtime,tmp_path):
    from qwen_latent_cot.diagnostics.visual import VisualStudent
    from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint
    runtime,row=visual_runtime
    path=tmp_path/'parent.pt'
    save_visual_checkpoint(path,VisualStudent(runtime),1000,'C2-observe')
    parent=torch.load(path,weights_only=True)
    cfg=json.loads(Path('configs/interaction_stage_a.json').read_text())
    cfg.update(writer_rank=4,consumer_rank=4,min_prompt_words=0,full_prompt_probability=0.)
    student=InteractionStudent(runtime,parent,cfg)
    return student,runtime.training_state(row,.9,3)


def test_whole_sentences_quotes_and_main_clause():
    text='A red vase is on a shelf. A sign reads "BIRDS. FLY"; three blue birds sit below. The sky is green.'
    split=split_prompt(text,17,min_words=0,full_probability=0)
    assert split.partial.startswith('A red vase is on a shelf.')
    assert split.removed and split.partial != text
    assert ''.join(split.units)==text
    assert '"BIRDS. FLY"' in split.partial or '"BIRDS. FLY"' in split.removed
    assert split == split_prompt(text,17,min_words=0,full_probability=0)
    assert split_prompt('Three red books on a table.',17,min_words=0).reason=='unsplittable'
    assert split_prompt(text,17,min_words=999).reason=='short'
    assert split_prompt(text,17,min_words=0,full_probability=1).reason=='full_mix'
    assert len(prompt_units('Dr. Smith holds 2.5 apples. A blue vase.'))==2


def test_dual_prompt_state_keeps_gen_cache_partial(stage_student,visual_runtime):
    student,_=stage_student
    _,row=visual_runtime
    row=dict(row,caption='A red token. A blue token. A green token.')
    state,split=training_state(student,row,.9,3,17)
    assert state.metadata['prompt']==split.partial and state.metadata['full_prompt']==split.full
    partial,_,_=student.runtime.flow_inputs(split.partial)
    full,_,_=student.runtime.flow_inputs(split.full)
    for layer in range(3):
        torch.testing.assert_close(state.flow['past_key_values'].key_cache[layer],partial['past_key_values'].key_cache[layer])
    assert len(state.prompt_kv[0].keys)==len(full['past_key_values'].key_cache[0])
    assert len(state.flow['past_key_values'].key_cache[0])<len(state.prompt_kv[0].keys)
    original=student.runtime.training_state(dict(row,prompt=split.full),.9,3)
    torch.testing.assert_close(state.x_t,original.x_t)
    torch.testing.assert_close(state.flow_target(),original.flow_target())


def test_applied_residual_precedes_cfg_and_preserves_reference(stage_student):
    student,state=stage_student
    runtime=student.runtime
    with torch.no_grad():
        base=conditional_gen(runtime,state)
        bank=student.write(base.source,state.t,prompt=state.prompt_kv)
        raw=conditional_gen(runtime,state,bank,capture=False)
        scaled=conditional_gen(runtime,state,bank,capture=False,velocity_baseline=base.velocity,velocity_scale=.5)
        expected=(base.velocity.float()+.5*(raw.velocity.float()-base.velocity.float())).to(raw.velocity.dtype)
        torch.testing.assert_close(scaled.velocity,expected,rtol=0,atol=0)
        guided_native=conditional_gen(runtime,state,guidance=True,capture=False)
        guided_zero=conditional_gen(runtime,state,bank,guidance=True,capture=False,
                                    velocity_baseline=base.velocity,velocity_scale=0.)
        torch.testing.assert_close(guided_zero.velocity,guided_native.velocity,rtol=0,atol=0)


def test_stage_a_real_loss_gradients_controls_and_resume(stage_student,visual_runtime,tmp_path):
    from qwen_latent_cot.interaction.training import Trainer,train_stage_a
    from qwen_latent_cot.interaction import checkpoint
    student,_=stage_student
    runtime,row=visual_runtime
    row=dict(row,caption='A red token. A blue token.',sample_id='a',family_id='a')
    donor=dict(row,caption='A green token. A white token.',sample_id='b',family_id='b')
    student.config.update(steps=1,global_batch=2,val_images=0,save_steps=[1],warmup_steps=0)
    trainer=Trainer(student,[row,donor],[],tmp_path/'train','stage_a',{'train':'tiny'})
    train_stage_a(trainer)
    log=json.loads((tmp_path/'train/train.jsonl').read_text())
    assert log['semantic_ce']>0 and log['masked_fraction']==1
    assert log['gradient_writer']>0 and log['gradient_bridge']>0
    # With alpha=0 the query adapter starts learning after the memory gate opens.
    assert log['gradient_consumer']==0
    assert any(v != 0 for v in log['memory_gates'])
    assert log['cos_loss']==0.0 and 'raw_correction_loss' not in log
    assert log['velocity_scale']==[1.0]
    assert all(name.endswith('q_proj') for name in student.adapters.consumer)
    saved=checkpoint.read(tmp_path/'train/step_000001.pt')
    assert saved['kind']==checkpoint.STAGE_A_KIND and saved['stage']=='stage_a'
    assert 'vel_scale' in saved['bridge'] and len(saved['adapters'])==18
    before=student.bridge.vel_scale.detach().clone()
    with torch.no_grad(): student.bridge.vel_scale.zero_()
    checkpoint.apply(saved,student)
    torch.testing.assert_close(student.bridge.vel_scale,before)
    student.config['steps']=4
    resumed=Trainer(student,[row,donor],[],tmp_path/'train','stage_a',{'train':'tiny'},saved=saved)
    train_stage_a(resumed)
    assert checkpoint.read(tmp_path/'train/step_000004.pt')['step']==4
    records=[json.loads(line) for line in (tmp_path/'train/train.jsonl').read_text().splitlines()]
    assert any(record['gradient_consumer']>0 for record in records[1:])
    # BF16 may round tiny applied corrections to zero during the first steps.
    assert any(record['delta_v_rms']>0 for record in records[1:])


def test_v1_generation_corpus_keeps_splits_and_sampler_resume(tmp_path):
    from qwen_latent_cot.diagnostics.corpus import REF
    from qwen_latent_cot.interaction.data import GenerationCorpus,CorpusPairs,corpus_validation
    (tmp_path/'records').mkdir();(tmp_path/'views').mkdir()
    offsets=[]
    with (tmp_path/'records/0.jsonl').open('wb') as stream:
        for i in range(4):
            offsets.append(stream.tell())
            value=dict(record_id=str(i),family_id='family'+str(i),source='v1',split='train' if i<2 else 'val',
                       target_caption='A red object. A blue object.',target_image=f'/images/{i}.png',
                       original={'path':'/annotations/data.jsonl'},annotation={'scope':'endpoint'})
            stream.write((json.dumps(value)+'\n').encode())
    views=[]
    for split,indexes in (('train',(0,1)),('val',(2,3))):
        (tmp_path/f'views/{split}.idx').write_bytes(b''.join(REF.pack(0,offsets[i],2) for i in indexes))
        views.append(dict(path=split+'.idx',task='generation',source='v1',split=split,count=2))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(format='annotation_index_v1',fingerprint='test_v1',
                registry={'roots':{}},shards=[{'path':'0.jsonl'}],views=views)))
    corpus=GenerationCorpus(tmp_path)
    cfg=dict(seed=17,sampler_steps=50,timestep_shift=3.)
    pairs=CorpusPairs(corpus,cfg);sampler=pairs.sampler()
    first=sampler.take(2)
    assert {r['sample_id'] for r in first}=={'0','1'}
    state=sampler.state_dict();expected=sampler.take(6)
    resumed=pairs.sampler();resumed.load_state_dict(state)
    assert resumed.take(6)==expected
    val=corpus_validation(corpus,2)
    assert {r['sample_id'] for r in val}=={'2','3'}
    assert not {r['family_id'] for r in first}&{r['family_id'] for r in val}
    assert all(r['flow_supervision'] and r['task']=='generation' for r in first)
    corpus.close()


def test_stage_a_checkpointed_loss_matches_eager(stage_student,visual_runtime):
    student,_=stage_student
    _,row=visual_runtime
    row=dict(row,caption='A red token. A blue token.')
    state,split=training_state(student,row,.9,3,17)
    with torch.no_grad():
        student.bridge.strength.fill_(.2)
        student.bridge.memory_gates.fill_(.3)
        student.bridge.vel_scale.fill_(.7)
        student.runtime.decoder.layers[0].self_attn.q_proj.weight.add_(.1)
    gradients=[];outputs=[]
    for enabled in (False,True):
        student.config['activation_checkpointing']=enabled
        student.writer.activation_checkpointing=enabled
        for layer in student.runtime.decoder.layers: layer._und_checkpointing=enabled
        student.zero_grad(set_to_none=True)
        output=student(state)
        total,*_=losses(student,state,output,split,shuffled_error=torch.tensor(1.))
        total.backward()
        outputs.append(total.detach());gradients.append([p.grad.clone() for p in student.parameters()])
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_stage_a_validation_uses_gap_and_memory_controls(stage_student,visual_runtime,tmp_path):
    from qwen_latent_cot.interaction.training import Trainer
    student,_=stage_student
    runtime,row=visual_runtime
    runtime.config['sampler_steps']=3
    row=dict(row,caption='A red token. A blue token.',sample_id='a',family_id='a')
    donor=dict(row,caption='A green token. A white token.',sample_id='b',family_id='b')
    student.config.update(val_images=1)
    trainer=Trainer(student,[row,donor],[row,donor],tmp_path/'val','stage_a',{'train':'tiny'})
    trainer.validate(0)
    path=tmp_path/'val/validation_000000'
    record=json.loads((path/'records.json').read_text())[0]
    assert record['partial_prompt']!=record['caption'] and record['removed_clauses']
    assert record['donor_sample_id']=='b'
    assert record['semantic_ce']>0 and record['off_flow']>0 and record['shuffled_flow']>0
    assert (path/'000_native.png').is_file() and (path/'000_loop.png').is_file()


def test_zero_initialization_matches_native_pixels_with_changed_parent(stage_student):
    from qwen_latent_cot.interaction.sampling import sample
    student, _ = stage_student
    runtime = student.runtime
    runtime.config['sampler_steps'] = 5
    # Emulate a trained UND parent that differs from original BAGEL.
    with torch.no_grad():
        runtime.decoder.layers[0].self_attn.q_proj.weight.add_(.3)
        for update in student.adapters.consumer.values():
            update.b.fill_(.2)
    native_image, _ = sample(student, 'a red token', 17, native=True)
    for updates in (1, 2, 3):
        image, _ = sample(student, 'a red token', 17, feedback_updates=updates)
        assert image.tobytes() == native_image.tobytes()
    assert student.bridge.strength.item() == 0
    assert torch.equal(student.bridge.vel_scale, torch.ones_like(student.bridge.vel_scale))
    assert not student.bridge.memory_gates.count_nonzero()


def test_memory_query_adapter_never_changes_native_attention(stage_student):
    student, state = stage_student
    # Gamma=1 exposes raw velocity, so zero gamma cannot conceal a broken native branch.
    with torch.no_grad():
        student.bridge.vel_scale.fill_(1)
        for update in student.adapters.consumer.values():
            update.b.fill_(.5)
        base = conditional_gen(student.runtime, state, capture=False, native_scope=student.native_context)
        result = student(state)
        torch.testing.assert_close(result.velocity, base.velocity, atol=0, rtol=0)
        student.bridge.memory_gates.fill_(.4)
        active = student(state)
        assert not torch.equal(active.velocity, base.velocity)
        masked = student(state, memory_masked=True)
        torch.testing.assert_close(masked.velocity, base.velocity, atol=0, rtol=0)
    assert all(name.endswith('q_proj') for name in student.adapters.consumer)
    for layer in student.runtime.decoder.layers:
        assert not hasattr(layer.self_attn.k_proj_moe_gen, 'update')
        assert not hasattr(layer.self_attn.v_proj_moe_gen, 'update')
        assert not hasattr(layer.self_attn.o_proj_moe_gen, 'update')


def test_predictive_canvas_and_velocity_baseline_follow_previous_round(stage_student, monkeypatch):
    import qwen_latent_cot.interaction.loop as loop
    student, state = stage_student
    with torch.no_grad():
        student.bridge.strength.fill_(.2)
        student.bridge.memory_gates.fill_(.3)
        student.bridge.vel_scale.fill_(.8)
    canvases, bases = [], []
    original_canvas, original_gen = loop.canvas_entrance, loop.conditional_gen
    def canvas(runtime, current, velocity):
        canvases.append(velocity)
        return original_canvas(runtime, current, velocity)
    def gen(*args, **kwargs):
        if kwargs.get('velocity_baseline') is not None:
            bases.append(kwargs['velocity_baseline'])
        return original_gen(*args, **kwargs)
    monkeypatch.setattr(loop, 'canvas_entrance', canvas)
    monkeypatch.setattr(loop, 'conditional_gen', gen)
    with torch.no_grad():
        output = student(state, feedback_updates=3)
    for index, previous in enumerate([output.baseline, *output.exits[:-1]]):
        assert canvases[index] is previous.velocity and bases[index] is previous.velocity
        assert previous.source[0].positions == output.baseline.source[0].positions
    # Canvas carries no final hidden state, time embedding, or cross-round gradient.
    velocity = output.baseline.velocity.detach().clone().requires_grad_()
    actual = original_canvas(student.runtime, state, velocity)
    from qwen_latent_cot.diagnostics.numerics import autocast_for
    with autocast_for(student.runtime.device):
        expected = student.runtime.model.vae2llm((state.x_t.float()-state.t*velocity.float()).detach().to(state.x_t.dtype))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not actual.requires_grad


def test_teacher_uses_same_state_and_frozen_full_prompt(stage_student, visual_runtime):
    student, _ = stage_student
    _, row = visual_runtime
    state, split = training_state(student, dict(row, caption='A red token. A blue token.'), .9, 3, 17)
    from dataclasses import replace
    with student.native_context():
        flow, lengths, positions = student.runtime.flow_inputs(split.full)
    teacher = conditional_gen(student.runtime, replace(state, flow=flow, lengths=tuple(lengths), positions=positions),
                              capture=False, native_scope=student.native_context).velocity
    eta = student.config['teacher_mix']
    torch.testing.assert_close(state.correction_target, eta*teacher.float()+(1-eta)*state.flow_target(), atol=0, rtol=0)
    assert not state.correction_target.requires_grad
    with pytest.raises(ValueError):
        from qwen_latent_cot.interaction import checkpoint
        checkpoint.apply({'kind': 'bagel_full_loop_stage_a_v4'}, student)


def test_residual_loss_amplitude_direction_and_zero_target():
    from qwen_latent_cot.interaction.stage_a import residual_correction_loss
    base = torch.zeros(2, 2, requires_grad=True)
    target = torch.ones(2, 2, requires_grad=True)
    good = torch.ones(2, 2, requires_grad=True)
    loss, mse, cosine = residual_correction_loss(good, base, target)
    torch.testing.assert_close(loss, torch.tensor(0.))
    wrong, _, wrong_cos = residual_correction_loss(-good, base, target)
    assert wrong > loss and wrong_cos == 2
    loss.backward()
    assert base.grad is None and target.grad is None and torch.isfinite(good.grad).all()
    zero, _, direction = residual_correction_loss(torch.zeros(2, 2), base, base.detach())
    assert zero == 0 and direction == 0


def test_fast_gate_scale_and_cosine_zero_residual_start(stage_student):
    student, _ = stage_student
    bridge = student.bridge
    anchor = torch.zeros(2, bridge.projection.in_features)
    canvas = torch.ones_like(anchor)
    with torch.no_grad():
        bridge.strength.fill_(.02)
    torch.testing.assert_close(bridge(anchor, canvas), torch.full_like(anchor, torch.tanh(torch.tensor(.2))))
    from qwen_latent_cot.interaction.stage_a import residual_correction_loss
    prediction = torch.zeros(2, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    loss, mse, direction = residual_correction_loss(prediction, torch.zeros_like(prediction), target)
    assert direction.item() == 0
    torch.testing.assert_close(loss, mse)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, -.5))
