"""Real tiny BAGEL MoT blocks: recurrent state, full-depth KV and deployment."""
import json
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from test_memory_topology import lm, runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint
from qwen_latent_cot.diagnostics.flow_training import conditional_gen, flow_mse
from qwen_latent_cot.diagnostics.record import FrozenGuard
from qwen_latent_cot.interaction.student import InteractionStudent
from qwen_latent_cot.interaction.loop import active_updates, training_times, supervised_loss, image_entrance
from qwen_latent_cot.interaction import checkpoint


@pytest.fixture
def full_student(visual_runtime, tmp_path):
    runtime, row = visual_runtime
    original = VisualStudent(runtime)
    path = tmp_path/'parent.pt'
    save_visual_checkpoint(path, original, 1000, 'C2-observe')
    parent = torch.load(path, weights_only=True)
    cfg = json.loads(Path('configs/interaction_sft.json').read_text())
    cfg.update(writer_rank=4, consumer_rank=4)
    student = InteractionStudent(runtime, parent, cfg)
    return student, runtime.training_state(row, .9, 3)


def test_full_depth_reentry_positions_and_cache(full_student):
    student, state = full_student
    runtime = student.runtime
    entrances = [[] for _ in runtime.decoder.layers]
    hooks = [layer.register_forward_pre_hook(
        lambda m, a, kw, i=i: entrances[i].append(kw['packed_query_sequence'].detach().clone())
        if kw.get('mode') == 'gen' and kw.get('past_key_values') is state.flow['past_key_values'] else None,
        with_kwargs=True) for i, layer in enumerate(runtime.decoder.layers)]
    before = {i: (k.clone(), state.flow['past_key_values'].value_cache[i].clone())
              for i, k in state.flow['past_key_values'].key_cache.items()}
    try:
        result = student(state)
    finally:
        for h in hooks: h.remove()
    assert all(len(values) == 3 for values in entrances)
    indexes = state.flow['packed_vae_token_indexes']
    anchor = image_entrance(runtime, state)
    from qwen_latent_cot.diagnostics.numerics import autocast_for
    with autocast_for(runtime.device):
        expected = student.bridge(anchor, result.baseline.hidden)
    torch.testing.assert_close(entrances[0][1][indexes], expected.to(entrances[0][1].dtype), rtol=0, atol=0)
    assert not torch.equal(entrances[0][0][indexes], entrances[0][1][indexes])
    texts = state.flow['packed_text_indexes']
    torch.testing.assert_close(entrances[0][0][texts], entrances[0][1][texts], rtol=0, atol=0)
    assert result.exits[0].source[0].positions[0] == result.baseline.source[0].positions[0]+16
    assert len(result.banks) == 2 and all(len(bank.layers) == 3 for bank in result.banks)
    for i, (k,v) in before.items():
        torch.testing.assert_close(state.flow['past_key_values'].key_cache[i], k, rtol=0, atol=0)
        torch.testing.assert_close(state.flow['past_key_values'].value_cache[i], v, rtol=0, atol=0)


def test_gradients_cross_round_and_frozen_reader(full_student):
    student, state = full_student
    guard = FrozenGuard(student.runtime.model, student.runtime.bundle.vae)
    result = student(state)
    previous = result.exits[0].hidden
    previous.retain_grad()
    flow_mse(result.exits[-1], state).backward()
    assert previous.grad is not None and previous.grad.abs().sum() > 0
    for name, parameters in student.parameter_groups().items():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), name
        assert sum(float(p.grad.abs().sum()) for p in parameters) > 0, name
    assert all(update.b.grad.abs().sum()>0 for name,update in student.adapters.consumer.items() if name.endswith('v_proj'))
    guard.check()
    assert all(not scope.get() for scope in student.adapters.scopes.values())


def test_writer_adaptation_does_not_modify_c2_reader(full_student):
    student, state = full_student
    decoder = student.runtime.decoder
    source = student(state).baseline.source
    seen = []
    # und_block is functional. Observe its hidden outputs directly, not layer hooks.
    import qwen_latent_cot.diagnostics.memory as memory
    original = memory.und_block
    def read(*args, **kwargs):
        value = original(*args, **kwargs)
        seen.append(value.detach().clone())
        return value
    from unittest.mock import patch
    student.writer.activation_checkpointing = False
    with patch.object(memory, 'und_block', read), torch.no_grad():
        first = student.write(source, state.t, prompt=state.prompt_kv)
        for update in student.adapters.writer.values(): update.b.add_(.1)
        second = student.write(source, state.t, prompt=state.prompt_kv)
    for before, after in zip(seen[:3], seen[3:]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.equal(first.layers[0].values, second.layers[0].values)
    assert len(student.adapters.writer) == 6 and len(student.adapters.consumer) == 9


def test_checkpoint_recomputation_matches_eager(full_student):
    student, state = full_student
    outputs, grads = [], []
    for enabled in (False, True):
        student.config['activation_checkpointing'] = enabled
        student.writer.activation_checkpointing = enabled
        student.zero_grad(set_to_none=True)
        result = student(state)
        terms = supervised_loss(result, state, student.config['round_loss_weights'])
        expected = sum(w*(f+.02*k) for w,f,k in zip(student.config['round_loss_weights'],terms.round_weighted_flow,terms.round_keep))
        torch.testing.assert_close(terms.total, expected)
        terms.total.backward()
        outputs.append(result.velocity.detach())
        grads.append([p.grad.clone() for p in student.parameters()])
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for a,b in zip(*grads): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_only_conditional_branch_receives_loop(full_student):
    student, state = full_student
    cache = state.flow['cfg_text_past_key_values']
    references = []
    layer = student.runtime.decoder.layers[0]
    handle = layer.register_forward_pre_hook(
        lambda m,a,kw: references.append((kw['packed_query_sequence'].clone(),kw['packed_query_position_embeddings']))
        if kw.get('past_key_values') is cache else None, with_kwargs=True)
    try:
        with torch.no_grad():
            conditional_gen(student.runtime, state, guidance=True)
            student(state, guidance=True)
    finally: handle.remove()
    assert len(references) == 2
    for hidden, rope in references[1:]:
        torch.testing.assert_close(hidden, references[0][0], rtol=0, atol=0)
        for a,b in zip(rope, references[0][1]): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_half_schedule_and_shared_sampling(full_student, monkeypatch):
    from qwen_latent_cot.interaction.sampling import sample
    from qwen_latent_cot.diagnostics.schedule import native_times
    student, state = full_student
    runtime = student.runtime
    for nodes, expected in ((50,25),(51,25),(6,3),(5,2)):
        cfg = dict(runtime.config,sampler_steps=nodes,timestep_shift=3)
        assert active_updates(cfg) == expected
        assert training_times(cfg) == native_times(cfg)[:expected]
    runtime.config['sampler_steps'] = 5
    monkeypatch.setattr(runtime, 'image', lambda x: x)
    x, records = sample(student,'a token',seed=17)
    assert [r['loop_active'] for r in records] == [True,True,False,False]
    assert [r['conditional_gen_layer_calls'] for r in records] == [9,9,3,3]
    flow,lengths,positions = runtime.flow_inputs('a token')
    initial = torch.randn(x.shape,generator=torch.Generator().manual_seed(17))
    nodes = native_times(runtime.config)
    direct_state = replace(state, x_t=initial, t=nodes[0], flow=flow, lengths=lengths, positions=positions, prompt_kv=None)
    with torch.no_grad(): first = student(direct_state, guidance=True).velocity
    original = student.forward
    seen = []
    def forward(current, **kwargs):
        result = original(current,**kwargs)
        seen.append(result.velocity.detach())
        return result
    monkeypatch.setattr(student, 'forward', forward)
    sample(student,'a token',seed=17)
    torch.testing.assert_close(first, seen[0], rtol=0, atol=0)


def test_checkpoint_reload_resume_and_reject_old(full_student,tmp_path):
    from qwen_latent_cot.interaction.data import PairSampler
    student,state = full_student
    optimizer = torch.optim.AdamW(student.optimizer_groups())
    terms = supervised_loss(student(state),state,student.config['round_loss_weights'])
    terms.total.backward();optimizer.step()
    expected = student(state).velocity.detach()
    sampler = PairSampler(list(range(7)),5);sampler.take(3)
    path=tmp_path/'step_000001.pt'
    checkpoint.save(path,student,'sft',1,optimizer=optimizer,sampler=sampler,provenance={'train':'tiny'})
    saved=checkpoint.read(path)
    draws,noise = sampler.take(7),torch.randn(3)
    with torch.no_grad():
        for p in student.parameters(): p.zero_()
    checkpoint.apply(saved,student)
    torch.testing.assert_close(student(state).velocity,expected,rtol=0,atol=0)
    resumed = PairSampler(list(range(7)),999)
    opt = torch.optim.AdamW(student.optimizer_groups())
    assert checkpoint.resume(saved,student,opt,resumed,{'train':'tiny'}) == 1
    assert resumed.take(7) == draws
    torch.testing.assert_close(torch.randn(3),noise,rtol=0,atol=0)
    assert saved['bridge'] and saved['kind'] == 'bagel_full_loop_v3'
    for kind in ('interaction_lora_v1','bagel_full_loop_v2'):
        with pytest.raises(ValueError): checkpoint.apply(dict(kind=kind),student)


def test_last_round_depends_on_previous_hidden(full_student):
    student,state = full_student
    with torch.no_grad():
        original=student(state).velocity
        handle=student.bridge.register_forward_hook(lambda m,a,out: a[0])
        try: no_reentry=student(state).velocity
        finally: handle.remove()
    assert not torch.equal(original,no_reentry)


def test_zero_bridge_reproduces_full_strength_memory_path(full_student):
    student,state = full_student
    student.config.update(feedback_updates=1, round_loss_weights=[1.0])
    with torch.no_grad():
        student.bridge.strength.zero_()
        base=conditional_gen(student.runtime,state)
        bank=student.write(base.source,state.t,prompt=state.prompt_kv)
        expected=conditional_gen(student.runtime,state,bank,capture=False)
        actual=student(state)
    torch.testing.assert_close(actual.velocity,expected.velocity,rtol=0,atol=0)


def test_one_step_training_saves_complete_new_state(full_student,visual_runtime,tmp_path):
    from qwen_latent_cot.interaction.training import Trainer,train_sft
    student,state = full_student
    runtime=student.runtime
    _, row = visual_runtime
    student.config.update(steps=1,global_batch=1,val_images=0,save_steps=[1])
    trainer=Trainer(student,[row],[],tmp_path/'train','sft',{'train':'tiny'})
    train_sft(trainer)
    saved=checkpoint.read(tmp_path/'train/step_000001.pt')
    assert saved['step']==1 and saved['bridge'] and saved['optimizer']
    assert saved['sampler'] and len(saved['rng_by_rank'])==1
    record=json.loads((tmp_path/'train/train.jsonl').read_text())
    assert len(record['round_flow'])==2 and record['gradient_bridge']>0


def test_sampling_cli_supports_explicit_training_free_initialization():
    from qwen_latent_cot.interaction.cli import parser
    args=parser().parse_args(['sample','--model','/model','--parent','/observe.pt',
                              '--caption','a red token','--output','/sample.png'])
    assert args.parent=='/observe.pt' and args.checkpoint is None
    with pytest.raises(SystemExit):
        parser().parse_args(['sample','--model','/model','--parent','/observe.pt',
                             '--checkpoint','/loop.pt','--caption','x','--output','/x.png'])


def test_global_bridge_preserves_spatial_energy_and_starts_at_point15():
    from qwen_latent_cot.interaction.loop import ReentryBridge
    bridge=ReentryBridge(2,torch.device('cpu'))
    anchor=torch.ones(2,2)
    previous=torch.tensor([[1.,2.],[10.,20.]],requires_grad=True)
    residual=bridge(anchor,previous)-anchor
    assert float(bridge.strength.detach())==pytest.approx(.15)
    assert not hasattr(bridge,'norm')
    torch.testing.assert_close(residual[1],10*residual[0])
    torch.testing.assert_close(residual.square().mean().sqrt(),torch.tensor(.15))
    bridge(anchor,previous).sum().backward()
    assert previous.grad is not None and torch.isfinite(previous.grad).all()
    assert torch.isfinite(bridge(anchor,torch.zeros_like(previous))).all()


def test_hard_patch_and_keep_loss_are_detached_and_exact():
    from types import SimpleNamespace
    from qwen_latent_cot.interaction.loop import LoopOutput,patch_weights
    from qwen_latent_cot.diagnostics.flow_training import FlowOutput
    baseline=torch.tensor([[0.],[3.]],requires_grad=True)
    target=torch.zeros(2,1,requires_grad=True)
    velocity=torch.tensor([[1.],[2.]],requires_grad=True)
    hard,keep=patch_weights(baseline,target,1.)
    torch.testing.assert_close(hard,torch.tensor([.5,1.5]))
    torch.testing.assert_close(keep,torch.tensor([1.,1/3]))
    assert not hard.requires_grad and not keep.requires_grad
    output=LoopOutput(FlowOutput(baseline,(),0),[FlowOutput(velocity,(),1)],[])
    state=SimpleNamespace(flow_target=lambda:target)
    terms=supervised_loss(output,state,[1.])
    torch.testing.assert_close(terms.round_flow[0],torch.tensor(2.5))
    torch.testing.assert_close(terms.round_weighted_flow[0],torch.tensor(3.25))
    torch.testing.assert_close(terms.round_keep[0],torch.tensor(2/3))
    torch.testing.assert_close(terms.total,torch.tensor(3.25+.02*2/3))
    terms.total.backward()
    assert baseline.grad is None and target.grad is None
    assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum()>0
    plain=supervised_loss(output,state,[1.],hard_patch_alpha=0,keep_weight=0)
    torch.testing.assert_close(plain.total,plain.round_flow[0])
    hard,keep=patch_weights(torch.zeros(2,1),torch.zeros(2,1),1.)
    torch.testing.assert_close(hard,torch.ones(2))
    torch.testing.assert_close(keep,torch.ones(2))
