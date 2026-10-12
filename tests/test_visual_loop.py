"""Real tiny MoT: layout, frozen GEN input gradients, actual-state labels and loop cache."""
import json
from dataclasses import replace
import pytest
import torch
from PIL import Image
from test_memory_topology import lm, runtime, setup_writer
from qwen_latent_cot.diagnostics.memory import MemoryWriter
from qwen_latent_cot.diagnostics.visual import VisualStudent, VisualReadAdapter
from qwen_latent_cot.diagnostics.flow_training import conditional_gen, flow_mse
from qwen_latent_cot.diagnostics.loop_training import memory_loop
from qwen_latent_cot.diagnostics.data import load_dataset, tensor_hash, visual_label
from qwen_latent_cot.diagnostics.record import FrozenGuard


@pytest.fixture
def visual_runtime(runtime, tmp_path):
    runtime.writer = MemoryWriter(runtime.decoder, 'C', 16, 4)
    runtime.config.update(visual_rank=4, train_writer=False, response_layers=[0, 2],
                          response_position_reserve=16, response_query_count=3,
                          response_energy_floor=1e-8, cross_round_gradients=False,
                          label_instructions={'target_clean': runtime.config['caption_instruction'],
                          'observation_proxy': 'Describe visible evidence. Say unknown.',
                          'target_observation_gap': 'Describe what the target lacks in the evidence.'})
    image = tmp_path/'visual.png'; Image.new('RGB', (32, 32), 'red').save(image)
    row = visual_label(dict(sample_id='a', scene_id='a', split='train', caption='a red token',
                            facts=[], verified=True, image=str(image)))
    return runtime, row


def test_visual_zero_and_off_restore_C_writer(lm):
    writer, seeds, source, positions = setup_writer(lm, 'C')
    visual = VisualReadAdapter(lm.model, 4)
    original = writer(seeds, source, positions)
    zero = writer(seeds, source, positions, source_kind='visual', t=.9, visual_adapter=visual)
    with torch.no_grad():
        for layer in visual.layers: layer.up.weight.normal_(std=.1)
    off = writer(seeds, source, positions, source_kind='text', visual_adapter=visual)
    for a, b, c in zip(original.layers, zero.layers, off.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, c.values, rtol=0, atol=0)


def test_paired_state_and_native_conditional_parity(visual_runtime):
    runtime, row = visual_runtime
    for t in (.25, .75, .9, 1.):
        state = runtime.training_state(row, t, 3)
        native = runtime.velocity(state.x_t, t, state.flow)[0]
        output = conditional_gen(runtime, state)
        torch.testing.assert_close(output.velocity, native, rtol=0, atol=0)
        torch.testing.assert_close(state.x_t, state.epsilon if t == 1 else
                                   (1-t)*state.x0+t*state.epsilon, rtol=0, atol=0)
        torch.testing.assert_close(state.flow_target(), state.epsilon.float()-state.x0.float())
        assert len(output.source) == 3
        kv = output.source[0]
        assert kv.t == t and kv.round == 0 and kv.expert == 'mixed_und_gen'
        assert kv.token_roles.count('boundary') == 2
        assert len(kv.positions) == len(kv.keys)


@pytest.mark.parametrize('activation_checkpoint', [False, True])
def test_complete_flow_gradient_and_two_round_cache(visual_runtime, activation_checkpoint):
    runtime, row = visual_runtime
    student = VisualStudent(runtime)
    state = runtime.training_state(row, .75, 5)
    cache = state.flow['past_key_values']
    original = [(cache.key_cache[i].clone(), cache.value_cache[i].clone()) for i in range(3)]
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, student.text_teacher)
    optimizer = torch.optim.AdamW(student.visual.parameters(), lr=.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = memory_loop(student, state, 2, checkpoint_activations=activation_checkpoint)
        loss = torch.stack([flow_mse(exit, state) for exit in output.exits]).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.visual.parameters())
        assert all(layer.up.weight.grad.abs().max() > 0 for layer in student.visual.layers)
        if step == 1:
            assert all(layer.down.weight.grad.abs().max() > 0 for layer in student.visual.layers)
        assert all(p.grad is None for p in student.writer.parameters())
        optimizer.step(); guard.check()
    for i, (k, v) in enumerate(original):
        torch.testing.assert_close(cache.key_cache[i], k, rtol=0, atol=0)
        torch.testing.assert_close(cache.value_cache[i], v, rtol=0, atol=0)
    assert output.gen_calls == 3


def test_fixed_bank_R2_and_diagnostics_do_not_change_R1(visual_runtime):
    runtime, row = visual_runtime; student = VisualStudent(runtime)
    state = runtime.training_state(row, .9, 7)
    with torch.no_grad():
        a = memory_loop(student, state, 2, feedback='fixed')
        b = memory_loop(student, state, 1, diagnostics=True)
    torch.testing.assert_close(a.exits[0].velocity, a.exits[1].velocity, rtol=0, atol=0)
    torch.testing.assert_close(a.exits[0].velocity, b.exits[0].velocity, rtol=0, atol=0)
    # Sources contain the actually shifted positions of the Memory conditional call.
    assert b.exits[0].source[0].positions[0] == a.baseline.source[0].positions[0]+16


def test_observation_state_hash_is_binding(visual_runtime):
    runtime, row = visual_runtime
    state = runtime.training_state(row, .75, 5)
    observed = dict(row, label_kind='observation_proxy', t=.75, noise_seed=5,
                    observation_state_hash=tensor_hash(state.x_t), observation_image_hash=state.metadata['image_hash'])
    runtime.training_state(observed, .75, 5)
    with pytest.raises(ValueError, match='t/noise'):
        runtime.training_state(observed, .9, 5)
    observed['observation_state_hash'] = '0'*64
    with pytest.raises(ValueError, match='different x_t'):
        runtime.training_state(observed, .75, 5)
    trajectory = replace(state, x0=None, epsilon=None)
    with pytest.raises(ValueError, match='no paired'):
        trajectory.flow_target()


def test_unreviewed_proxy_never_becomes_confirmed(tmp_path):
    image = tmp_path/'image.bin'; image.write_bytes(b'image')
    row = dict(sample_id='a', scene_id='a', split='train', verified=False, label_kind='observation_proxy',
               t=.75, noise_seed=1, round=0, image='image.bin', observation_caption='unknown',
               observation_facts=[], uncertain_facts=[], observation_method='human', review_status='pending',
               observation_state_hash='0'*64, observation_image_hash='1'*64)
    path = tmp_path/'data.jsonl'; path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='unverified'):
        load_dataset(path, visual=True)
    loaded = load_dataset(path, visual=True, allow_unreviewed_proxy=True)[0]
    assert loaded['verified'] is False and loaded['confirmed_label'] is False


def test_response_teacher_uses_real_deployment_slot_layout(visual_runtime):
    from qwen_latent_cot.diagnostics.visual_objectives import state_probe, bank_response
    runtime, row = visual_runtime; student = VisualStudent(runtime)
    state = runtime.training_state(row, .9, 4)
    with torch.no_grad(): source = conditional_gen(runtime, state).source
    bank = student.write(source, .9); teacher = student.teacher_bank(row['target_caption'])
    probe = state_probe(runtime, state)
    loss, metrics = bank_response(runtime, bank, teacher, probe)
    loss.backward()
    assert all(layer.up.weight.grad.abs().max() > 0 for layer in student.visual.layers)
    assert all(p.grad is None for p in student.text_teacher.parameters())
    assert len(metrics) == 2
    runtime.config['response_position_reserve'] = 64
    with pytest.raises(ValueError, match='reserve'):
        state_probe(runtime, state)


def test_cross_round_source_gradient_is_an_explicit_policy(visual_runtime):
    runtime, row = visual_runtime; student = VisualStudent(runtime)
    state = runtime.training_state(row, .75, 4)
    stopped = memory_loop(student, state, 2)
    assert not stopped.exits[0].source[0].keys.requires_grad
    runtime.config['cross_round_gradients'] = True
    live = memory_loop(student, state, 2, checkpoint_activations=True)
    assert live.exits[0].source[-1].keys.requires_grad
    # Layer-zero source is captured before its first Memory-conditioned attention.
    assert not live.exits[0].source[0].keys.requires_grad
    loss = flow_mse(live.exits[1], state)
    loss.backward()
    assert student.visual.layers[0].up.weight.grad.abs().max() > 0


def test_joint_training_and_strict_resume(visual_runtime, tmp_path):
    from qwen_latent_cot.diagnostics.visual import train
    from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint, load_visual_checkpoint
    from qwen_latent_cot.diagnostics.data import file_hash
    runtime, row = visual_runtime
    config = json.loads(__import__('pathlib').Path('configs/visual_gen_joint.json').read_text())
    # Keep the real tiny model's existing native interface (32px, 3 layers).
    config.update({key: runtime.config[key] for key in ('resolution','rank','caption_instruction','response_layers','response_query_count','visual_rank','label_instructions')})
    config.update(steps=1, effective_batch=4, eval_every=1, max_caption_tokens=2,
                  eval_t=[.75,1.], noise_seeds=[0], replay_fraction=.25)
    runtime.config = config
    student = VisualStudent(runtime); student.text_teacher_hash = 'test'
    train_rows = [row, dict(row, sample_id='b', split='val')]
    replay = [dict(row, split=split, sample_id='text_'+split) for split in ('train','val')]
    out = tmp_path/'joint'; out.mkdir()
    train(student, train_rows, replay, 'C2-joint', out, None)
    log = json.loads((out/'training.jsonl').read_text().splitlines()[0])
    assert log['losses']['gen'] > 0 and log['group_gradient_norms']['visual'] > 0
    assert log['gen_conditional_calls'] == 10  # 3 visual*(S0+GEN+probe) + 1 replay probe
    assert (out/'last.pt').exists() and (out/'best.pt').exists()
    saved = load_visual_checkpoint(out/'last.pt', student)
    assert saved['step'] == 1 and saved['optimizer'] and saved['rng']
    assert saved['text_teacher_hash'] == 'test'
    runtime.config['steps'] = 2
    resumed = tmp_path/'resumed'; resumed.mkdir()
    train(student, train_rows, replay, 'C2-joint', resumed, str(out/'last.pt'), resume=str(out/'last.pt'), skip_validation=True)
    assert json.loads((resumed/'training.jsonl').read_text().splitlines()[0])['step'] == 2
    runtime.config['label_instructions'] = dict(runtime.config['label_instructions'], observation_proxy='changed')
    with pytest.raises(ValueError, match='interface mismatch'):
        load_visual_checkpoint(out/'last.pt', student)


def test_complete_sampling_updates_once_and_resets_seed(visual_runtime):
    from qwen_latent_cot.diagnostics.visual_evaluation import complete_sampling
    runtime, row = visual_runtime; student = VisualStudent(runtime)
    a, records = complete_sampling(student, row, 3, 2, 'fixed')
    b, other = complete_sampling(student, row, 3, 1, 'new')
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(records) == runtime.config['sampler_steps']-1
    assert all(r['sampler_updates'] == 1 for r in records)
    assert sum(r['gen_calls_including_cfg'] for r in records) > sum(r['gen_calls_including_cfg'] for r in other)
