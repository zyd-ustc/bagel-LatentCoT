"""Real tiny MoT: layout, frozen GEN input gradients, actual-state labels and loop cache."""
import json
from dataclasses import replace
import pytest
import torch
from PIL import Image
from test_memory_topology import lm, runtime, setup_writer
from qwen_latent_cot.diagnostics.memory import MemoryWriter
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.flow_training import conditional_gen, flow_mse
from qwen_latent_cot.diagnostics.loop_training import memory_loop
from qwen_latent_cot.diagnostics.data import load_dataset, tensor_hash, visual_label
from qwen_latent_cot.diagnostics.record import FrozenGuard


@pytest.fixture
def visual_runtime(runtime, tmp_path):
    runtime.writer = MemoryWriter(runtime.decoder, 'C', 16)
    native = json.loads(__import__('pathlib').Path('configs/visual_memory_pilot.json').read_text())
    runtime.config.update(und_train=native['und_train'],source_layers=[0,1,2],writer_architecture='native_und_v1')
    runtime.config.update(response_layers=[0, 2],
                          response_position_reserve=16, response_query_count=3,
                          response_energy_floor=1e-8, cross_round_gradients=False,
                          label_instructions={'observe': runtime.config['caption_instruction'], 'feedback':'State the correction.', 'target_clean': runtime.config['caption_instruction'],
                          'observation_proxy': 'Describe visible evidence. Say unknown.',
                          'target_observation_gap': 'Describe what the target lacks in the evidence.'})
    image = tmp_path/'visual.png'; Image.new('RGB', (32, 32), 'red').save(image)
    row = visual_label(dict(sample_id='a', scene_id='a', split='train', caption='a red token',
                            facts=[], verified=True, image=str(image),task='generation',flow_supervision=True,annotation={'origin':'source_annotation','scope':'endpoint'}))
    return runtime, row




def test_source_caption_review_preserves_recipient_state_and_swaps_only_source(visual_runtime, tmp_path):
    from qwen_latent_cot.diagnostics.source_review import source_review
    runtime, row = visual_runtime
    runtime.config['max_caption_tokens'] = 4
    blue = tmp_path/'donor.png'; Image.new('RGB', (32,32), 'blue').save(blue)
    donor = dict(row, sample_id='b', scene_id='b', family_id='b', image=str(blue),
                 caption='a blue token', target_caption='a blue token', swap_id='a')
    row = dict(row, swap_id='b')
    student = VisualStudent(runtime)
    result = source_review(student, [(row,.75,0),(donor,.75,0)], [row,donor], tmp_path/'review')
    by = {(r['sample_id'],r['arm']):r for r in result}
    assert len(result)==6
    assert by['a','Correct']['state']==by['a','Source-swap']['state']
    assert by['a','Source-swap']['source_state']==by['b','Correct']['source_state']
    assert by['a','Correct']['state']['x_t_hash']!=by['a','Source-swap']['source_state']['x_t_hash']
    assert by['a','Source-swap']['caption']['text']==by['b','Correct']['caption']['text']
    assert by['a','Source-off']['caption']['text']==by['b','Source-off']['caption']['text']
    assert all(not r['gen_memory_injection'] for r in result)
    cost=json.loads((tmp_path/'review/execution.json').read_text())
    assert cost['gen_entries']==4


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
    with pytest.raises(ValueError, match='t/noise|x_t'):
        runtime.training_state(observed, .9, 5)
    observed['observation_state_hash'] = '0'*64
    with pytest.raises(ValueError, match='another x_t'):
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






def test_complete_sampling_updates_once_and_resets_seed(visual_runtime):
    from qwen_latent_cot.diagnostics.visual_evaluation import complete_sampling
    runtime, row = visual_runtime; student = VisualStudent(runtime)
    a, records = complete_sampling(student, row, 3, 2, 'fixed')
    b, other = complete_sampling(student, row, 3, 1, 'new')
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(records) == runtime.config['sampler_steps']-1
    assert all(r['sampler_updates'] == 1 for r in records)
    assert sum(r['gen_calls_including_cfg'] for r in records) > sum(r['gen_calls_including_cfg'] for r in other)


def test_layer_and_final_norm_hooks_are_read_only(visual_runtime):
    from qwen_latent_cot.diagnostics.visual_evaluation import observe_layers
    runtime, row = visual_runtime
    state = runtime.training_state(row, .75, 4)
    with torch.no_grad():
        expected = conditional_gen(runtime, state, capture=False)
        with observe_layers(runtime, state.flow['past_key_values']) as tensors:
            actual = conditional_gen(runtime, state, capture=False)
    torch.testing.assert_close(actual.velocity, expected.velocity, rtol=0, atol=0)
    assert set(tensors) == {0,1,2,'final_head'}
    assert set(tensors[0]) == {'hidden_before_norm','hidden_after_norm','attention_output'}
    assert not tensors['final_head']['normalized_hidden'].requires_grad




def test_weak_teacher_response_is_reported_and_filtered(visual_runtime):
    from qwen_latent_cot.diagnostics.visual_objectives import state_probe, bank_response
    runtime, row = visual_runtime; student = VisualStudent(runtime,runtime.writer)
    runtime.config['response_filter_weak_teacher'] = True
    runtime.config['response_condition_energy_min'] = 1e10  # lock out both layers
    state = runtime.training_state(row, .9, 4)
    with torch.no_grad(): source = conditional_gen(runtime, state).source
    bank = student.write(source, .9); teacher = student.teacher_bank(row['target_caption'])
    loss, metrics = bank_response(runtime, bank, teacher, state_probe(runtime,state))
    assert float(loss.detach()) == 0 and all(r['weak_teacher_filtered'] for r in metrics)
    assert all('condition_energy' in r for r in metrics)


def test_R2_controls_keep_original_prompt_and_use_actual_sampler_nodes(visual_runtime, tmp_path):
    from qwen_latent_cot.diagnostics.visual_evaluation import evaluate
    runtime, row = visual_runtime;student = VisualStudent(runtime,runtime.writer)
    runtime.config['eval_t']=[.75];runtime.config['noise_seeds']=[0]
    other_image=tmp_path/'blue.png';Image.new('RGB',(32,32),'blue').save(other_image)
    a=dict(row,split='test',swap_id='b')
    b=dict(row,sample_id='b',scene_id='b',family_id='b',split='test',image=str(other_image),swap_id='a')
    cache,lens,positions=runtime.prompt_context(runtime.config['neutral_prompt'])
    reference=[cache.key_cache[i].clone() for i in range(3)]
    out=tmp_path/'controls';out.mkdir()
    records=evaluate(student,[a,b],out,rounds=2,save_images=True,continue_sampling=True,layer_diagnostics=True)
    arms={r['arm'] for r in records}
    assert {'R0','R1','R2-new','R2-old','R2-fixed','R2-swap','Source-swap',
            'Target-text-oracle','Target-C1-oracle','G-fixed','G-neutral','G-layout'} <= arms
    assert all(r['requested_t']==.75 and r['t']!=.75 for r in records)
    assert all(r['image_policy']=='single_native_node_injection_then_native_suffix' for r in records)
    assert all((out/r['layer_tensors']).exists() for r in records)
    for i,k in enumerate(reference):torch.testing.assert_close(k,cache.key_cache[i],rtol=0,atol=0)
    assert (out/'blind_mapping.json').exists()


def test_observe_evaluation_reports_captions_without_flow_supervision(visual_runtime,tmp_path):
    from qwen_latent_cot.diagnostics.visual_evaluation import evaluate
    runtime,row=visual_runtime
    runtime.config.update(stage='C2-observe',eval_t=[.2],noise_seeds=[0],source_layers=[0,1,2])
    row=dict(row,task='observe',flow_supervision=False)
    student=VisualStudent(runtime)
    records=evaluate(student,[row],tmp_path/'observe')
    assert {'R0','R1','Source-off'} <= {r['arm'] for r in records}
    assert all(r['conditional_flow_mse'] is None for r in records)
    assert all(torch.isfinite(torch.tensor(r['caption']['nll'])) for r in records)
    assert (tmp_path/'observe/summary.json').exists()
