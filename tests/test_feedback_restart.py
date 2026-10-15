"""Same-noise complete restart and fixed-memory early windows on native BAGEL."""
import json
import pytest
import torch
from test_memory_topology import lm, runtime
from test_explicit_feedback import continuous
from qwen_latent_cot.diagnostics.explicit_feedback import Feedback
from qwen_latent_cot.diagnostics.feedback_restart import (PROTOCOL, initial_noise, generate_base,
    sample_fixed, compare_restart, save_restart, validate_windows)


def test_initial_noise_and_native_baseline_match_shallow_sampler(runtime, monkeypatch):
    from qwen_latent_cot.diagnostics.shallow_sampling import sample
    from types import SimpleNamespace
    continuous(runtime)
    prepared = generate_base(runtime, 'target', 7)
    monkeypatch.setattr(runtime, 'image', lambda latent: latent.clone())
    shallow_latent, _ = sample(SimpleNamespace(runtime=runtime), 'target', 7, passes=1, early=0)
    torch.testing.assert_close(shallow_latent, prepared['base']['latent'], rtol=0, atol=0)
    torch.manual_seed(42); torch.randn(100)
    again = initial_noise(runtime, prepared['flow'], 7)
    torch.testing.assert_close(again, prepared['initial'], rtol=0, atol=0)
    assert len(prepared['base']['nodes']) == runtime.config['sampler_steps']-1


def test_fixed_bank_written_once_and_all_generations_restart_identically(runtime, monkeypatch, tmp_path):
    import qwen_latent_cot.diagnostics.feedback_restart as restart
    continuous(runtime)
    prepared = generate_base(runtime, 'target', 0)
    original = prepared['initial'].clone()
    writer = restart.write_feedback; writes = []; memory_calls = []; bank_snapshot = []
    def write(*args, **kwargs):
        bank = writer(*args, **kwargs); writes.append(bank)
        bank_snapshot.extend((kv.keys.clone(),kv.values.clone()) for kv in bank.layers)
        return bank
    monkeypatch.setattr(restart, 'write_feedback', write)
    gen = runtime.gen_arm
    def consume(x, t, flow, bank, lengths, positions, masked=False):
        memory_calls.append((bank, t, x.clone(), masked))
        return gen(x, t, flow, bank, lengths, positions, masked)
    monkeypatch.setattr(runtime, 'gen_arm', consume)
    entries = []; layer_counts = [0]*len(runtime.decoder.layers)
    def hook(index):
        def capture(module, args, kwargs):
            if kwargs.get('mode') == 'gen' and kwargs['past_key_values'] is prepared['flow']['past_key_values']:
                layer_counts[index] += 1
                if index == 0:
                    entries.append(kwargs['packed_query_sequence'].clone())
        return capture
    handles = [layer.register_forward_pre_hook(hook(i), with_kwargs=True) for i,layer in enumerate(runtime.decoder.layers)]
    try:
        result = compare_restart(runtime, prepared, early=[1,2], feedback=Feedback('one red cup','color/count','two blue cups'))
    finally:
        for h in handles: h.remove()
    assert len(writes) == 1
    assert len(memory_calls) == 6  # visible + masked: (1+2)*2
    assert all(item[0] is writes[0] for item in memory_calls)
    for kv,(keys,values) in zip(writes[0].layers,bank_snapshot):
        torch.testing.assert_close(kv.keys,keys,rtol=0,atol=0)
        torch.testing.assert_close(kv.values,values,rtol=0,atol=0)
    assert layer_counts == [12]*3  # 4 arms * 3 steps, no extra shallow passes
    starts = [x for _,t,x,_ in memory_calls if t == 1]
    assert len(starts) == 4
    for x in starts: torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(prepared['initial'], original, rtol=0, atol=0)
    for window in (1,2):
        arm = result['arms'][f'early{window}_Text-memory']
        assert arm['active_updates'] == window
        assert [n['memory_visible'] for n in arm['nodes']] == [i < window for i in range(3)]
        assert all(n['conditional_gen_passes'] == 1 for n in arm['nodes'])
    assert len({arm['initial_noise_hash'] for arm in result['arms'].values()}) == 1
    record = save_restart(tmp_path, 'sample_bagel_auto', result, dict(sample_id='sample',prompt='target',noise_seed=0))
    assert record['shallow_loop'] is False and record['memory_refresh'] is False
    assert record['final_image_diagnostics']['early1_Text-memory']['reference'] == 'early1_Layout-memory'
    saved = torch.load(tmp_path/record['memory_file'], weights_only=True)
    assert saved['slots'] == 16 and all(not kv['rotated'] for kv in saved['layers'])


def test_reflection_occurs_after_complete_generation_before_writer(runtime, monkeypatch):
    import qwen_latent_cot.diagnostics.feedback_restart as restart
    continuous(runtime)
    velocity = runtime.velocity; events = []
    def run(x,t,flow):
        events.append(('gen',t)); return velocity(x,t,flow)
    monkeypatch.setattr(runtime,'velocity',run)
    prepared = generate_base(runtime,'target',0)
    def reflector(image):
        assert len(events) == 3 and image is prepared['image']
        events.append(('reflection',None))
        return Feedback('red cup','color','blue cup',origin='bagel_auto')
    write = restart.write_feedback
    def writer(*args, **kwargs):
        assert events[-1][0] == 'reflection'
        events.append(('writer',None)); return write(*args,**kwargs)
    monkeypatch.setattr(restart,'write_feedback',writer)
    compare_restart(runtime,prepared,early=[2],reflector=reflector)
    assert [event[0] for event in events].count('writer') == 1


def test_satisfied_feedback_skips_memory_and_reserved_layout(runtime, monkeypatch):
    continuous(runtime)
    prepared = generate_base(runtime,'target',0)
    def fail(*args, **kwargs): raise AssertionError('strict bypass must not consume memory')
    monkeypatch.setattr(runtime,'gen_arm',fail)
    result = compare_restart(runtime,prepared,early=[1,2],feedback=Feedback('correct','','',True))
    assert result['writer_updates'] == 0 and result['bank'] is None
    assert result['images']['early2_Text-memory'] is result['images']['Base']
    assert all(n['feedback_active'] is False for n in result['arms']['early2_Text-memory']['nodes'])


def test_windows_follow_exact_shallow_early10_20_schedule():
    from qwen_latent_cot.diagnostics.schedule import native_times
    config = dict(sampler_steps=50,timestep_shift=3)
    validate_windows(config,[10,20]); times = native_times(config)
    assert times[10] == pytest.approx(.92125976)
    assert times[20] == pytest.approx(.81308419)
    with pytest.raises(ValueError): validate_windows(config,[50])
    with pytest.raises(ValueError): validate_windows(config,[10,10])


def test_restart_cli_default_route_preparation_and_exact_replay(runtime,tmp_path,monkeypatch):
    import qwen_latent_cot.bagel.backbone as backbone
    from qwen_latent_cot.diagnostics import feedback_cli, feedback_restart_cli
    from qwen_latent_cot.diagnostics.record import save_checkpoint
    continuous(runtime)
    checkpoint=tmp_path/'c1.pt'; save_checkpoint(checkpoint,runtime,1,'C1')
    monkeypatch.setattr(backbone,'load_native',lambda *args:runtime.bundle)
    monkeypatch.setattr(feedback_restart_cli,'manifest',lambda *args,**kwargs:dict(model_files=runtime.native_files))
    settings=dict(protocol=PROTOCOL,early=[1,2],max_reflection_tokens=32,vit_max_image_size=980,vit_min_image_size=224)
    config=tmp_path/'config.json';config.write_text(json.dumps(settings))
    data=tmp_path/'data.jsonl';data.write_text(json.dumps(dict(sample_id='sample',prompt='target',noise_seed=0,t=.5))+'\n')
    prepared=tmp_path/'prepared'; compared=tmp_path/'compared'
    args=['--model',str(tmp_path),'--text-checkpoint',str(checkpoint),'--config',str(config)]
    feedback_cli.main(args+['--data',str(data),'--output',str(prepared),'--prepare-only'])
    template=prepared/'feedback_template.jsonl'; row=json.loads(template.read_text())
    assert 'state' not in row and 't' not in row and 'initial_state' in row
    row['feedback']=dict(observation='red',gap='color',correction='blue cup')
    template.write_text(json.dumps(row)+'\n')
    feedback_cli.main(args+['--data',str(template),'--output',str(compared),'--text-control'])
    a=torch.load(prepared/'sample_initial.pt',weights_only=True)
    b=torch.load(compared/'sample_initial.pt',weights_only=True)
    torch.testing.assert_close(a['x_initial'],b['x_initial'],rtol=0,atol=0)
    result=json.loads((compared/'sample_manual_checked.json').read_text())
    assert result['protocol'] == PROTOCOL and result['writer_updates'] == 1
    assert len(result['images']) == 9
    assert (compared/'review.html').exists()
    assert all(arm['initial_noise_hash'] == a['initial_noise_hash'] for arm in result['arms'].values())
    from qwen_latent_cot.diagnostics.feedback_audit import audit
    summary = audit(compared,tmp_path/'audit')
    assert summary['samples'] == 1 and summary['window_comparisons'] == 2
    audited = json.loads((tmp_path/'audit'/'image_audit.json').read_text())
    assert [record['intervention_steps'] for record in audited] == [1,2]
    assert [record['native_suffix_steps'] for record in audited] == [2,1]


def test_restart_rejects_old_midtrajectory_state(tmp_path):
    from qwen_latent_cot.diagnostics.feedback_restart_cli import load_rows
    data=tmp_path/'data.jsonl'
    data.write_text(json.dumps(dict(sample_id='sample',prompt='target',noise_seed=0,state='old.pt',t=.5)))
    with pytest.raises(ValueError,match='mid-trajectory'):load_rows(data)
