"""Same-noise complete restart and fixed-memory early windows on native BAGEL."""
import json
import pytest
import torch
from test_memory_topology import lm, runtime
from qwen_latent_cot.diagnostics.explicit_feedback import (Feedback, PROTOCOL, initial_noise, generate_base,
    sample_fixed, compare, save_comparison, validate_windows, correction_source, reflect)
from qwen_latent_cot.diagnostics.feedback_source import compose_source, prompt_source, write_feedback
from dataclasses import replace


def continuous(runtime):
    from qwen_latent_cot.diagnostics.memory import MemoryWriter
    runtime.writer = MemoryWriter(runtime.decoder, 'C', 16)
    return runtime


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
    import qwen_latent_cot.diagnostics.explicit_feedback as restart
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
        result = compare(runtime, prepared, early=[1,2], feedback=Feedback('one red cup','color/count','two blue cups'))
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
    record = save_comparison(tmp_path, 'sample_bagel_auto', result, dict(sample_id='sample',prompt='target',noise_seed=0))
    assert record['shallow_loop'] is False and record['memory_refresh'] is False
    assert record['final_image_diagnostics']['early1_Text-memory']['reference'] == 'early1_Layout-memory'
    saved = torch.load(tmp_path/record['memory_file'], weights_only=True)
    assert saved['slots'] == 16 and all(not kv['rotated'] for kv in saved['layers'])


def test_reflection_occurs_after_complete_generation_before_writer(runtime, monkeypatch):
    import qwen_latent_cot.diagnostics.explicit_feedback as restart
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
    compare(runtime,prepared,early=[2],reflector=reflector)
    assert [event[0] for event in events].count('writer') == 1


def test_satisfied_feedback_skips_memory_and_reserved_layout(runtime, monkeypatch):
    continuous(runtime)
    prepared = generate_base(runtime,'target',0)
    def fail(*args, **kwargs): raise AssertionError('strict bypass must not consume memory')
    monkeypatch.setattr(runtime,'gen_arm',fail)
    result = compare(runtime,prepared,early=[1,2],feedback=Feedback('correct','','',True))
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


def test_cli_prepares_and_replays_exact_initial_noise(runtime,tmp_path,monkeypatch):
    import qwen_latent_cot.bagel.backbone as backbone
    from qwen_latent_cot.diagnostics import feedback_cli
    from qwen_latent_cot.diagnostics.record import save_checkpoint
    continuous(runtime)
    checkpoint=tmp_path/'c1.pt'; save_checkpoint(checkpoint,runtime,1,'C1')
    monkeypatch.setattr(backbone,'load_native',lambda *args:runtime.bundle)
    monkeypatch.setattr(feedback_cli,'manifest',lambda *args,**kwargs:dict(model_files=runtime.native_files))
    settings=dict(protocol=PROTOCOL,early=[1,2],max_reflection_tokens=32,vit_max_image_size=980,vit_min_image_size=224)
    config=tmp_path/'config.json';config.write_text(json.dumps(settings))
    data=tmp_path/'data.jsonl';data.write_text(json.dumps(dict(sample_id='sample',prompt='target',noise_seed=0))+'\n')
    prepared=tmp_path/'prepared'; compared=tmp_path/'compared'
    args=['--model',str(tmp_path),'--text-checkpoint',str(checkpoint),'--config',str(config)]
    feedback_cli.main(args+['--data',str(data),'--output',str(prepared),'--prepare-only'])
    template=prepared/'feedback_template.jsonl'; row=json.loads(template.read_text())
    assert set(row)=={'sample_id','prompt','noise_seed','initial_state','base_image','feedback'}
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


def test_composition_preserves_rotated_keys_and_packed_isolation(runtime):
    prompt = runtime.text('original prompt')[0]
    extra = runtime.text('correction')[0]
    merged = compose_source(prompt, extra, [0])
    torch.testing.assert_close(merged[0].keys[:len(prompt[0].keys)], prompt[0].keys, rtol=0, atol=0)
    torch.testing.assert_close(merged[0].keys[len(prompt[0].keys):], extra[0].keys, rtol=0, atol=0)
    assert merged[1] is prompt[1] and merged[2] is prompt[2]
    assert all(k.rotated for k in merged)
    p = replace(prompt[0], keys=prompt[0].keys[:5], values=prompt[0].values[:5], lengths=(2, 3),
                positions=(), token_roles=())
    e = replace(extra[0], keys=extra[0].keys[:5], values=extra[0].values[:5], lengths=(3, 2),
                positions=(), token_roles=())
    out = compose_source((p,), (e,))[0]
    assert out.lengths == (5, 5)
    torch.testing.assert_close(out.keys[:5], torch.cat((p.keys[:2], e.keys[:3])), rtol=0, atol=0)
    with pytest.raises(ValueError, match='RoPE'):
        compose_source(prompt, [replace(e, rotated=False)] , [0])


def test_prompt_survives_outside_source_window(runtime):
    continuous(runtime)
    p = runtime.text('original prompt')[0]; g = runtime.text('query')[0]
    bank = write_feedback(runtime, p, g, layers=[0])
    assert bank.slots == 16 and len(bank.layers) == 3
    changed = tuple(replace(kv, values=kv.values+2) if kv.layer == 1 else kv for kv in p)
    other = write_feedback(runtime, changed, g, layers=[0])
    assert not torch.equal(bank.layers[1].keys, other.layers[1].keys)
    # The common writer accepts per-layer token counts with the same sample count.
    assert runtime.memory(compose_source(p, g, [0])).slots == 16


def test_correction_extracts_only_new_native_text_and_keeps_prompt(runtime):
    flow, lengths, positions = runtime.flow_inputs('target')
    before = [k.clone() for k in flow['past_key_values'].key_cache.values()]
    p = prompt_source(flow, lengths, positions)
    c, augmented, count = correction_source(runtime, flow, lengths, positions, 'two blue cups')
    source = compose_source(p, c)
    assert count == 6 and source[0].lengths == (lengths[0]+count,)
    for index, kv in enumerate(source):
        torch.testing.assert_close(kv.keys, augmented['past_key_values'].key_cache[index], rtol=0, atol=0)
        torch.testing.assert_close(before[index], flow['past_key_values'].key_cache[index], rtol=0, atol=0)
    assert c[0].positions == tuple(range(positions[0], positions[0]+count))
    assert augmented['cfg_text_past_key_values'] is flow['cfg_text_past_key_values']


def test_feedback_rejects_invalid_automatic_output():
    text = json.dumps(dict(observation='one red cup', gap='color', correction='blue cup', satisfied=False))
    assert Feedback.parse('```json\n'+text+'\n```').origin == 'bagel_auto'
    with pytest.raises(ValueError): Feedback.parse('This is not JSON')
    with pytest.raises(ValueError): Feedback('cup', 'color', '')
    with pytest.raises(ValueError): Feedback('cup', '', '', 'true')


def test_reflection_uses_isolated_native_chat_stages(runtime, monkeypatch):
    calls = []
    raw = json.dumps(dict(observation='one red cup', gap='color', correction='blue cup', satisfied=False))
    monkeypatch.setattr(runtime.model.config, 'visual_und', True)
    def vit(lengths, positions, images, transform, token_ids):
        calls.append('vit_prepare'); return {}, [3], [1]
    def vit_update(cache, **kwargs):
        calls.append(('vit_cache', cache)); return cache
    def prompts(lengths, positions, texts, tokenizer, ids):
        assert texts[0].endswith('Original target prompt:\ntarget')
        calls.append('text_prepare'); return {}, [9], [7]
    def text_update(cache, **kwargs):
        calls.append(('text_cache', cache)); return cache
    monkeypatch.setattr(runtime.model, 'prepare_vit_images', vit)
    monkeypatch.setattr(runtime.model, 'forward_cache_update_vit', vit_update)
    monkeypatch.setattr(runtime.model, 'prepare_prompts', prompts)
    monkeypatch.setattr(runtime.model, 'forward_cache_update_text', text_update)
    monkeypatch.setattr(runtime.model, 'prepare_start_tokens', lambda *args: {})
    monkeypatch.setattr(runtime.model, 'generate_text', lambda **kw: torch.tensor([[1],[2]]))
    monkeypatch.setattr(runtime.bundle.tokenizer, 'decode', lambda *args, **kw: raw)
    result = reflect(runtime, object(), 'target', object())
    assert result.origin == 'bagel_auto' and result.correction == 'blue cup'
    assert calls[1][1] is calls[3][1]


def test_load_rows_rejects_unchecked_feedback(tmp_path):
    from qwen_latent_cot.diagnostics.feedback_cli import load_rows
    path = tmp_path/'data.jsonl'
    path.write_text(json.dumps(dict(sample_id='a', prompt='target', noise_seed=0,
        feedback=dict(observation='', gap='', correction='blue cup', origin='bagel_auto'))))
    with pytest.raises(ValueError, match='manually checked'): load_rows(path)


def test_image_diagnostics_uses_layout_and_reports_identical_images():
    from PIL import Image
    from qwen_latent_cot.diagnostics.explicit_feedback import image_diagnostics
    base = Image.new('RGB', (2, 2), (0, 0, 0)); layout = Image.new('RGB', (2, 2), (5, 5, 5))
    result = image_diagnostics({'Base':base, 'Layout-text':layout, 'Text':layout.copy()})
    assert result['Text']['identical'] and result['Text']['reference'] == 'Layout-text'
    assert result['Layout-text']['mae_255'] == 5
