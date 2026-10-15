"""Formal source and same-state feedback contracts on tiny native BAGEL."""
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
import torch
from test_memory_topology import lm, runtime
from qwen_latent_cot.diagnostics.feedback_source import compose_source, prompt_source, write_feedback
from qwen_latent_cot.diagnostics.explicit_feedback import (Feedback, correction_source, intervention_state,
                                                        compare, save_comparison, render_review, reflect)


def continuous(runtime):
    from qwen_latent_cot.diagnostics.memory import MemoryWriter
    runtime.writer = MemoryWriter(runtime.decoder, 'C', 16, runtime.config['rank'])
    return runtime


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


def test_preview_and_r2_share_exact_saved_state_with_native_suffix(runtime, monkeypatch, tmp_path):
    continuous(runtime)
    x, t, flow, lengths, positions, meta = intervention_state(runtime, 'target', 3, .5)
    original = x.clone(); velocity = runtime.velocity; calls = []
    def record(x, current, inputs):
        calls.append((x.clone(), current, inputs))
        return velocity(x, current, inputs)
    monkeypatch.setattr(runtime, 'velocity', record)
    feedback = Feedback('one red cup', 'wrong count and color', 'two blue cups')
    result = compare(runtime, x, t, flow, lengths, positions, feedback=feedback, source_layers=[0])
    assert set(result['images']) == {'Base','Layout-memory','Layout-text','Text','Text-memory'}
    starts = [state for state, current, _ in calls if current == t]
    assert len(starts) == 5
    for start in starts:
        torch.testing.assert_close(start, original, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert result['token_counts'] == {'memory':16,'text':6}
    assert result['preview'] is result['images']['Base']
    record = save_comparison(tmp_path, 'sample', result, dict(meta, sample_id='sample'))
    render_review(tmp_path, [record])
    assert (tmp_path/'review.html').exists()
    assert all((tmp_path/file).exists() for file in record['images'].values())


def test_satisfied_feedback_is_strict_native_bypass(runtime, monkeypatch):
    continuous(runtime)
    x, t, flow, lengths, positions, _ = intervention_state(runtime, 'target', 0, .5)
    def fail(*args, **kwargs):
        raise AssertionError('satisfied feedback must not inject any new condition')
    monkeypatch.setattr(runtime, 'gen_arm', fail)
    result = compare(runtime, x, t, flow, lengths, positions, feedback=Feedback('correct', '', '', True))
    assert result['bypass'] and not result['banks']
    assert result['images']['Text-memory'] is result['images']['Base']


def test_implicit_and_swap_are_same_layout_and_validate_donor(runtime):
    continuous(runtime)
    from qwen_latent_cot.diagnostics.visual import VisualStudent
    student = VisualStudent(runtime).eval().requires_grad_(False)
    x, t, flow, lengths, positions, meta = intervention_state(runtime, 'target', 0, .5)
    _, _, _, _, _, ometa = intervention_state(runtime, 'target', 1, .5)
    donor = dict(feedback=Feedback('three green cups', 'wrong count', 'two blue cups'), metadata=ometa)
    result = compare(runtime, x, t, flow, lengths, positions,
        feedback=Feedback('one red cup', 'wrong color', 'two blue cups'), student=student,
        source_layers=[0], donor=donor, original_prompt='target')
    assert {'Swapped-memory', 'Implicit-memory'} <= result['images'].keys()
    assert result['swap_discriminating'] is False
    bad = dict(donor, metadata=dict(ometa, prompt='different'))
    with pytest.raises(ValueError, match='same prompt'):
        compare(runtime, x, t, flow, lengths, positions, feedback=Feedback('red', 'color', 'blue'),
                source_layers=[0], donor=bad, original_prompt='target')


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
    path.write_text(json.dumps(dict(sample_id='a', prompt='target', t=.5, noise_seed=0,
        feedback=dict(observation='', gap='', correction='blue cup', origin='bagel_auto'))))
    with pytest.raises(ValueError, match='manually checked'): load_rows(path)


def test_cli_prepares_and_reuses_exact_state(runtime, tmp_path, monkeypatch):
    from qwen_latent_cot.diagnostics import feedback_cli
    from qwen_latent_cot.diagnostics.record import save_checkpoint
    import qwen_latent_cot.bagel.backbone as backbone
    continuous(runtime)
    checkpoint = tmp_path/'c1.pt'
    save_checkpoint(checkpoint, runtime, 1, 'C1')
    monkeypatch.setattr(backbone, 'load_native', lambda *args: runtime.bundle)
    monkeypatch.setattr(feedback_cli, 'manifest', lambda *args, **kw: dict(model_files=runtime.native_files))
    settings = dict(protocol='explicit_feedback_v1', preview_kind='complete', source_layers=[0],
                    max_reflection_tokens=32, vit_max_image_size=980, vit_min_image_size=224)
    config = tmp_path/'config.json'; config.write_text(json.dumps(settings))
    dataset = tmp_path/'samples.jsonl'
    dataset.write_text(json.dumps(dict(sample_id='sample', prompt='target', noise_seed=3, t=.5))+'\n')
    prepared = tmp_path/'prepared'
    base = ['--model', str(tmp_path), '--text-checkpoint', str(checkpoint), '--config', str(config)]
    feedback_cli.main(base+['--data', str(dataset), '--output', str(prepared), '--prepare-only'])
    template = prepared/'feedback_template.jsonl'
    row = json.loads(template.read_text())
    row['feedback'] = dict(observation='one red cup', gap='count/color', correction='two blue cups')
    template.write_text(json.dumps(row)+'\n')
    compared = tmp_path/'compared'
    feedback_cli.main(base+['--data', str(template), '--output', str(compared)])
    a = torch.load(prepared/'sample_state.pt', weights_only=True)
    b = torch.load(compared/'sample_state.pt', weights_only=True)
    torch.testing.assert_close(a['x_t'], b['x_t'], rtol=0, atol=0)
    assert a['t'] == b['t']
    assert json.loads((compared/'manifest.json').read_text())['status'].startswith('executed')
    assert (compared/'review.html').exists() and (compared/'annotations.csv').exists()
    diagnostic = json.loads((compared/'sample_manual_checked.json').read_text())
    replay = torch.load(compared/diagnostic['velocity_file'], weights_only=True)
    assert replay['x_t_hash'] == a['x_t_hash']
    assert set(replay['velocities']) == set(diagnostic['images'])
    from qwen_latent_cot.diagnostics.feedback_audit import audit
    summary = audit(compared, tmp_path/'audit')
    assert summary['samples'] == 1 and summary['velocity_response_records'] == 1


def test_response_diagnostics_detects_noop_and_separates_cfg_suppression():
    from qwen_latent_cot.diagnostics.explicit_feedback import response_diagnostics
    x = torch.ones(2, 4)
    base = (torch.ones_like(x), torch.ones_like(x))
    layout = (torch.full_like(x, 2), torch.full_like(x, 2))
    response = (torch.full_like(x, 3), layout[1].clone())
    result = response_diagnostics(x, .5, .4, {'Base':base, 'Layout-text':layout, 'Text':response})
    text = result['arms']['Text']
    assert text['reference'] == 'Layout-text'
    assert text['conditional']['normalized_difference'] == .5
    assert text['cfg']['normalized_difference'] == 0
    assert text['first_step_delta_l2'] == 0
    noop = response_diagnostics(x, .5, .4, {'Base':base, 'Layout-text':layout, 'Text':layout})
    assert noop['arms']['Text']['conditional']['normalized_difference'] == 0
    changed = response_diagnostics(x, .5, .4, {'Base':base, 'Layout-text':layout,
                                              'Text':(response[0], response[0])})
    assert changed['arms']['Text']['first_step_delta_l2'] == pytest.approx(.1*8**.5)


def test_image_diagnostics_uses_layout_and_reports_identical_images():
    from PIL import Image
    from qwen_latent_cot.diagnostics.explicit_feedback import image_diagnostics
    base = Image.new('RGB', (2, 2), (0, 0, 0)); layout = Image.new('RGB', (2, 2), (5, 5, 5))
    result = image_diagnostics({'Base':base, 'Layout-text':layout, 'Text':layout.copy()})
    assert result['Text']['identical'] and result['Text']['reference'] == 'Layout-text'
    assert result['Layout-text']['mae_255'] == 5
