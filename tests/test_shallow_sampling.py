"""Shallow passes on real native MoT blocks: counts, CFG and cache isolation."""
import pytest
import torch
from test_memory_topology import lm, runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.shallow_sampling import shallow_loop


@pytest.mark.parametrize('passes', [2, 3, 4])
def test_shallow_counts_cfg_cache_and_restoration(visual_runtime, passes):
    runtime, row = visual_runtime
    student = VisualStudent(runtime)
    state = runtime.training_state(row, .7, 0)
    decoder = runtime.decoder
    original_forward = decoder.forward_inference
    cache = state.flow['past_key_values']
    snapshot = [(cache.key_cache[i].clone(), cache.value_cache[i].clone()) for i in range(3)]
    counts = [0]*3; entries = []
    def hook(index):
        def observe(module, args, kw):
            if kw.get('mode') == 'gen':
                counts[index] += 1
                if index == 0 and kw['past_key_values'] is cache:
                    entries.append(kw['packed_query_sequence'].clone())
        return observe
    handles = [layer.register_forward_pre_hook(hook(i), with_kwargs=True) for i, layer in enumerate(decoder.layers)]
    try:
        with torch.no_grad(), shallow_loop(student, state.flow, state.lengths, state.positions, .7, passes, end_layer=2):
            conditional, velocity = runtime.velocity(state.x_t, .7, state.flow)
    finally:
        for handle in handles: handle.remove()
    cfg_calls = int(runtime.config['cfg_text_scale'] > 1)
    assert counts == [passes+cfg_calls, passes+cfg_calls, 1+cfg_calls]
    assert len(entries) == passes
    for h in entries: torch.testing.assert_close(h, entries[0], rtol=0, atol=0)
    assert torch.isfinite(velocity).all() and torch.isfinite(conditional).all()
    assert decoder.forward_inference == original_forward
    for i, (k, v) in enumerate(snapshot):
        torch.testing.assert_close(cache.key_cache[i], k, rtol=0, atol=0)
        torch.testing.assert_close(cache.value_cache[i], v, rtol=0, atol=0)


def test_base_uses_native_velocity_without_memory(visual_runtime, monkeypatch):
    import qwen_latent_cot.diagnostics.shallow_sampling as module
    runtime, row = visual_runtime
    student = VisualStudent(runtime)
    runtime.config.update(sampler_steps=4, timestep_shift=3.)
    monkeypatch.setattr(runtime, 'image', lambda x: x)
    def unexpected(*args, **kwargs):
        raise AssertionError('Base entered Memory loop')
    monkeypatch.setattr(module, 'shallow_loop', unexpected)
    image, records = module.sample(student, 'a red token', 0, 1, 0, steps=2)
    assert torch.isfinite(image).all()
    assert len(records) == 2 and all(r['und_layer_calls'] == 0 for r in records)


def test_shared_random_selection_and_portable_page(tmp_path):
    import json
    from qwen_latent_cot.diagnostics.shallow_review import prepare, render
    source = tmp_path/'prompts.jsonl'
    source.write_text('\n'.join(json.dumps(dict(prompt=f'<prompt {i}>', prompt_id=str(i))) for i in range(32)))
    output = tmp_path/'run'
    plan = prepare(source, output, 16, [2,3,4], [10,20], 123)
    indexes = [r['source_index'] for r in plan['rows']]
    assert len(set(indexes)) == 16 and indexes != list(range(16))
    assert json.loads((output/'selection.json').read_text()) == plan
    again = prepare(source, tmp_path/'repeat', 16, [2,3,4], [10,20], 123)
    assert again['rows'] == plan['rows']
    for i in range(16):
        for arm in [item['name'] for item in plan['arms']]:
            (output/f'p{i:02d}_{arm}.png').touch()
    render(output)
    page = (output/'index.html').read_text()
    assert page.count('<img ') == 112
    assert '&lt;prompt' in page and 'p00_base.png' in page



def test_full_depth_matches_training_loop_and_full_window(visual_runtime, monkeypatch):
    from qwen_latent_cot.diagnostics.loop_training import memory_loop
    from qwen_latent_cot.diagnostics.shallow_sampling import sample, window
    runtime,row=visual_runtime;student=VisualStudent(runtime).eval().requires_grad_(False)
    runtime.config.update(sampler_steps=4)
    state=runtime.training_state(row,.7,0)
    with torch.no_grad():
        expected=memory_loop(student,state,2).exits[-1].velocity
        with shallow_loop(student,state.flow,state.lengths,state.positions,.7,3,end_layer=3):
            actual,_=runtime.velocity(state.x_t,.7,state.flow)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    monkeypatch.setattr(runtime,'image',lambda x:x)
    full,nodes=sample(student,'a red token',0,2,'full',loop_layers=3)
    numeric,_=sample(student,'a red token',0,2,3,loop_layers=3)
    torch.testing.assert_close(full,numeric,rtol=0,atol=0)
    _,short=sample(student,'a red token',0,2,2,loop_layers=1)
    assert len(nodes)==3 and all(r['memory_active'] for r in nodes)
    assert [r['memory_active'] for r in short]==[True,True,False]
    assert all(r['und_layer_calls']==3 for r in short[:2])
    assert [window(v) for v in ('early10','20','30','full')]==[10,20,30,'full']
