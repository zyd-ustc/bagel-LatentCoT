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
