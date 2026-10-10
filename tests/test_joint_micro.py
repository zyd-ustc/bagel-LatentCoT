"""Contracts for synchronized native GEN/UND reads and continuous micro states."""
from dataclasses import replace
from copy import deepcopy
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
import pytest
import torch
from helpers import prepared, fixture
from qwen_latent_cot.bagel.internal_loop import LoopConfig, JOINT_MICRO
from qwen_latent_cot.bagel.native_gen import project_gen


def joint_fixture(steps=4, **kwargs):
    model, inputs, runtime = prepared(**kwargs)
    runtime.config = replace(runtime.config, mode=JOINT_MICRO, extra_rounds=0, micro_steps=steps)
    runtime.diagnostics_enabled = False
    return model, inputs, runtime


@pytest.mark.parametrize('setting', [dict(micro_steps=0), dict(micro_steps=True),
    dict(micro_steps=1.5), dict(extra_rounds=1), dict(extra_rounds=True)])
def test_joint_counts_are_explicit_total_micro_steps(setting):
    with pytest.raises(ValueError):
        LoopConfig(**(dict(mode=JOINT_MICRO, extra_rounds=0) | setting))


@pytest.mark.parametrize('device_type', ['cpu', 'npu'])
def test_gen_input_kv_projection_exactly_matches_native_attention(device_type):
    from test_npu_runtime import npu_device
    device = 'cpu' if device_type == 'cpu' else npu_device()
    model, inputs = fixture(device=device)
    decoder = model.language_model.model
    layer = decoder.layers[0]
    cos, sin = decoder.rotary_emb(inputs['packed_query_sequence'], inputs['packed_query_position_ids'].unsqueeze(0))
    rope = cos.squeeze(0), sin.squeeze(0)
    temporary = deepcopy(inputs['past_key_values'])
    args = {k: v for k, v in inputs.items() if k != 'packed_query_position_ids'}
    layer.forward_inference(**dict(args, past_key_values=temporary,
        packed_query_position_embeddings=rope, update_past_key_values=True))
    k, v = project_gen(layer, inputs['packed_query_sequence'], rope,
                       inputs['packed_text_indexes'], inputs['packed_vae_token_indexes'])
    assert torch.equal(k, temporary.key_cache[0][inputs['packed_query_indexes']])
    assert torch.equal(v, temporary.value_cache[0][inputs['packed_query_indexes']])


@pytest.mark.parametrize('steps', [1, 2, 4])
def test_micro_reads_use_old_states_and_outputs_continue_across_layers(steps):
    _, inputs, runtime = joint_fixture(steps, special_tokens=True)
    decoder = runtime.decoder
    seed = runtime.layerwise.seeds[inputs['past_key_values']]
    calls, events, originals = [], [], []
    runtime.joint_observer = lambda **e: events.append(e)
    for index, layer in enumerate(decoder.layers):
        original = layer.forward_inference
        originals.append(original)
        def capture(*, i=index, fn=original, **kw):
            if i >= runtime.config.start_layer:
                calls.append(dict(layer=i, mode=kw['mode'], hidden=kw['packed_query_sequence'].clone(),
                    lengths=kw['key_values_lens'].tolist(),
                    keys=kw['past_key_values'].key_cache[i].clone(),
                    values=kw['past_key_values'].value_cache[i].clone()))
            return fn(**kw)
        layer.forward_inference = capture
    try:
        out = decoder.forward_inference(**inputs).packed_query_sequence
        assert torch.isfinite(out).all()
        expected = [(i, mode) for i in range(1, 4)
                    for _ in range(steps if i < 3 else 1) for mode in ('gen', 'und')]
        assert [(c['layer'], c['mode']) for c in calls] == expected
        content = ~seed.special_mask
        for e, gen_call, und_call in zip(events, calls[::2], calls[1::2]):
            assert torch.equal(gen_call['hidden'], e['gen_before'])
            assert torch.equal(und_call['hidden'], e['memory_before'])
            assert gen_call['lengths'] == list(seed.lengths)
            assert und_call['lengths'] == [p + g for p, g in zip(seed.lengths, e['gen_kv'].lengths)]
            # UND must consume G at the start of the micro step, never G after GEN ran.
            pparts = inputs['past_key_values'].key_cache[e['layer']].split(seed.lengths)
            gparts = e['gen_kv'].keys.split(e['gen_kv'].lengths)
            assert torch.equal(und_call['keys'], torch.cat([torch.cat([p, g]) for p, g in zip(pparts, gparts)]))
            assert torch.equal(gen_call['keys'], e['memory_kv'].keys)
            if e['micro_steps'] == 1:
                assert torch.equal(e['gen_after'], e['gen_native'])
            else:
                assert torch.equal(e['gen_after'], e['gen_before'] + (e['gen_native']-e['gen_before']) / steps)
            assert torch.equal(e['memory_after'][seed.special_mask], seed.layer_hidden[e['layer']][seed.special_mask])
            assert torch.equal(e['memory_kv'].keys[seed.special_mask], e['native_prompt_kv'].keys[seed.special_mask])
        for a, b in zip(events, events[1:]):
            assert torch.equal(a['gen_after'], b['gen_before'])
            assert torch.equal(a['memory_after'][content], b['memory_before'][content])
        if steps > 1:
            first = [e for e in events if e['layer'] == 1]
            assert not torch.equal(first[0]['gen_before'], first[1]['gen_before'])
            assert not torch.equal(first[0]['memory_before'][content], first[1]['memory_before'][content])
    finally:
        for layer, original in zip(decoder.layers, originals):
            layer.forward_inference = original
        runtime.close()


def test_gen_output_perturbation_reaches_memory_only_on_the_next_micro_step():
    _, inputs, runtime = joint_fixture(4, start=0, end=2, special_tokens=False)
    decoder = runtime.decoder
    original = decoder.layers[0].forward_inference
    def run(perturb):
        events = []
        runtime.joint_observer = lambda **e: events.append(e)
        count = 0
        def altered(**kw):
            nonlocal count
            out, cache = original(**kw)
            if kw['mode'] == 'gen':
                if perturb and count == 0:
                    out = out + torch.linspace(-8, 8, out.shape[-1], device=out.device).to(out.dtype)
                count += 1
            return out, cache
        decoder.layers[0].forward_inference = altered
        decoder.forward_inference(**inputs)
        return [e for e in events if e['layer'] == 0]
    try:
        baseline, edited = run(False), run(True)
        assert torch.equal(baseline[0]['memory_after'], edited[0]['memory_after'])
        assert not torch.equal(baseline[0]['gen_after'], edited[0]['gen_after'])
        assert not torch.equal(baseline[1]['memory_after'], edited[1]['memory_after'])
    finally:
        decoder.layers[0].forward_inference = original
        runtime.close()


def test_one_micro_step_in_one_layer_recovers_native_gen_with_identical_context():
    _, inputs, runtime = joint_fixture(1, start=0, end=1, depth=1)
    try:
        native = runtime.original(**inputs).packed_query_sequence
        out = runtime.decoder.forward_inference(**inputs).packed_query_sequence
        assert torch.equal(native, out)
    finally:
        runtime.close()


@pytest.mark.parametrize('device_type', ['cpu', 'npu'])
def test_joint_repeatability_observer_parity_cache_isolation_and_native_bypass(device_type):
    from test_npu_runtime import npu_device
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache
    device = 'cpu' if device_type == 'cpu' else npu_device()
    _, inputs, runtime = joint_fixture(4, start=0, end=2, special_tokens=False, device=device)
    decoder = runtime.decoder
    cache = inputs['past_key_values']
    frozen = deepcopy(cache)
    seed = runtime.layerwise.seeds[cache]
    seeds = {i: h.clone() for i, h in seed.layer_hidden.items()}
    state = inputs['packed_query_sequence'].clone()
    weights = {name: weight.clone() for name, weight in runtime.model.language_model.state_dict().items()}
    try:
        out = decoder.forward_inference(**inputs).packed_query_sequence
        events = []
        runtime.joint_observer = lambda **e: events.append(e)
        assert torch.equal(out, decoder.forward_inference(**inputs).packed_query_sequence)
        assert events and torch.isfinite(out).all()
        changed = state.clone()
        changed[4:] += torch.linspace(-6, 6, state.shape[-1], device=state.device).to(state.dtype)
        edited = decoder.forward_inference(**{**inputs, 'packed_query_sequence': changed}).packed_query_sequence
        assert torch.equal(out[:4], edited[:4])
        assert not torch.equal(out[4:], edited[4:])
        assert torch.equal(inputs['packed_query_sequence'], state)
        assert all(torch.equal(cache.key_cache[i], frozen.key_cache[i])
            and torch.equal(cache.value_cache[i], frozen.value_cache[i]) for i in cache.key_cache)
        assert all(torch.equal(seed.layer_hidden[i], h) for i, h in seeds.items())
        assert all(torch.equal(runtime.model.language_model.state_dict()[name], weight) for name, weight in weights.items())
        native = runtime.original(**inputs).packed_query_sequence
        cfg = runtime.config
        runtime.config = replace(cfg, mode='BASE', micro_steps=1)
        assert torch.equal(native, decoder.forward_inference(**inputs).packed_query_sequence)
        runtime.config = replace(cfg, progress_end=19/48)
        runtime.progress = 20/48
        assert torch.equal(native, decoder.forward_inference(**inputs).packed_query_sequence)
        runtime.progress = 0
        null = {**inputs, 'past_key_values': NaiveCache(len(decoder.layers)),
            'key_values_lens': torch.zeros_like(inputs['key_values_lens']),
            'packed_key_value_indexes': torch.empty(0, dtype=torch.long, device=device),
            'packed_query_indexes': torch.arange(len(state), device=device)}
        assert torch.equal(runtime.original(**null).packed_query_sequence,
                           decoder.forward_inference(**null).packed_query_sequence)
    finally:
        runtime.close()


@pytest.mark.parametrize('steps', [1, 4])
def test_real_weight_check_orchestration_uses_actual_branch_kv_and_restores_runtime(steps):
    from qwen_latent_cot.bagel.joint_micro_checks import validate_joint_micro
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache
    model, kwargs, runtime = joint_fixture(steps, start=0, end=2)
    cache = kwargs['past_key_values']
    def flow(this, *, x_t, cfg_text_scale=1., **unused):
        conditional = runtime.decoder.forward_inference(**dict(kwargs,packed_query_sequence=x_t)).packed_query_sequence
        if cfg_text_scale == 1:
            return conditional
        null = dict(kwargs,packed_query_sequence=x_t,past_key_values=NaiveCache(4),
            key_values_lens=torch.zeros(2,dtype=torch.int32),
            packed_key_value_indexes=torch.empty(0,dtype=torch.long),
            packed_query_indexes=torch.arange(len(x_t)))
        uncond = runtime.decoder.forward_inference(**null).packed_query_sequence
        return uncond + cfg_text_scale*(conditional-uncond)
    model._forward_flow = MethodType(flow,model)
    class Generator:
        def autocast(self):return nullcontext()
        def prepare(self,*args):
            return dict(packed_init_noises=kwargs['packed_query_sequence'].clone(),past_key_values=cache),['a','b']
    cfg = runtime.config
    original = runtime.original
    runtime.progress = .2
    try:
        result = validate_joint_micro(SimpleNamespace(model=model),Generator(),runtime)
        assert result['passed']
        assert all(x['checks']['null_text_native_bypass'] for x in result['timesteps'].values())
        assert all(len(x['layers']) == 2*steps+2 for x in result['timesteps'].values())
        assert runtime.config is cfg and runtime.progress == .2 and runtime.original is original
        assert runtime.joint_observer is None
    finally:
        runtime.close()


def test_joint_comparison_binds_total_K_and_counts_layer_calls_without_old_R_labels():
    import json
    from pathlib import Path
    from qwen_latent_cot.evaluation.windows import validate_config,arm_configs,window_metadata,comparison_pairs,reference_arm
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/joint_micro_pilot.json').read_text())
    arms = arm_configs(validate_config(config,28));windows = window_metadata(config)
    assert list(arms) == ['BASE','LEGACY_EARLY_20_R2','JOINT_EARLY_20_K1','JOINT_EARLY_20_K2','JOINT_EARLY_20_K4']
    for k in (1,2,4):
        name=f'JOINT_EARLY_20_K{k}';arm=LoopConfig(**arms[name]);meta=windows[name]
        assert arm.mode == JOINT_MICRO and arm.extra_rounds == 0 and arm.micro_steps == k
        assert meta['loop_step_indexes'] == list(range(20)) and meta['step_scale'] == 1/k
        assert meta['gen_layer_calls_per_active_call'] == 8*k+20
        assert meta['und_layer_calls_per_active_call'] == 8*k+20
        assert meta['total_gen_layer_calls'] == 29*28+20*(8*k+20)
        assert meta['total_und_layer_calls'] == 20*(8*k+20)
    assert len(comparison_pairs(config)) == 5 and reference_arm(config) == 'LEGACY_EARLY_20_R2'
    shifted = {**config, 'layer_window':dict(start_layer=12,end_layer=28)}
    validate_config(shifted,28)
    assert window_metadata(shifted)['JOINT_EARLY_20_K4']['und_layer_calls_per_active_call'] == 64


@pytest.mark.parametrize('start,end', [(0,8), (12,28)])
def test_28_layer_windows_have_native_prefix_and_one_suffix_traversal(start,end):
    _, inputs, runtime = joint_fixture(4, start=start, end=end, depth=28)
    events = []
    runtime.joint_observer = lambda **e: events.append((e['layer'], e['micro_steps']))
    try:
        out = runtime.decoder.forward_inference(**inputs).packed_query_sequence
        assert torch.isfinite(out).all()
        assert events == [(i,4 if i<end else 1) for i in range(start,28)
                          for _ in range(4 if i<end else 1)]
    finally:
        runtime.close()
