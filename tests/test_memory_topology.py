"""New protocol tests using real tiny native MoT layers, never historical loops."""
from types import SimpleNamespace
import json
import pytest
import torch
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM, NaiveCache
from qwen_latent_cot.diagnostics.memory import (
    KV, MemoryWriter, neutral_seeds, rope_for, rotate, project, consume_bank, packed_attention, und_block)
from qwen_latent_cot.diagnostics.native import teacher_forcing, gen_memory_call, capture_gen
from qwen_latent_cot.diagnostics.data import controlled_noise, score_facts, load_dataset, velocity_stats
from qwen_latent_cot.diagnostics.record import FrozenGuard


@pytest.fixture
def lm():
    torch.manual_seed(7)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                     num_attention_heads=4, num_key_value_heads=2, layer_module='Qwen2MoTDecoderLayer')
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)


def native_call(lm, h, positions, lengths, cache, prefix_lengths, mode='und', update=False):
    query = []; past = []; offset = 0
    for n, p in zip(lengths, prefix_lengths):
        past.extend(range(offset, offset+p)); query.extend(range(offset+p, offset+p+n)); offset += p+n
    kwargs = dict(packed_query_sequence=h, query_lens=torch.tensor(lengths, dtype=torch.int32),
                  packed_query_position_ids=positions, packed_query_indexes=torch.tensor(query),
                  past_key_values=cache, key_values_lens=torch.tensor(prefix_lengths, dtype=torch.int32),
                  packed_key_value_indexes=torch.tensor(past, dtype=torch.long), update_past_key_values=update,
                  is_causal=mode == 'und', mode=mode)
    if mode == 'gen':
        kwargs.update(packed_text_indexes=torch.tensor([0, len(h)-1]),
                      packed_vae_token_indexes=torch.arange(1, len(h)-1))
    return lm.forward_inference(**kwargs)


def source_for(lm, samples=1):
    hidden = lm.model.embed_tokens(torch.arange(4, 9).repeat(samples))
    cache = NaiveCache(3)
    native_call(lm, hidden, torch.arange(5).repeat(samples), (5,)*samples, cache, (0,)*samples, update=True)
    return tuple(KV(i, cache.key_cache[i], cache.value_cache[i], (5,)*samples) for i in range(3)), cache


def setup_writer(lm, topology='C', samples=1):
    source, _ = source_for(lm, samples)
    seeds = neutral_seeds(lm.model, torch.arange(16), samples)
    writer = MemoryWriter(lm.model, topology)
    return writer, seeds, source, torch.arange(20, 36).repeat(samples)


def test_neutral_seed_prefill_matches_native_inputs(lm):
    ids = torch.arange(16)
    seeds = neutral_seeds(lm.model, ids)
    captured = []; handles = []
    for layer in lm.model.layers:
        handles.append(layer.register_forward_pre_hook(lambda m, args, kw: captured.append(kw['packed_query_sequence'].clone()), with_kwargs=True))
    native_call(lm, lm.model.embed_tokens(ids), torch.arange(16), (16,), NaiveCache(3), (0,), update=True)
    for handle in handles: handle.remove()
    for ours, native in zip(seeds, captured): torch.testing.assert_close(ours, native, rtol=0, atol=0)


def test_rope_once_and_same_layer_guard(lm):
    writer, seeds, source, positions = setup_writer(lm)
    bank = writer(seeds, source, positions)
    consumed = consume_bank(lm.model, bank, [42])
    original = bank.layers[0].keys.clone()
    rope = rope_for(lm.model, original, torch.arange(42, 58))
    torch.testing.assert_close(consumed[0].keys, rotate(original, rope), rtol=0, atol=0)
    torch.testing.assert_close(bank.layers[0].keys, original, rtol=0, atol=0)
    bad = list(source); bad[1] = source[0]
    with pytest.raises(ValueError): writer(seeds, bad, positions)


def test_packed_sample_isolation_and_whole_bank_swap(lm):
    writer, seeds, source, positions = setup_writer(lm, samples=2)
    bank = writer(seeds, source, positions)
    changed = tuple(KV(i, torch.cat((s.keys[:5], s.keys[5:]+10)), s.values, s.lengths) for i, s in enumerate(source))
    other = writer(seeds, changed, positions)
    for a, b in zip(bank.layers, other.layers):
        torch.testing.assert_close(a.keys[:16], b.keys[:16], rtol=0, atol=0)
    swapped = bank.swap([1, 0])
    for a, b in zip(bank.layers, swapped.layers):
        torch.testing.assert_close(b.keys[:16], a.keys[16:], rtol=0, atol=0)


def test_mask_excludes_keys_instead_of_zeroing_values():
    q = torch.tensor([[[1.]]]); k = torch.tensor([[[1.]], [[100.]]]); v = torch.tensor([[[7.]], [[0.]]])
    masked = packed_attention(q, k, v, (1,), (2,), visible=torch.tensor([True, False]))
    zero_value = packed_attention(q, k, v, (1,), (2,))
    assert masked.item() == 7 and zero_value.item() < .01


def test_caption_is_causal_and_targets_are_shifted(lm):
    writer, seeds, source, positions = setup_writer(lm)
    bank = writer(seeds, source, positions)
    bundle = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    prefix = consume_bank(lm.model, bank, [0])
    loss, logits = teacher_forcing(bundle, [1, 2, 3], [7, 8, 9], prefix, 16)
    changed_loss, changed = teacher_forcing(bundle, [1, 2, 3], [7, 8, 10], prefix, 16)
    torch.testing.assert_close(logits, changed, rtol=0, atol=0)  # final target never fed as input
    assert loss != changed_loss
    _, changed_prefix = teacher_forcing(bundle, [1, 2, 3], [7, 11, 9], prefix, 16)
    torch.testing.assert_close(logits[:2], changed_prefix[:2], rtol=0, atol=0)


def test_source_off_ignores_image(lm):
    writer, seeds, source, positions = setup_writer(lm)
    changed = tuple(KV(s.layer, s.keys*100, s.values+100, s.lengths) for s in source)
    a, b = writer(seeds, source, positions, True), writer(seeds, changed, positions, True)
    for x, y in zip(a.layers, b.layers): torch.testing.assert_close(x.keys, y.keys, rtol=0, atol=0)


def test_native_gen_capture_matches_native_cache(lm):
    _, cache = source_for(lm)
    h = lm.model.embed_tokens(torch.arange(6))
    with capture_gen(lm.model, cache) as captured:
        native_call(lm, h, torch.full((6,), 5), (6,), cache, (5,), 'gen', update=True)
    for i, kv in captured.items():
        torch.testing.assert_close(kv.keys, cache.key_cache[i][5:], rtol=0, atol=0)
        torch.testing.assert_close(kv.values, cache.value_cache[i][5:], rtol=0, atol=0)


def test_gen_layout_matches_native_shift_and_restores(lm):
    _, cache = source_for(lm)
    writer, seeds, source, positions = setup_writer(lm)
    fixed = writer.fixed_bank(seeds)
    h = lm.model.embed_tokens(torch.arange(6))
    snapshot = [cache.key_cache[i].clone() for i in range(3)]
    original = lm.model.forward_inference
    expected = native_call(lm, h, torch.full((6,), 21), (6,), cache, (5,), 'gen').packed_query_sequence
    with gen_memory_call(lm.model, cache, fixed, (5,), [5], masked=True):
        actual = native_call(lm, h, torch.full((6,), 5), (6,), cache, (5,), 'gen').packed_query_sequence
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert lm.model.forward_inference == original
    for i in range(3): torch.testing.assert_close(cache.key_cache[i], snapshot[i], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        with gen_memory_call(lm.model, cache, fixed, (5,), [5]):
            native_call(lm, h, torch.full((6,), 5), (6,), cache, (5,), 'gen')
            native_call(lm, h, torch.full((6,), 5), (6,), cache, (5,), 'gen')
    assert lm.model.forward_inference == original


def test_pure_noise_independent_of_x0_and_rng():
    state = torch.random.get_rng_state().clone()
    a = controlled_noise(torch.zeros(8, 4), 1., 12)
    b = controlled_noise(torch.full((8, 4), float('nan')), 1., 12)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)


def test_freeze_guard_rejects_updates(lm):
    guard = FrozenGuard(lm)
    with torch.no_grad(): lm.lm_head.weight.add_(1)
    with pytest.raises(RuntimeError): guard.check()


def test_scene_split_and_swap_contract(tmp_path):
    rows = [dict(sample_id=str(i), scene_id='shared', split=split, caption='a red cube', label_type='clean',
                 verified=True, facts=[]) for i, split in enumerate(['train', 'test'])]
    path = tmp_path/'data.jsonl'; path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match='family'): load_dataset(path)


def test_fact_rules_keep_unknown_and_negation_separate():
    facts = [dict(category='attribute', aliases=['red'], alternatives=['blue']),
             dict(category='count', aliases=['two'], alternatives=['three'], unknown=True)]
    assert score_facts('A red cube.', facts)['fact_accuracy'] == 1
    assert score_facts('No red cube.', facts)['fact_accuracy'] == 0
    assert score_facts('Two red cubes.', facts)['unknown_assertion_rate'] == 1


def test_velocity_metrics_reject_nan():
    with pytest.raises(FloatingPointError): velocity_stats(torch.tensor([float('nan')]), torch.ones(1))

@pytest.fixture
def runtime(lm, tmp_path):
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel, BagelConfig
    from qwen_latent_cot.diagnostics.runtime import DiagnosticRuntime
    class Tokenizer:
        all_special_ids = [1, 2, 3]
        def encode(self, text, add_special_tokens=True):
            return list(range(4, 24)) if len(text) > 30 else [4, 5, 6, 7]
        def decode(self, ids, skip_special_tokens=True):
            return 'token' if ids else ''
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        def encode(self, pixels):
            return torch.nn.functional.avg_pool2d(pixels[:, :2], 8)
        def decode(self, latent):
            return torch.nn.functional.interpolate(torch.cat((latent, latent[:, :1]), 1), scale_factor=8)
    model = Bagel(lm, None, BagelConfig(visual_und=False, llm_config=lm.config,
                  vae_config=SimpleNamespace(downsample=8, z_channels=2), max_latent_size=4)).to(torch.bfloat16).eval()
    torch.nn.init.normal_(model.llm2vae.weight, std=.01)
    cfg = json.loads(__import__('pathlib').Path('configs/memory_topology.json').read_text())
    cfg.update(resolution=32, steps=1, effective_batch=1, eval_every=1, max_caption_tokens=2,
               eval_t=[.5, 1.], noise_seeds=[0], sampler_steps=4)
    bundle = SimpleNamespace(model=model, vae=VAE(), tokenizer=Tokenizer(), token_ids={
        'bos_token_id':1, 'eos_token_id':2, 'start_of_image':1, 'end_of_image':3}, model_path=str(tmp_path))
    runtime = DiagnosticRuntime(bundle, cfg, 'C'); runtime.native_files = {'fake': 'engineering-only'}
    return runtime


def test_checkpoint_rejects_native_provenance_change(runtime, tmp_path):
    from qwen_latent_cot.diagnostics.record import validate_checkpoint
    saved = dict(native_files=runtime.native_files)
    runtime.native_files = {'different': 'checkpoint'}
    with pytest.raises(ValueError, match='provenance'):
        validate_checkpoint(saved, runtime)


def test_original_image_cannot_cross_split_under_different_ids(tmp_path):
    path = tmp_path/'image.bin'; path.write_bytes(b'same original image')
    rows = [dict(sample_id=str(i), scene_id=str(i), split=split, caption='x', verified=True,
                 label_type='clean', image='image.bin', facts=[]) for i, split in enumerate(['train', 'test'])]
    dataset = tmp_path/'data.jsonl'; dataset.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='original image'): load_dataset(dataset)


def test_literal_negation_is_local_to_assertion():
    facts = [dict(category='attribute', aliases=['red'], alternatives=['blue'])]
    for text in ('a red cube, not blue', 'a red cube but not blue', 'not blue, a red cube'):
        assert score_facts(text, facts)['fact_accuracy'] == 1
    for text in ('no red cube', 'not a red cube', 'a red and blue cube', 'unknown red cube'):
        assert score_facts(text, facts)['fact_accuracy'] == 0


def test_c0_uses_own_repeat_instruction(runtime):
    inputs, _, _ = runtime.model.prepare_prompts([0], [0], [runtime.config['c0_instruction']],
        runtime.bundle.tokenizer, runtime.bundle.token_ids)
    expected = inputs['packed_text_ids'].tolist()
    assert runtime.c0_instruction_ids == expected
    assert runtime.config['c0_instruction'] != runtime.config['caption_instruction']


def test_continuation_rejects_partial_first_step(runtime):
    x = torch.ones(4, 8)
    with pytest.raises(ValueError, match='native'):
        runtime.continue_once(x, .05, x, {}, [0.])


def test_real_qwen2_tokenizer_matches_native_prompt_wrapper(runtime, tmp_path):
    from transformers import Qwen2Tokenizer
    from transformers.models.qwen2.tokenization_qwen2 import bytes_to_unicode
    vocab = {char: index for index, char in enumerate(bytes_to_unicode().values())}
    vocab.update({'<|endoftext|>': 256, '<|im_start|>': 257, '<|im_end|>': 258})
    vocab_path = tmp_path/'vocab.json'; vocab_path.write_text(json.dumps(vocab))
    merges_path = tmp_path/'merges.txt'; merges_path.write_text('#version: 0.2\n')
    tokenizer = Qwen2Tokenizer(vocab_file=str(vocab_path), merges_file=str(merges_path))
    runtime.bundle.tokenizer = tokenizer
    runtime.bundle.token_ids.update(bos_token_id=257, eos_token_id=258)
    for instruction in (runtime.config['c0_instruction'], runtime.config['caption_instruction'], '复述红色方块。'):
        actual = runtime.prompt_ids(instruction)
        prepared, _, _ = runtime.model.prepare_prompts([0], [0], [instruction], tokenizer, runtime.bundle.token_ids)
        assert actual == prepared['packed_text_ids'].tolist()
        assert actual == [257]+tokenizer.encode(instruction, add_special_tokens=False)+[258]
    assert runtime.prompt_ids(runtime.config['c0_instruction']) != runtime.prompt_ids(runtime.config['caption_instruction'])
    assert runtime.answer_ids('a red cube') == tokenizer.encode('a red cube', add_special_tokens=False)+[258]


def test_controlled_continuation_uses_complete_native_steps(runtime, monkeypatch):
    from qwen_latent_cot.diagnostics.schedule import native_times, native_suffix, nearest_node
    runtime.config.update(sampler_steps=50, timestep_shift=3)
    times = native_times(runtime.config)
    for requested in (.05, .25, .5, .75, .9):
        index, actual = nearest_node(runtime.config, requested)
        seen = []
        def velocity(x, t, flow):
            seen.append(t); return x, torch.ones_like(x)
        monkeypatch.setattr(runtime, 'velocity', velocity)
        result = runtime.continue_once(torch.ones(4, 8), actual, torch.ones(4, 8), {}, native_suffix(runtime.config, actual))
        assert seen == times[index+1:-1]
        torch.testing.assert_close(result, torch.full_like(result, 1-actual))
        assert native_suffix(runtime.config, actual)[0] == times[index+1]
        with pytest.raises(ValueError, match='native'):
            runtime.continue_once(torch.ones(4, 8), actual, torch.ones(4, 8), {}, native_suffix(runtime.config, actual)[:-1])


def test_image_posterior_seeds_depend_on_content_and_preserve_rng(runtime, tmp_path):
    from PIL import Image
    a = tmp_path/'a.png'; b = tmp_path/'b.png'; same = tmp_path/'same.png'
    Image.new('RGB', (32, 32), color='red').save(a)
    Image.new('RGB', (32, 32), color='blue').save(b)
    same.write_bytes(a.read_bytes())
    before = torch.random.get_rng_state().clone()
    assert runtime.image_posterior_seed(a) == runtime.image_posterior_seed(same)
    assert runtime.image_posterior_seed(a) != runtime.image_posterior_seed(b)
    runtime.encode_image(a, runtime.image_posterior_seed(a))
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)


def test_trajectory_rejects_different_native_schedule(runtime, tmp_path):
    path = tmp_path/'state.pt'
    torch.save(dict(state_source='trajectory', prompt='an object', t=.5, noise_seed=0,
                    sampler_steps=50, timestep_shift=3, x_t=torch.ones(4,8), step_index=1), path)
    with pytest.raises(ValueError, match='sampler grid'):
        runtime.state(dict(state=str(path), prompt='an object', label_type='observed'), .5, 0)
