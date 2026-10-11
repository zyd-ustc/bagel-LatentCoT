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


def setup_writer(lm, topology='L', samples=1):
    source, _ = source_for(lm, samples)
    seeds = neutral_seeds(lm.model, torch.arange(16), samples)
    writer = MemoryWriter(lm.model, topology, rank=4)
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


@pytest.mark.parametrize('topology', ['L', 'C'])
def test_full_bank_post_block_readout_and_gradient(lm, topology):
    writer, seeds, source, positions = setup_writer(lm, topology)
    guard = FrozenGuard(lm)
    bank = writer(seeds, source, positions)
    assert len(bank.layers) == 3 and all(not kv.rotated for kv in bank.layers)
    for kv in bank.layers: assert kv.keys.shape == (16, 2, 8)
    bundle = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    loss, logits = teacher_forcing(bundle, [1, 2, 3], [7, 8, 9], consume_bank(lm.model, bank, [0]), 16)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in writer.parameters())
    assert all(adapter.up.weight.grad.abs().max() > 0 for adapter in writer.adapters)
    before = {n: p.clone() for n, p in lm.named_parameters()}
    torch.optim.AdamW(writer.parameters(), lr=.01).step()
    guard.check()
    for n, p in lm.named_parameters(): torch.testing.assert_close(p, before[n], rtol=0, atol=0)


def test_topology_changes_only_hidden_propagation(lm):
    l, seeds, source, positions = setup_writer(lm, 'L')
    c = MemoryWriter(lm.model, 'C', rank=4); c.load_state_dict(l.state_dict())
    lb, cb = l(seeds, source, positions), c(seeds, source, positions)
    torch.testing.assert_close(lb.layers[0].keys, cb.layers[0].keys, rtol=0, atol=0)
    assert not torch.equal(lb.layers[1].keys, cb.layers[1].keys)
    # Alter layer zero adapter only. L layer one cannot change; C must change.
    with torch.no_grad(): l.adapters[0].up.weight.fill_(.1); c.adapters[0].up.weight.fill_(.1)
    ll, cc = l(seeds, source, positions), c(seeds, source, positions)
    torch.testing.assert_close(lb.layers[1].keys, ll.layers[1].keys, rtol=0, atol=0)
    assert not torch.equal(cb.layers[1].keys, cc.layers[1].keys)
    assert sum(p.numel() for p in l.parameters()) == sum(p.numel() for p in c.parameters())


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
    runtime = DiagnosticRuntime(bundle, cfg, 'L'); runtime.native_files = {'fake': 'engineering-only'}
    return runtime


def test_train_transfer_caption_and_gen_pipeline(runtime, tmp_path):
    from pathlib import Path
    from PIL import Image
    from qwen_latent_cot.diagnostics.caption import train, evaluate
    from qwen_latent_cot.diagnostics.gen import diagnose_gen
    from qwen_latent_cot.diagnostics.record import load_checkpoint
    image = tmp_path/'image.png'; Image.new('RGB', (32, 32), color='red').save(image)
    def row(name, split, swap=None):
        return dict(sample_id=name, scene_id=name, split=split, caption='a token', label_type='clean',
                    verified=True, prompt='an object', image=str(image), swap_id=swap,
                    facts=[dict(category='object', aliases=['token'], alternatives=['missing'])])
    rows = [row('train', 'train'), row('val', 'val'), row('test_a', 'test', 'test_b'), row('test_b', 'test', 'test_a')]
    c1 = tmp_path/'c1'; c1.mkdir()
    train(runtime, rows, 'C1', c1)
    assert (c1/'best.pt').exists() and (c1/'last.pt').exists()
    c2 = tmp_path/'c2'; c2.mkdir()
    train(runtime, rows, 'C2', c2, str(c1/'best.pt'))
    assert (c2/'best.pt').exists()
    cap = tmp_path/'caption'; cap.mkdir()
    evaluated = evaluate(runtime, rows[2:], 'C2', cap)
    assert {'Correct', 'Source-off', 'Source-swap', 'Memory-swap', 'Memory-masked', 'Pure-noise'} <= {r['arm'] for r in evaluated}
    runtime.config['eval_t'] = [.5]
    gen = tmp_path/'gen'; gen.mkdir()
    diagnose_gen(runtime, rows[2:], gen, str(c1/'best.pt'), str(c2/'best.pt'), continuation=True)
    records = [json.loads(line) for line in (gen/'gen.jsonl').read_text().splitlines()]
    assert {'G-base', 'G-layout', 'G-fixed', 'G-neutral', 'G-text', 'G-TL', 'G-VL', 'G-swap-TL', 'G-swap-VL'} == {r['arm'] for r in records}
    assert (gen/'blind_annotations.csv').exists()
    assert all(r['prompt'] == 'an object' and r['writer_prompt'] == 'An image.' for r in records)
    assert all(r['frozen_guard'] == 'passed' for r in records)
    from qwen_latent_cot.diagnostics.schedule import native_suffix, nearest_node
    actual = nearest_node(runtime.config, .5)[1]
    assert all(r['requested_t'] == .5 and r['t'] == actual for r in records)
    assert all(r['continuation_times'] == [actual]+native_suffix(runtime.config, actual) for r in records)
    expected = runtime.state(rows[2], actual, 0)[0]
    stored = torch.load(gen/'test_a_000.pt', weights_only=True)
    torch.testing.assert_close(stored['x_t'], expected, rtol=0, atol=0)
    swap_rows = [r for r in evaluated if 'swap' in r['arm']]
    assert all(r['intervention_evidence_group'] == 'donor_bank_swap' for r in swap_rows)
    assert all(r['equivalent_to'] == 'Source-swap' for r in swap_rows if r['arm'] == 'Memory-swap')
    train_records = [json.loads(line) for line in (c2/'train.jsonl').read_text().splitlines()]
    assert all(g['k_max'] > 0 and g['v_max'] > 0 for g in train_records[0]['memory_gradient_audit'])
    assert train_records[0]['sampled_states'][0]['t'] in runtime.config['train_t']


def test_checkpoint_rejects_native_provenance_change(runtime, tmp_path):
    from qwen_latent_cot.diagnostics.record import save_checkpoint, load_checkpoint
    path = tmp_path/'checkpoint.pt'; save_checkpoint(path, runtime, 1, 'C1')
    runtime.native_files = {'different': 'checkpoint'}
    with pytest.raises(ValueError, match='provenance'): load_checkpoint(path, runtime)


def test_trajectory_records_actual_native_times(runtime, tmp_path):
    from qwen_latent_cot.diagnostics.trajectory import collect
    folder = tmp_path/'trajectory'; folder.mkdir()
    collect(runtime, [dict(sample_id='sample', scene_id='scene', split='test', prompt='an object')], folder)
    rows = [json.loads(line) for line in (folder/'observation_candidates.jsonl').read_text().splitlines()]
    assert rows and all(not r['verified'] and r['label_type'] == 'observed' for r in rows)
    for row in rows:
        state = torch.load(folder/row['state'], weights_only=True)
        assert state['t'] == row['actual_t'] and state['state_source'] == 'trajectory'


def test_report_keeps_four_tables_and_does_not_invent_blind_labels(tmp_path):
    from qwen_latent_cot.diagnostics.report import summarize
    from qwen_latent_cot.diagnostics.record import write_json
    run = tmp_path/'run'; run.mkdir()
    write_json(run/'manifest.json', dict(model_files={}, dataset_hash='fixed', depth=3, slots=16,
                                        config={}, source_hashes={}, topology='L'))
    item = dict(topology='L', stage='C1', arm='Correct', label_type='clean', state_source='text_oracle',
                initialization_only=False, nll=2., fact_accuracy=1., scene_id='one', facts=[])
    (run/'captions.jsonl').write_text(json.dumps(item)+'\n')
    output = tmp_path/'report'; summarize([str(run)], output)
    tables = json.loads((output/'tables.json').read_text())
    assert set(tables) == {'text_bridge', 'visual_caption', 'gen_local', 'final_image'}
    assert tables['final_image'] == [] and tables['text_bridge'][0]['fact_accuracy']['mean'] == 1.


def test_report_rejects_incomparable_models(tmp_path):
    from qwen_latent_cot.diagnostics.report import summarize
    from qwen_latent_cot.diagnostics.record import write_json
    runs = []
    for index in range(2):
        run = tmp_path/str(index); run.mkdir(); runs.append(str(run))
        write_json(run/'manifest.json', dict(model_files={'weights':str(index)}, dataset_hash='fixed', depth=3,
                                            slots=16, config={}, source_hashes={}, topology='L'))
    with pytest.raises(ValueError, match='different'): summarize(runs, tmp_path/'output')


def test_original_image_cannot_cross_split_under_different_ids(tmp_path):
    path = tmp_path/'image.bin'; path.write_bytes(b'same original image')
    rows = [dict(sample_id=str(i), scene_id=str(i), split=split, caption='x', verified=True,
                 label_type='clean', image='image.bin', facts=[]) for i, split in enumerate(['train', 'test'])]
    dataset = tmp_path/'data.jsonl'; dataset.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='original image'): load_dataset(dataset)


def test_cli_c0_manifest_serializes_native_vae_config(runtime, tmp_path, monkeypatch):
    import sys
    from qwen_latent_cot.diagnostics.cli import main
    import qwen_latent_cot.bagel.backbone as backbone
    from qwen_latent_cot.bagel.modeling.autoencoder import AutoEncoderParams
    # Use the ACTUAL native VAE config dataclass, which transformers leaves nested.
    runtime.model.config.vae_config = AutoEncoderParams(256, 3, 8, 128, 3, [1,2,4,4], 2, 16, .3611, .1159)
    native_dir = tmp_path/'native'; native_dir.mkdir()
    (native_dir/'llm_config.json').write_text('{}')
    runtime.bundle.model_path = str(native_dir)
    config = dict(runtime.config, resolution=512)
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    dataset = tmp_path/'dataset.jsonl'
    dataset.write_text(json.dumps(dict(sample_id='calibrate', scene_id='scene', split='val',
                       caption='a token', label_type='clean', verified=True,
                       facts=[dict(category='object', aliases=['token'], alternatives=['absent'])]))+'\n')
    monkeypatch.setattr(backbone, 'load_native', lambda *args: runtime.bundle)
    output = tmp_path/'calibration'
    monkeypatch.setattr(sys, 'argv', ['memory-topology', 'calibrate', '--model', str(native_dir),
                        '--data', str(dataset), '--config', str(config_path), '--split', 'val', '--output', str(output)])
    main()
    record = json.loads((output/'manifest.json').read_text())
    assert record['native_config']['vae_config']['scale_factor'] == .3611
    assert record['status'].startswith('executed')
    assert not json.loads((output/'calibration.json').read_text())['passed']
    assert not any('build/' in name for name in record['source_hashes'])

@pytest.mark.parametrize('path', ['keys', 'values'])
@pytest.mark.parametrize('topology', ['L', 'C'])
def test_caption_gradient_reaches_each_memory_path(lm, topology, path):
    from qwen_latent_cot.diagnostics.memory import MemoryBank
    writer, seeds, source, positions = setup_writer(lm, topology)
    original = writer(seeds, source, positions)
    for kv in original.layers:
        getattr(kv, path).retain_grad()
    bank = MemoryBank(tuple(KV(kv.layer,
        kv.keys if path == 'keys' else kv.keys.detach(),
        kv.values if path == 'values' else kv.values.detach(), kv.lengths, False)
        for kv in original.layers), topology, 16)
    loss, _ = teacher_forcing(SimpleNamespace(model=SimpleNamespace(language_model=lm)),
                             [1, 2, 3], [7, 8, 9], consume_bank(lm.model, bank, [0]), 16)
    loss.backward()
    assert all(getattr(kv, path).grad is not None and getattr(kv, path).grad.abs().max() > 0
               for kv in original.layers)
    assert all(a.up.weight.grad is not None and a.up.weight.grad.abs().max() > 0 for a in writer.adapters)


def test_chain_gradient_crosses_frozen_und_blocks(lm):
    from qwen_latent_cot.diagnostics.memory import MemoryBank
    writer, seeds, source, positions = setup_writer(lm, 'C')
    # Only the last layer bank is connected: earlier adapters need downstream UND input gradients.
    bank = writer(seeds, source, positions)
    isolated = MemoryBank(tuple(KV(kv.layer, kv.keys if i == 2 else kv.keys.detach(),
        kv.values if i == 2 else kv.values.detach(), kv.lengths, False)
        for i, kv in enumerate(bank.layers)), 'C', 16)
    loss, _ = teacher_forcing(SimpleNamespace(model=SimpleNamespace(language_model=lm)),
                             [1, 2, 3], [7, 8, 9], consume_bank(lm.model, isolated, [0]), 16)
    loss.backward()
    assert writer.adapters[0].up.weight.grad.abs().max() > 0
    assert all(p.grad is None and not p.requires_grad for p in lm.parameters())


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


def test_low_rank_residual_is_linear(lm):
    from qwen_latent_cot.diagnostics.memory import LowRankResidual
    adapter = LowRankResidual(32, 4)
    with torch.no_grad(): adapter.up.weight.normal_()
    x = torch.randn(16, 32)
    torch.testing.assert_close(adapter(2*x), 2*adapter(x))


def test_continuation_rejects_partial_first_step(runtime):
    x = torch.ones(4, 8)
    with pytest.raises(ValueError, match='native'):
        runtime.continue_once(x, .05, x, {}, [0.])


def test_pure_noise_audit_compares_hashes_without_swap():
    from qwen_latent_cot.diagnostics.caption import check_pure_noise
    rows = [dict(t=1, arm='Correct', state_source='controlled', noise_seed=0,
                 x_t_hash=h, generated_ids=[1]) for h in ('a', 'b')]
    with pytest.raises(RuntimeError, match='pure.noise'):
        check_pure_noise(rows)


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


def test_c0_both_decode_paths_receive_repeat_instruction(runtime, monkeypatch):
    import qwen_latent_cot.diagnostics.caption as caption
    calls = []
    runtime.c0_instruction_ids = [1, 11, 12, 2]
    runtime.instruction_ids = [1, 20, 21, 2]
    def loss(bundle, instruction, answer, prefix, start):
        calls.append(instruction); return torch.tensor(1.), None
    def free(bundle, instruction, prefix, start, max_tokens):
        calls.append(instruction); return 'a red cube', [11]
    monkeypatch.setattr(caption, 'teacher_forcing', loss)
    monkeypatch.setattr(caption, 'free_caption', free)
    caption.caption_result(runtime, dict(caption='a red cube', facts=[]), None, 0, 'C0')
    assert calls == [runtime.c0_instruction_ids]*2


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


def test_checkpoint_rejects_pre_correction_contract(runtime, tmp_path):
    from qwen_latent_cot.diagnostics.record import save_checkpoint, load_checkpoint
    path = tmp_path/'old.pt'; save_checkpoint(path, runtime, 1, 'C1')
    old = torch.load(path, weights_only=True)
    del old['config']['adapter_type']; torch.save(old, path)
    with pytest.raises(ValueError, match='adapter_type'):
        load_checkpoint(path, runtime)


def test_pure_noise_audit_rejects_output_mismatch_at_same_hash():
    from qwen_latent_cot.diagnostics.caption import check_pure_noise
    rows = [dict(t=1, arm='Correct', state_source='controlled', noise_seed=0,
                 x_t_hash='same', generated_ids=[token]) for token in (1, 2)]
    with pytest.raises(RuntimeError, match='pure.noise'):
        check_pure_noise(rows)


def test_trajectory_rejects_different_native_schedule(runtime, tmp_path):
    path = tmp_path/'state.pt'
    torch.save(dict(state_source='trajectory', prompt='an object', t=.5, noise_seed=0,
                    sampler_steps=50, timestep_shift=3, x_t=torch.ones(4,8), step_index=1), path)
    with pytest.raises(ValueError, match='sampler grid'):
        runtime.state(dict(state=str(path), prompt='an object', label_type='observed'), .5, 0)
