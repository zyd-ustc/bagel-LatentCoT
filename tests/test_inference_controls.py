"""Inference controls on real tiny BAGEL blocks and benchmark run plans."""
import json
from pathlib import Path
import sys
import pytest
import torch
from test_memory_topology import lm, runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.diagnostics.inference import ARMS, InferenceEngine, resolve_checkpoint
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint
from qwen_latent_cot.diagnostics.shallow_sampling import sample, shallow_loop
from qwen_latent_cot.diagnostics.schedule import native_times


@pytest.fixture
def engine(visual_runtime, tmp_path, monkeypatch):
    runtime, _ = visual_runtime
    runtime.config['sampler_steps'] = 4
    monkeypatch.setattr(runtime, 'image', lambda x: x)
    dtypes = {id(p): p.dtype for p in runtime.model.parameters()}
    student = VisualStudent(runtime)
    path = tmp_path/'step_002000.pt'
    save_visual_checkpoint(path, student, 2000, 'C2-feedback')
    checkpoint = torch.load(path, weights_only=True)
    # A changed C2 payload is required to catch accidental native/C2 mixing.
    for key, value in checkpoint['und'].items():
        checkpoint['und'][key] = value+0.1
    for p in student.parameters():
        p.data = p.data.to(dtypes[id(p)])
    return InferenceEngine(runtime, checkpoint)


def test_native_and_c2_weights_are_isolated_and_switch_back(engine):
    runtime = engine.runtime
    # Engine construction promotes selected parameters; generate restores the
    # actual native dtypes before computing the baseline.
    for name, value in engine.original.items():
        engine.student.native_tensor(name).data = value.clone()
    native, _ = sample(engine.student, 'a red token', 0, 1, 0)
    original, nodes = engine.generate('a red token', 0, arm='bagel')
    torch.testing.assert_close(original, native, rtol=0, atol=0)
    assert all(not r['memory_active'] and r['und_layer_calls'] == 0 for r in nodes)
    c2, _ = engine.generate('a red token', 0, arm='c2_no_memory')
    assert not torch.equal(c2, original)
    for name, value in engine.trained.items():
        torch.testing.assert_close(engine.student.native_tensor(name), value, rtol=0, atol=0)
    restored, _ = engine.generate('a red token', 0, arm='bagel')
    torch.testing.assert_close(original, restored, rtol=0, atol=0)
    assert all(engine.student.native_tensor(n).dtype == v.dtype for n, v in engine.original.items())
    assert all(not p.requires_grad for p in runtime.model.parameters())


def test_memory_and_layout_mask_match_existing_native_paths(engine):
    prompt, seed = 'a red token', 3
    image, nodes = engine.generate(prompt, seed, arm='c2_memory', early=2)
    expected, _ = sample(engine.student, prompt, seed, 2, 2)
    torch.testing.assert_close(image, expected, rtol=0, atol=0)
    assert [r['memory_visible'] for r in nodes] == [True, True, False]
    masked, records = engine.generate(prompt, seed, arm='c2_memory_masked', early=2)
    runtime = engine.runtime
    flow, lengths, positions = runtime.flow_inputs(prompt)
    size = runtime.config['resolution']//runtime.model.latent_downsample
    width = runtime.model.latent_channel*runtime.model.latent_patch_size**2
    x = torch.randn(size*size, width, generator=torch.Generator().manual_seed(seed))
    from qwen_latent_cot.diagnostics.numerics import autocast_for
    with autocast_for(runtime.device):
        bank = runtime.writer.fixed_bank(runtime.seeds)
    for i, (t, s) in enumerate(zip(native_times(runtime.config), native_times(runtime.config)[1:])):
        velocity = (runtime.gen_arm(x, t, flow, bank, lengths, positions, masked=True)[1]
                    if i < 2 else runtime.velocity(x, t, flow)[1])
        x = x-(t-s)*velocity
    torch.testing.assert_close(masked, x, rtol=0, atol=0)
    assert not torch.equal(masked, image)
    assert all(not r['memory_visible'] for r in records)
    assert records[0]['und_layer_calls'] == nodes[0]['und_layer_calls']


def test_swap_uses_same_time_independent_correct_memory_trajectory(engine):
    prompt = 'a red token'
    image, records = engine.generate(prompt, 0, arm='c2_memory_swap', early=2, donor_seed=7)
    runtime = engine.runtime
    flow, lengths, positions = runtime.flow_inputs(prompt)
    size = runtime.config['resolution']//runtime.model.latent_downsample
    width = runtime.model.latent_channel*runtime.model.latent_patch_size**2
    x = torch.randn(size*size, width, generator=torch.Generator().manual_seed(0))
    donor = torch.randn(size*size, width, generator=torch.Generator().manual_seed(7))
    # Independently compose existing primitive calls: donor writes its bank,
    # recipient consumes it, and each trajectory advances with its own velocity.
    for i, (t, s) in enumerate(zip(native_times(runtime.config), native_times(runtime.config)[1:])):
        if i < 2:
            banks = []
            with shallow_loop(engine.student, flow, lengths, positions, t, 2, on_memory=banks.append):
                _, donor_v = runtime.velocity(donor, t, flow)
            _, velocity = runtime.gen_arm(x, t, flow, banks[0], lengths, positions)
            donor = donor-(t-s)*donor_v
            assert records[i]['donor_t'] == t and records[i]['donor_prompt'] == prompt
            assert records[i]['recipient_state_hash'] != records[i]['donor_state_hash']
        else:
            _, velocity = runtime.velocity(x, t, flow)
        x = x-(t-s)*velocity
    torch.testing.assert_close(image, x, rtol=0, atol=0)
    assert records[0]['donor_und_layer_calls'] == 3 and records[2]['donor_und_layer_calls'] == 0
    with pytest.raises(ValueError, match='distinct donor'):
        engine.generate(prompt, 0, arm='c2_memory_swap', donor_seed=0)


def test_latest_checkpoint_is_numbered_and_not_mutable_last(tmp_path):
    for name in ('step_001750.pt', 'step_002000.pt', 'last.pt', 'best.pt'):
        torch.save({'step': name}, tmp_path/name)
    (tmp_path/'step_009000.pt').touch()
    (tmp_path/'step_009250.pt').write_bytes(b'PK\x03\x04truncated')
    assert resolve_checkpoint(tmp_path).name == 'step_002000.pt'


def test_benchmark_suite_keeps_arm_plans_and_scores_separate(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import benchmark_memory_controls as controls
    import memory_benchmark as benchmark
    checkpoint = tmp_path/'step_002000.pt'; checkpoint.write_bytes(b'checkpoint')
    # Two rows with the same prompt still have distinct official item IDs.
    monkeypatch.setattr(benchmark, 'catalog', lambda args: {'geneval': [
        ('00000', 'a red token', {'tag':'color'}), ('00001', 'a red token', {'tag':'color'})]})
    args = controls.benchmark.build_parser().parse_args([
        'prepare', '--run', str(tmp_path/'run'), '--checkpoint', str(checkpoint)])
    args.arms = list(ARMS)
    controls.run(args)
    identity = json.loads((Path(args.run)/'controls.json').read_text())
    assert identity['checkpoint'] == str(checkpoint)
    for arm in ARMS:
        plan = json.loads((Path(args.run)/arm/'plan.json').read_text())
        assert len(plan['jobs']) == 2 and plan['arm'] == arm
        assert all(f'/{arm}/images/' in j['image'] for j in plan['jobs'])
        folder = Path(args.run)/arm/'scores'; folder.mkdir()
        (folder/'geneval_00.jsonl').write_text(json.dumps(dict(
            benchmark='geneval', item_id='00000', seed=0, score=1., tag='color'))+'\n')
    args.command, args.benchmarks = 'summary', ['geneval']
    controls.run(args)
    result = json.loads((Path(args.run)/'summary.json').read_text())
    assert set(result) == set(ARMS)
    assert all(r['geneval']['scored_images'] == 1 and not r['geneval']['coverage_complete'] for r in result.values())
