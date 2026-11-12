from qwen_latent_cot.u15_target_correction.cli import _parser


def test_inference_commands_accept_checkpoint_modes():
    parser = _parser()
    common = ['--model', 'model', '--source', 'source', '--config', 'config', '--output', 'out']
    edit = parser.parse_args(['edit', *common, '--image', 'source.png', '--instruction', 'Add a tree.', '--checkpoint-mode', 'native'])
    assert edit.checkpoint_mode == 'native'
    t2i = parser.parse_args(['t2i', *common, '--prompt', 'A tree.', '--checkpoint-mode', 'trained', '--checkpoint', 'step'])
    assert t2i.checkpoint_mode == 'trained'


def test_reflection_parser_rejects_invalid_match():
    from qwen_latent_cot.u15_target_correction.diagnosis import parse_reflection
    result = parse_reflection('DESCRIPTION: A tree.\nMATCH: maybe\nINSTRUCTION: Add a dog.\nPRESERVE: Tree.')
    assert result['outcome'] == 'failed_parse'
    assert result['instruction'] == ''


def test_source_image_uses_stage1_linear_noising(monkeypatch):
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.u15_target_correction import inference

    source = torch.full((1, 3, 2, 2), 0.5)
    rt = SimpleNamespace(width=2, height=2, device=torch.device('cpu'), noise_scale=lambda: 2)
    monkeypatch.setattr(inference, 'load_canvas', lambda *args: (source, None, None))
    monkeypatch.setattr(inference, '_edit', lambda editor, state, time, instruction, steps: state + 1)
    monkeypatch.setattr(inference, 'phase_grid', lambda *args: [0, 0.36, 1])
    monkeypatch.setattr(inference, 'NativeT2I', lambda *args: None)
    monkeypatch.setattr(inference, 'preview_image', lambda state, *args: state)
    monkeypatch.setattr(inference, 'euler', lambda state, *args: state)
    cfg = {'read_time': 0.36, 'edit_steps': 8}
    images, edited, record = inference.edit_image(rt, None, cfg, 'source.png', 'blue', 9)
    generator = torch.Generator().manual_seed(9)
    expected = 0.36 * source + 0.64 * 2 * torch.randn(source.shape, generator=generator) + 1
    assert torch.equal(edited, expected)
    assert torch.equal(images['result'], expected)
    assert record['instruction'] == 'blue'


def test_t2i_reflection_controls_edit_and_retains_decision(monkeypatch):
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.u15_target_correction import inference

    rt = SimpleNamespace(width=2, height=2, device=torch.device('cpu'), noise_scale=lambda: 1)
    cfg = {'read_time': 0.36, 'edit_steps': 8}
    monkeypatch.setattr(inference, 'phase_grid', lambda *args: [0, 0.36, 1])
    monkeypatch.setattr(inference, 'NativeT2I', lambda *args: None)
    monkeypatch.setattr(inference, 'euler', lambda state, *args: state)
    monkeypatch.setattr(inference, 'preview_image', lambda state, *args: state)
    monkeypatch.setattr(inference, 'review_reflection', lambda *args: {'reflection': {
        'outcome': 'edit', 'instruction': 'Add a tree.', 'raw': 'response'}})
    monkeypatch.setattr(inference, '_edit', lambda editor, state, time, instruction, steps: state + 1)
    images, edited, record = inference.t2i_feedback(rt, None, cfg, 'A tree.', 4)
    assert torch.equal(images['result'], edited)
    assert record['reflection']['instruction'] == 'Add a tree.'
    assert 'edited_preview' in images


def test_training_field_uses_shared_noise_for_pair():
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.u15_target_correction.train import _linear_fields

    source = torch.full((1, 3, 2, 2), -0.5)
    target = torch.full_like(source, 0.75)
    sample = {'source': source, 'target': target, 'time': .36, 'valid': torch.ones(1, 1, 2, 2)}
    editor = SimpleNamespace(rt=SimpleNamespace(noise_scale=lambda: 2.0))
    anchor, destination, _, valid, _ = _linear_fields(sample, editor, {}, None)
    torch.testing.assert_close(destination - anchor, .36 * (target - source))
    assert valid is sample['valid']


def test_stage1_config_matches_checkpoint_recipe():
    from qwen_latent_cot.u15_target_correction.config import load_config, stage_updates
    cfg = load_config('configs/u15_isochronic_state/stage1_full490k.json')
    assert cfg['condition_protocol'] == 'isochronic_condition_v2'
    assert cfg['state_times'] == [0.36]
    assert stage_updates(471112, cfg, 64) == 14723


def test_und_generation_passes_pixels_with_visual_features(monkeypatch):
    import contextlib
    import sys
    import types
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.u15_target_correction import diagnosis

    pixels, features = torch.ones(2), torch.ones(3)
    calls = {}
    class Model:
        template = 'test'
        def _build_t2i_query(self, prompt, **kwargs):
            return prompt
        def extract_feature(self, value, **kwargs):
            assert value is pixels
            return features
        def generate(self, **kwargs):
            calls.update(kwargs)
            return torch.tensor([[1]])
    class Tokenizer:
        def __call__(self, query, return_tensors):
            return {'input_ids': torch.tensor([[1]]), 'attention_mask': torch.tensor([[1]])}
        def convert_tokens_to_ids(self, value):
            return 1
        def batch_decode(self, value, **kwargs):
            return ['a dog']
    conversation = types.ModuleType('sensenova_u1.models.neo_unify.conversation')
    conversation.get_conv_template = lambda template: SimpleNamespace(sep='<end>')
    for name in ('sensenova_u1', 'sensenova_u1.models', 'sensenova_u1.models.neo_unify'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, conversation.__name__, conversation)
    monkeypatch.setattr(diagnosis, '_image_inputs', lambda rt, image: (torch.ones(1), pixels, '<img>'))
    monkeypatch.setattr(diagnosis, 'autocast', lambda device: contextlib.nullcontext())
    rt = SimpleNamespace(model=Model(), tokenizer=Tokenizer(), device='cpu')
    assert diagnosis.ask_with_image(rt, None, 'Describe.', {'und_max_new_tokens': 10}) == 'a dog'
    assert calls['pixel_values'] is pixels
    assert calls['visual_features'] is features


def test_stage1_sampler_replays_after_restoring_state():
    import random
    from qwen_latent_cot.u15_target_correction.data import TrainingSampler
    class Pool:
        count = 3
        def role_pools(self): return {'L_edit': self}
        def choose_time(self, rng, time=None): return rng.random()
        def choose_index(self, time, rng): return rng.randrange(3)
        def sample(self, index, time, keep=False): return {'uid': index, 'time': time}
    cfg = {'seed': 17, 'role_weights': {'L_edit': 1.0}}
    first = TrainingSampler(Pool(), cfg)
    first.draw()
    saved = first.state_dict()
    expected = first.draw()
    restored = TrainingSampler(Pool(), cfg)
    restored.load_state_dict(saved)
    assert restored.draw() == expected
