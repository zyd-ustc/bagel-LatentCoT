"""Critical interaction contracts on real tiny BAGEL MoT blocks."""
import json
import pytest
import torch
from test_memory_topology import lm, runtime
from test_visual_loop import visual_runtime
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.visual_record import save_visual_checkpoint
from qwen_latent_cot.diagnostics.flow_training import flow_mse, conditional_gen
from qwen_latent_cot.diagnostics.loop_training import memory_loop
from qwen_latent_cot.diagnostics.record import FrozenGuard
from qwen_latent_cot.interaction.student import InteractionStudent
from qwen_latent_cot.interaction import checkpoint
from qwen_latent_cot.interaction.sde import transition, gaussian_kl, advantages, policy_loss


@pytest.fixture
def interaction(visual_runtime, tmp_path):
    runtime, row = visual_runtime
    original = VisualStudent(runtime)
    path = tmp_path/'parent.pt'
    save_visual_checkpoint(path, original, 1000, 'C2-observe')
    parent = torch.load(path, weights_only=True)
    cfg = json.loads(__import__('pathlib').Path('configs/interaction_sft.json').read_text())
    state = runtime.training_state(row, .9, 3)
    with torch.no_grad():
        expected = memory_loop(original, state, 1).exits[0].velocity.clone()
    student = InteractionStudent(runtime, parent, cfg)
    return student, state, expected


def test_zero_parity_gradients_and_scopes(interaction):
    student, state, expected = interaction
    guard = FrozenGuard(student.runtime.model, exclude_trainable=True)
    prompt_before = student.runtime.text('a red token')[0]
    base_before = student.proposal(state).velocity.clone()
    actual, _, _ = student(state)
    torch.testing.assert_close(actual.velocity, expected, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(student.optimizer_groups())
    flow_mse(actual, state).backward()
    for name, parameters in student.parameter_groups().items():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), name
        assert sum(float(p.grad.abs().sum()) for p in parameters) > 0, name
    optimizer.step()
    guard.check()
    torch.testing.assert_close(student.proposal(state).velocity, base_before, rtol=0, atol=0)
    for before, after in zip(prompt_before, student.runtime.text('a red token')[0]):
        torch.testing.assert_close(before.keys, after.keys, rtol=0, atol=0)
    assert all(scope.get() == 'off' for scope in student.adapters.scopes.values())
    # Recompute must enter exactly the same scopes as eager execution.
    gradients = []
    for active in (False, True):
        student.config['activation_checkpointing'] = active
        student.writer.activation_checkpointing = active
        student.zero_grad(set_to_none=True)
        output, _, _ = student(state)
        flow_mse(output, state).backward()
        gradients.append([p.grad.clone() for p in student.adapters.parameters()])
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_checkpoint_reference_restores_both_policies(interaction, tmp_path):
    student, state, _ = interaction
    student.begin_rl()
    with torch.no_grad():
        next(iter(student.adapters.consumer.values())).b.add_(.01)
    path = tmp_path/'step_000001.pt'
    checkpoint.save(path, student, 'rl', 1)
    saved = checkpoint.read(path)
    expected = student(state)[0].velocity.detach()
    reference = student(state, mode='reference')[0].velocity.detach()
    for p in student.adapters.consumer.parameters():
        p.data.zero_()
    checkpoint.apply(saved, student)
    torch.testing.assert_close(student(state)[0].velocity, expected, rtol=0, atol=0)
    torch.testing.assert_close(student(state, mode='reference')[0].velocity, reference, rtol=0, atol=0)
    assert not any(p.requires_grad for p in student.adapters.writer.parameters())
    assert saved['parent']['stage'] == 'C2-observe' and saved['reference']


def test_training_and_deployed_cfg_loop_match(interaction):
    from qwen_latent_cot.diagnostics.shallow_sampling import shallow_loop
    student, state, _ = interaction
    with torch.no_grad():
        for update in (*student.adapters.writer.values(), *student.adapters.consumer.values()):
            update.b.normal_(std=.003)
        trained = student(state, guidance=True)[0].velocity
        with shallow_loop(student, state.flow, state.lengths, state.positions, state.t, 2):
            deployed = student.runtime.velocity(state.x_t, state.t, state.flow)[1]
        torch.testing.assert_close(trained, deployed, rtol=0, atol=0)


def test_interaction_engine_keeps_native_parent_and_consumer_separate(visual_runtime, tmp_path, monkeypatch):
    import copy
    from qwen_latent_cot.interaction.adapters import Adapters
    from qwen_latent_cot.diagnostics.inference import InferenceEngine
    runtime, _ = visual_runtime
    original = VisualStudent(runtime)
    path = tmp_path/'parent.pt'
    save_visual_checkpoint(path, original, 1000, 'C2-observe')
    parent = torch.load(path, weights_only=True)
    config = json.loads(__import__('pathlib').Path('configs/interaction_sft.json').read_text())
    adapters = Adapters(copy.deepcopy(runtime.decoder))
    saved = dict(kind=checkpoint.KIND, stage='sft', step=1, parent=parent,
                 config=config, adapters=adapters.state_dict(), reference={})
    engine = InferenceEngine(runtime, saved)
    monkeypatch.setattr(runtime, 'image', lambda x: x)
    kwargs = dict(early=1, steps=1)
    parent_image, _ = engine.generate('a token', arm='parent_memory', **kwargs)
    actual, _ = engine.generate('a token', arm='interaction_memory', **kwargs)
    torch.testing.assert_close(parent_image, actual, rtol=0, atol=0)
    baseline, _ = engine.generate('a token', arm='bagel', **kwargs)
    with torch.no_grad():
        for update in engine.student.adapters.consumer.values():
            update.b.add_(.03)
    changed, _ = engine.generate('a token', arm='interaction_memory', **kwargs)
    restored, _ = engine.generate('a token', arm='bagel', **kwargs)
    parent_again, _ = engine.generate('a token', arm='parent_memory', **kwargs)
    torch.testing.assert_close(parent_again, parent_image, rtol=0, atol=0)
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
    assert not torch.equal(changed, actual)


def test_sde_replay_gaussian_variance_and_group_semantics():
    torch.manual_seed(2)
    x, velocity = torch.randn(4, 3), torch.randn(4, 3, requires_grad=True)
    sampled = transition(x, velocity, .8, .7, .7, noise=torch.randn_like(x))
    replay = transition(x, velocity, .8, .7, .7, next_sample=sampled.sample)
    torch.testing.assert_close(sampled.score, replay.score, rtol=0, atol=0)
    assert sampled.variance == pytest.approx((.8/.2)*.7**2*.1)
    reference = transition(x, velocity.detach()+1, .8, .7, .7, next_sample=sampled.sample)
    assert gaussian_kl(replay, reference) > 0
    rows = [dict(group=g, candidate=c, reward=float(c) if g == 0 else 1.) for g in range(2) for c in range(8)]
    values, flat = advantages(rows)
    assert flat == .5 and sum(values[(0, c)] for c in range(8)) == pytest.approx(0, abs=1e-6)
    assert all(values[(1, c)] == 0 for c in range(8))
    loss, ratio, _ = policy_loss(replay, sampled.score.detach(), values[(0, 7)], reference, .001, .01)
    assert ratio == 1
    loss.backward()
    assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='eight'):
        advantages(rows[:-1])


def test_optimizer_sampler_rng_resume(interaction, tmp_path):
    from qwen_latent_cot.interaction.data import PairSampler
    student, state, _ = interaction
    optimizer = torch.optim.AdamW(student.optimizer_groups())
    flow_mse(student(state)[0], state).backward()
    optimizer.step()
    sampler = PairSampler(list(range(7)), 5)
    sampler.take(3)
    path = tmp_path/'step_000001.pt'
    provenance = {'train_sha256': 'tiny'}
    checkpoint.save(path, student, 'sft', 1, optimizer=optimizer, sampler=sampler, provenance=provenance)
    saved = checkpoint.read(path)
    expected_draw, expected_noise = sampler.take(7), torch.randn(3)
    checkpoint.apply(saved, student)
    restored = torch.optim.AdamW(student.optimizer_groups())
    resumed = PairSampler(list(range(7)), 999)
    assert checkpoint.resume(saved, student, restored, resumed, provenance) == 1
    assert resumed.take(7) == expected_draw
    torch.testing.assert_close(torch.randn(3), expected_noise, rtol=0, atol=0)
    for original, loaded in zip(optimizer.state.values(), restored.state.values()):
        torch.testing.assert_close(original['exp_avg'], loaded['exp_avg'], rtol=0, atol=0)


def test_training_reward_aliases_count_each_token_once():
    from qwen_latent_cot.interaction.rewards import answer_aliases
    class Tokenizer:
        def encode(self, text):
            return [text.strip().lower()]
    assert answer_aliases(Tokenizer(), 'yes') == ['Yes']
    assert answer_aliases(Tokenizer(), 'no') == ['No']


def test_native_rl_replay_and_fixed_writer_reference(interaction):
    from qwen_latent_cot.interaction.rl import replay, update_metrics
    student, state, _ = interaction
    student.config.update(noise_level=.7, clip=.001, beta=.01)
    student.begin_rl()
    writer = {k: v.clone() for k, v in student.adapters.writer.state_dict().items()}
    with torch.no_grad():
        velocity = student(state, guidance=True)[0].velocity
        sampled = transition(state.x_t, velocity, state.t, .8, .7, noise=torch.randn_like(state.x_t))
    row = dict(x_t=state.x_t, x_s=sampled.sample.detach(), t=state.t, s=.8,
               old_score=sampled.score.detach(), caption=state.metadata['prompt'])
    before = replay(student, row, mode='reference')
    row['reference_mean'] = before.mean.detach()
    current = replay(student, row)
    loss, ratio, _ = policy_loss(current, row['old_score'], 1., before, .001, .01)
    assert float(ratio.detach()) == 1.
    optimizer = torch.optim.AdamW(student.optimizer_groups(), weight_decay=0.)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in student.adapters.consumer.parameters()) > 0
    optimizer.step()
    torch.testing.assert_close(replay(student, row, mode='reference').score, before.score, rtol=0, atol=0)
    for key, value in student.adapters.writer.state_dict().items():
        torch.testing.assert_close(value, writer[key], rtol=0, atol=0)
    assert all(__import__('math').isfinite(v) for v in update_metrics(student, [row]))
