from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qwen_latent_cot.bagel.inferencer import InterleaveInferencer
from qwen_latent_cot.bagel.loop_distill import (
    delta_velocity_distillation_loss,
    replay_velocity,
    sample_replay_step_indices,
)


def _contexts():
    return {
        name: {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": object(),
        }
        for name in ("full", "text_removed", "image_removed")
    }


class _PrepareModel:
    def __init__(self):
        self.calls = []

    @staticmethod
    def _flow(k):
        return {
            "packed_init_noises": torch.zeros(1, 2),
            "packed_loop_token_indexes": torch.arange(int(k)),
            "packed_vae_seqlens": torch.tensor([1], dtype=torch.int32),
        }

    @staticmethod
    def _cfg(k):
        return {
            "cfg_packed_position_ids": torch.zeros(1, dtype=torch.long),
            "cfg_packed_query_indexes": torch.zeros(1, dtype=torch.long),
            "cfg_key_values_lens": torch.zeros(1, dtype=torch.int32),
            "cfg_packed_key_value_indexes": torch.zeros(1, dtype=torch.long),
            "packed_loop_token_indexes": torch.arange(int(k)),
        }

    def prepare_vae_latent(self, **kwargs):
        self.calls.append(("full", kwargs["num_loop_tokens"]))
        return self._flow(kwargs["num_loop_tokens"])

    def prepare_vae_latent_cfg(self, **kwargs):
        self.calls.append(("cfg", kwargs["num_loop_tokens"]))
        return self._cfg(kwargs["num_loop_tokens"])


def test_same_inferencer_prepares_base_teacher_k0_and_student_k8():
    inferencer = InterleaveInferencer.__new__(InterleaveInferencer)
    inferencer.model = _PrepareModel()
    inferencer.device = torch.device("cpu")
    inferencer.new_token_ids = SimpleNamespace()

    base = inferencer.prepare_velocity_bundle(
        name="base", contexts=_contexts(), image_shape=(64, 64), num_loop_tokens=0
    )
    teacher = inferencer.prepare_velocity_bundle(
        name="teacher", contexts=_contexts(), image_shape=(64, 64), num_loop_tokens=0
    )
    student = inferencer.prepare_velocity_bundle(
        name="student", contexts=_contexts(), image_shape=(64, 64), num_loop_tokens=8
    )

    assert base.flow_input["packed_loop_token_indexes"].numel() == 0
    assert teacher.flow_input["packed_loop_token_indexes"].numel() == 0
    assert student.flow_input["packed_loop_token_indexes"].numel() == 8
    assert inferencer.model.calls == [
        ("full", 0),
        ("cfg", 0),
        ("cfg", 0),
        ("full", 0),
        ("cfg", 0),
        ("cfg", 0),
        ("full", 8),
        ("cfg", 8),
        ("cfg", 8),
    ]


def test_weighted_step_sampler_is_unique_deterministic_and_early_biased():
    first = sample_replay_step_indices(
        29, 4, generator=torch.Generator().manual_seed(17)
    )
    second = sample_replay_step_indices(
        29, 4, generator=torch.Generator().manual_seed(17)
    )
    assert first == second
    assert len(first) == len(set(first)) == 4
    assert all(0 <= index < 29 for index in first)

    counts = [0, 0, 0]
    generator = torch.Generator().manual_seed(23)
    for _ in range(2000):
        index = sample_replay_step_indices(29, 1, generator=generator)[0]
        counts[0 if index < 12 else 1 if index < 22 else 2] += 1
    fractions = torch.tensor(counts, dtype=torch.float32) / sum(counts)
    assert torch.allclose(
        fractions, torch.tensor([0.6, 0.3, 0.1]), atol=0.04
    )


class _ReplayInferencer:
    @staticmethod
    def build_image_velocity_kwargs(**kwargs):
        return {
            "x_t": kwargs["x_t"],
            "timestep": torch.as_tensor(kwargs["timestep"]).reshape(1),
            "within_step_loop_start": None,
            "within_step_loop_end": None,
            "within_step_loop_repeat": 1,
            "within_step_loop_damping": 1.0,
        }


class _ReplayModel:
    def __init__(self):
        self.loop_memory = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        self.calls = []

    def _forward_flow(self, x_t, **kwargs):
        self.calls.append(("base", kwargs))
        return x_t + 1.0

    def _forward_flow_loop(self, x_t, **kwargs):
        self.calls.append(("loop", kwargs))
        return x_t + 2.0, kwargs["loop_memory"], None, None, {}


def _condition(k):
    return SimpleNamespace(
        flow_input={
            "packed_loop_token_indexes": torch.arange(k),
            "packed_vae_seqlens": torch.tensor([2], dtype=torch.int32),
        }
    )


def test_replay_dispatches_k0_to_native_and_k8_to_strict_loop():
    model = _ReplayModel()
    state = {"sample": torch.zeros(2, 2), "timestep": torch.tensor(0.8)}
    base = replay_velocity(model, _ReplayInferencer(), _condition(0), state)
    student = replay_velocity(model, _ReplayInferencer(), _condition(8), state)
    assert torch.equal(base.velocity, torch.ones(2, 2))
    assert torch.equal(student.velocity, torch.full((2, 2), 2.0))
    assert [call[0] for call in model.calls] == ["base", "loop"]
    loop_kwargs = model.calls[1][1]
    assert loop_kwargs["memory_loop_repeat"] == 2
    assert loop_kwargs["memory_loop_start"] == 12
    assert loop_kwargs["memory_loop_end"] == 20
    assert loop_kwargs["round0_memory_write_enabled"] is False


def test_persist_memory_is_detached_before_student_replay():
    model = _ReplayModel()
    memory = torch.ones(8, 2, requires_grad=True)
    state = {
        "sample": torch.zeros(2, 2),
        "timestep": torch.tensor(0.8),
        "m_in": memory,
    }
    replay_velocity(model, _ReplayInferencer(), _condition(8), state)
    assert model.calls[-1][1]["memory_body_in"].requires_grad is False


def test_delta_velocity_loss_backpropagates_only_through_student():
    base = torch.tensor([0.5, -0.5], requires_grad=True)
    teacher = torch.tensor([1.5, 0.5], requires_grad=True)
    student = torch.tensor([1.0, 0.0], requires_grad=True)
    result = delta_velocity_distillation_loss(
        student_velocity=student,
        teacher_velocity=teacher,
        base_velocity=base,
        is_noop=False,
    )
    result.loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None
    assert base.grad is None
    assert result.direction_active is True
    assert torch.isfinite(result.relative_error)


def test_noop_uses_only_restraint_loss():
    base = torch.zeros(2)
    teacher = torch.tensor([4.0, -3.0])
    student = torch.tensor([1.0, -1.0], requires_grad=True)
    result = delta_velocity_distillation_loss(
        student_velocity=student,
        teacher_velocity=teacher,
        base_velocity=base,
        is_noop=True,
        lambda_noop=2.0,
    )
    assert torch.allclose(result.loss, 2.0 * result.noop_loss)
    result.loss.backward()
    assert student.grad is not None


def test_capture_step_contract_rejects_capture_without_trajectory():
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel

    class Tiny:
        language_model = SimpleNamespace(
            model=SimpleNamespace(enable_taylorseer=False)
        )
        config = SimpleNamespace(loop_depth=2)

        @staticmethod
        def prepare_image_schedule(*_args):
            return torch.tensor([1.0]), torch.tensor([1.0])

    common = dict(
        packed_text_ids=torch.zeros(0, dtype=torch.long),
        packed_text_indexes=torch.zeros(0, dtype=torch.long),
        packed_init_noises=torch.zeros(1, 2),
        packed_vae_position_ids=torch.zeros(1, 3, dtype=torch.long),
        packed_vae_token_indexes=torch.zeros(1, dtype=torch.long),
        packed_vae_seqlens=torch.tensor([1], dtype=torch.int32),
        packed_boundary_token_indexes=torch.zeros(0, dtype=torch.long),
        packed_seqlens=torch.tensor([1], dtype=torch.int32),
        packed_position_ids=torch.zeros(1, 3, dtype=torch.long),
        packed_indexes=torch.zeros(1, dtype=torch.long),
        past_key_values=object(),
        key_values_lens=torch.tensor([0], dtype=torch.int32),
        packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
        num_timesteps=2,
        capture_step_indices=(0,),
        return_trajectory=False,
    )
    with pytest.raises(ValueError, match="return_trajectory"):
        Bagel.generate_image.__wrapped__(Tiny(), **common)


def test_selected_euler_state_is_captured_before_its_update():
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel

    class Tiny:
        language_model = SimpleNamespace(
            model=SimpleNamespace(enable_taylorseer=False)
        )
        config = SimpleNamespace(
            loop_depth=2,
            loop_uncond_memory="m0",
            loop_recycle_mode="same_depth",
            loop_memory_persist=False,
            memory_loop_start_layer=12,
            memory_loop_end_layer=20,
            round0_memory_write_enabled=False,
        )

        @staticmethod
        def prepare_image_schedule(*_args):
            return torch.tensor([1.0, 0.5]), torch.tensor([0.5, 0.5])

        @staticmethod
        def predict_image_velocity(x_t, **_kwargs):
            return torch.ones_like(x_t)

        image_euler_step = staticmethod(Bagel.image_euler_step)

    initial = torch.zeros(1, 2)
    latent, states = Bagel.generate_image.__wrapped__(
        Tiny(),
        packed_text_ids=torch.zeros(0, dtype=torch.long),
        packed_text_indexes=torch.zeros(0, dtype=torch.long),
        packed_init_noises=initial,
        packed_vae_position_ids=torch.zeros(1, 3, dtype=torch.long),
        packed_vae_token_indexes=torch.zeros(1, dtype=torch.long),
        packed_vae_seqlens=torch.tensor([1], dtype=torch.int32),
        packed_boundary_token_indexes=torch.zeros(0, dtype=torch.long),
        packed_seqlens=torch.tensor([1], dtype=torch.int32),
        packed_position_ids=torch.zeros(1, 3, dtype=torch.long),
        packed_indexes=torch.zeros(1, dtype=torch.long),
        past_key_values=object(),
        key_values_lens=torch.tensor([0], dtype=torch.int32),
        packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
        num_timesteps=3,
        capture_step_indices=(1,),
        return_trajectory=True,
    )
    assert len(states) == 1
    assert states[0]["kind"] == "velocity_state"
    assert states[0]["step_index"] == 1
    assert torch.equal(states[0]["sample"], torch.full((1, 2), -0.5))
    assert torch.equal(latent[0], torch.full((1, 2), -1.0))
