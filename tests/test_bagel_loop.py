from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_latent_cot.bagel.loop import (
    BagelCrossStepFlowModule,
    GENERATION_ATTENTION_PROJECTIONS,
    TEXT_ATTENTION_PROJECTIONS,
    LoopLoRALinear,
    inject_loop_lora,
    loop_trainable_names,
)
from qwen_latent_cot.bagel.loop_data import (
    LoopFlowCollator,
    LoopFlowCollatorConfig,
)


class _FlowRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = []

    def _forward_flow(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("return_loop_state"):
            return kwargs["x_t"], kwargs["x_t"]
        return kwargs["x_t"]


def _loop_module(bagel=None, **overrides):
    values = {
        "bagel": bagel or _FlowRecorder(),
        "tokenizer": None,
        "token_ids": None,
        "vae_model": nn.Identity(),
        "loop_start_layer": 1,
        "loop_end_layer": 2,
        "loop_state_scale": 0.2,
        "rollout_steps": 3,
        "timestep_shift": 3.0,
    }
    values.update(overrides)
    return BagelCrossStepFlowModule(**values)


def test_cross_step_module_samples_descending_active_timesteps():
    module = _loop_module(loop_state_timestep_threshold=0.75)
    assert torch.allclose(
        module._shift_timestep(torch.tensor([0.5])), torch.tensor([0.75])
    )
    for _ in range(16):
        rollout = module._sample_rollout_timesteps(torch.device("cpu"))
        assert int(rollout.numel()) == 3
        assert bool(torch.all(rollout[1:] <= rollout[:-1]))
        assert float(rollout.min()) >= 0.75 - 1e-6


def test_cross_step_module_velocity_threads_state():
    bagel = _FlowRecorder()
    module = _loop_module(bagel=bagel)
    layout = {
        key: torch.zeros(1)
        for key in (
            "packed_vae_token_indexes",
            "packed_vae_position_ids",
            "packed_text_ids",
            "packed_text_indexes",
            "packed_boundary_token_indexes",
            "packed_indexes",
            "packed_position_ids",
            "packed_seqlens",
            "key_values_lens",
            "packed_key_value_indexes",
        )
    }
    state = torch.zeros(2, 3)
    module._velocity(
        layout, torch.zeros(1), torch.tensor([0.8]), None, loop_state_in=state
    )
    call = bagel.calls[-1]
    assert call["loop_state_in"] is state
    assert call["loop_state_scale"] == 0.2
    assert call["loop_start_layer"] == 1
    assert call["loop_end_layer"] == 2
    assert call["return_loop_state"] is True


def test_cross_step_module_contract_rejects_invalid_values():
    with pytest.raises(ValueError, match="loop_state_scale"):
        _loop_module(loop_state_scale=-0.1)
    with pytest.raises(ValueError, match="rollout_steps"):
        _loop_module(rollout_steps=0)
    with pytest.raises(ValueError, match="threshold"):
        _loop_module(loop_state_timestep_threshold=1.1)


def test_loop_collator_keeps_prompt_input_separate_from_fix_target(monkeypatch):
    from qwen_latent_cot.bagel import loop_data

    monkeypatch.setattr(loop_data, "load_image", lambda _: object())
    collator = LoopFlowCollator.__new__(LoopFlowCollator)
    collator.cfg = LoopFlowCollatorConfig(latent_downsample=16)
    collator.transform = lambda _: torch.zeros(3, 16, 16)
    batch = collator(
        [
            {
                "prompt": "a red cube beside a blue sphere",
                "edit_instruction": "keep the cube left of the sphere",
                "target_image_path": "unused.png",
            }
        ]
    )
    assert batch["prompts"] == ["a red cube beside a blue sphere"]
    assert batch["semantic_targets"] == ["keep the cube left of the sphere"]
    assert "semantic_texts" not in batch


def _linear(value: float = 1.0) -> nn.Linear:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2) * value)
    return layer


def test_loop_lora_is_disabled_by_default_and_explicitly_gated():
    layer = LoopLoRALinear(_linear(2.0), rank=1, alpha=1)
    with torch.no_grad():
        layer.lora_A.weight.fill_(1.0)
        layer.lora_B.weight.fill_(1.0)
    inputs = torch.tensor([[1.0, 2.0]])
    assert torch.equal(layer(inputs), torch.tensor([[2.0, 4.0]]))
    layer.set_loop_enabled(True)
    assert torch.equal(layer(inputs), torch.tensor([[5.0, 7.0]]))
    layer.set_loop_enabled(False)
    assert torch.equal(layer(inputs), torch.tensor([[2.0, 4.0]]))


def test_loop_lora_keeps_trainable_weights_and_gradients_in_fp32():
    layer = LoopLoRALinear(_linear(2.0).to(torch.bfloat16), rank=1, alpha=1)
    assert layer.lora_A.weight.dtype == torch.float32
    assert layer.lora_B.weight.dtype == torch.float32
    layer.set_loop_enabled(True)
    output = layer(torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert layer.lora_A.weight.grad.dtype == torch.float32
    assert layer.lora_B.weight.grad.dtype == torch.float32


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in GENERATION_ATTENTION_PROJECTIONS:
            setattr(self, name, _linear())
        for name in TEXT_ATTENTION_PROJECTIONS:
            setattr(self, name, _linear())


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Injectable(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([_Layer() for _ in range(4)])
        self.other = nn.Parameter(torch.ones(()))


def test_injection_and_allowlist_are_restricted_to_body_layers():
    model = _Injectable()
    for parameter in model.parameters():
        parameter.requires_grad = False
    inject_loop_lora(model, start_layer=1, end_layer=3, rank=2, alpha=4)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
    names = loop_trainable_names(model, start_layer=1, end_layer=3)
    assert len(names) == 2 * 4 * 2
    assert all("layers.1." in name or "layers.2." in name for name in names)
    assert not any("layers.0." in name or "layers.3." in name for name in names)


def test_allowlist_rejects_a_base_parameter():
    model = _Injectable()
    for parameter in model.parameters():
        parameter.requires_grad = False
    inject_loop_lora(model, start_layer=1, end_layer=2, rank=2, alpha=4)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
    model.other.requires_grad = True
    with pytest.raises(RuntimeError, match="escaped"):
        loop_trainable_names(model, start_layer=1, end_layer=2)


def test_text_kv_lora_is_loop_gated_and_explicitly_allowlisted():
    model = _Injectable()
    for parameter in model.parameters():
        parameter.requires_grad = False
    inject_loop_lora(
        model,
        start_layer=1,
        end_layer=3,
        rank=2,
        alpha=4,
        include_text_kv=True,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
    names = loop_trainable_names(
        model, start_layer=1, end_layer=3, include_text_kv=True
    )
    assert len(names) == 2 * 6 * 2
    assert any(".k_proj." in name for name in names)
    assert any(".v_proj." in name for name in names)


class _TinyRotary(nn.Module):
    def forward(self, hidden, position_ids):
        shape = (1, hidden.shape[0], hidden.shape[-1])
        return hidden.new_ones(shape), hidden.new_zeros(shape)


def _tiny_model(layer_modules):
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Model

    class TinyModel(nn.Module):
        forward_inference = Qwen2Model.forward_inference

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(layer_modules)
            self.rotary_emb = _TinyRotary()
            self.norm = nn.Identity()
            self.use_moe = False
            self.gradient_checkpointing = False

    return TinyModel()


class _AddLayer(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward_inference(self, *, packed_query_sequence, past_key_values, **_):
        return packed_query_sequence + self.value, past_key_values


def _common_args(inputs, vae_indexes, boundary_indexes):
    return dict(
        packed_query_sequence=inputs,
        query_lens=torch.tensor([int(inputs.shape[0])]),
        packed_query_position_ids=torch.arange(int(inputs.shape[0])),
        packed_query_indexes=torch.arange(int(inputs.shape[0])),
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=torch.tensor(vae_indexes),
        packed_text_indexes=torch.tensor(boundary_indexes),
        packed_boundary_token_indexes=torch.tensor(boundary_indexes),
    )


def test_cross_step_state_zero_scale_matches_parity_exactly():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    model = _tiny_model([_AddLayer(1), _AddLayer(2), _AddLayer(4)])
    inputs = torch.zeros(3, 2)
    common = _common_args(inputs, vae_indexes=[1, 2], boundary_indexes=[0])
    parity = model.forward_inference(**common)
    stated = model.forward_inference(
        **common,
        loop_start_layer=1,
        loop_end_layer=2,
        loop_state_in=torch.full((3, 2), 99.0),
        loop_state_scale=0.0,
        return_loop_state=True,
    )
    assert torch.equal(stated.packed_query_sequence, parity.packed_query_sequence)
    assert stated.loop_state_out is not None
    # State positions are VAE + boundary: body entry is 1 everywhere after the
    # prefix, body adds 2, so the state equals 3 on all three positions.
    assert torch.equal(stated.loop_state_out, torch.full((3, 2), 3.0))


def test_cross_step_state_entry_merge_moves_state_positions_only():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    model = _tiny_model([_AddLayer(1), _AddLayer(2), _AddLayer(4)])
    inputs = torch.zeros(3, 2)
    common = _common_args(inputs, vae_indexes=[1, 2], boundary_indexes=[0])
    # body entry after prefix = 1 on every position. Merging the state 11 with
    # scale 1.0 and per-token RMS cap: delta = 10, cap weight = 1 * 1/10 = 0.1,
    # merged entry = 1 + 0.1 * 10 = 2 for every state position.
    out = model.forward_inference(
        **common,
        loop_start_layer=1,
        loop_end_layer=2,
        loop_state_in=torch.full((3, 2), 11.0),
        loop_state_scale=1.0,
    )
    # All three positions are state positions (VAE + boundary), so all follow
    # merged entry 2 -> body 4 -> suffix 8.
    assert torch.equal(out.packed_query_sequence, torch.full((3, 2), 8.0))


def test_cross_step_state_bootstrap_extracts_state_and_matches_parity():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    model = _tiny_model([_AddLayer(1), _AddLayer(2), _AddLayer(4)])
    inputs = torch.zeros(3, 2)
    common = _common_args(inputs, vae_indexes=[1, 2], boundary_indexes=[0])
    parity = model.forward_inference(**common)
    boot = model.forward_inference(
        **common,
        loop_start_layer=1,
        loop_end_layer=2,
        loop_state_in=None,
        loop_state_scale=0.2,
        return_loop_state=True,
    )
    # Bootstrap: no incoming state, so the body runs natively and the output
    # matches parity exactly; only the state is extracted.
    assert torch.equal(boot.packed_query_sequence, parity.packed_query_sequence)
    assert boot.loop_state_out is not None
    assert torch.equal(boot.loop_state_out, torch.full((3, 2), 3.0))


def test_cross_step_state_requires_gen_mode_and_rejects_bad_inputs():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    model = _tiny_model([_AddLayer(1), _AddLayer(2), _AddLayer(4)])
    inputs = torch.zeros(3, 2)
    common = _common_args(inputs, vae_indexes=[1, 2], boundary_indexes=[0])
    with pytest.raises(ValueError, match="only valid in gen mode"):
        model.forward_inference(
            **{**common, "mode": "und"},
            loop_start_layer=1,
            loop_end_layer=2,
            loop_state_in=torch.zeros(3, 2),
            loop_state_scale=0.2,
        )
    with pytest.raises(ValueError, match="loop_state_scale"):
        model.forward_inference(
            **common,
            loop_start_layer=1,
            loop_end_layer=2,
            loop_state_in=torch.zeros(3, 2),
        )
    with pytest.raises(ValueError, match="loop_state_in"):
        model.forward_inference(
            **common,
            loop_start_layer=1,
            loop_end_layer=2,
            loop_state_in=torch.zeros(5, 2),
            loop_state_scale=0.2,
        )


def test_external_state_write_replaces_self_state_at_entry():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    model = _tiny_model([_AddLayer(1), _AddLayer(2), _AddLayer(4)])
    inputs = torch.zeros(3, 2)
    common = _common_args(inputs, vae_indexes=[1, 2], boundary_indexes=[0])
    out = model.forward_inference(
        **common,
        loop_start_layer=1,
        loop_end_layer=2,
        loop_external_state=torch.tensor([[11.0, 11.0]]),
        loop_external_state_scale=1.0,
    )
    # The external write follows the same broadcast + RMS-capped merge, so the
    # result matches the self-state variant at scale 1.0.
    assert torch.equal(out.packed_query_sequence, torch.full((3, 2), 8.0))


def test_state_path_enables_loop_lora_and_keeps_gradients():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    class ScaleLayer(nn.Module):
        def __init__(self, adapter: bool = False):
            super().__init__()
            if adapter:
                self.scale = LoopLoRALinear(_linear(2.0), rank=1, alpha=1)
                with torch.no_grad():
                    self.scale.lora_A.weight.fill_(1.0)
                    self.scale.lora_B.weight.fill_(0.5)
            else:
                self.scale = _linear(2.0)

        def forward_inference(self, *, packed_query_sequence, past_key_values, **_):
            return self.scale(packed_query_sequence), past_key_values

    model = _tiny_model([ScaleLayer(False), ScaleLayer(True), ScaleLayer(False)])
    inputs = torch.tensor([[1.0, 1.0]])
    common = _common_args(inputs, vae_indexes=[0], boundary_indexes=[0])
    out = model.forward_inference(
        **common,
        loop_start_layer=1,
        loop_end_layer=2,
        loop_state_in=torch.full((1, 2), 3.0),
        loop_state_scale=0.5,
    )
    assert model.layers[1].scale.loop_enabled is False
    out.packed_query_sequence.sum().backward()
    assert model.layers[1].scale.lora_B.weight.grad is not None
    assert model.layers[1].scale.loop_enabled is False


def test_bounded_residual_merge_has_exact_zero_alpha_and_tokenwise_cap():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import (
        bounded_residual_merge,
    )

    base = torch.tensor([[1.0, -1.0], [2.0, 2.0]])
    reviewed = torch.tensor([[101.0, 99.0], [-98.0, 102.0]])
    identity = bounded_residual_merge(base, reviewed, residual_scale=0.1, alpha=0.0)
    assert identity.data_ptr() == base.data_ptr()
    assert torch.equal(identity, base)

    merged = bounded_residual_merge(base, reviewed, residual_scale=0.1, alpha=1.0)
    merged_rms = (merged - base).square().mean(dim=-1).sqrt()
    base_rms = base.square().mean(dim=-1).sqrt()
    assert torch.all(merged_rms <= 0.1 * base_rms + 1e-6)


def test_bagel_forward_flow_is_not_globally_no_grad_wrapped():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel

    # Inference entry points own their no-grad context. A decorator here
    # disconnects every loop SFT/RL loss from its trainable adapters.
    assert not hasattr(Bagel._forward_flow, "__wrapped__")
