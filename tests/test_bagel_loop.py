from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_latent_cot.bagel.loop import (
    GENERATION_ATTENTION_PROJECTIONS,
    LoopLoRALinear,
    TEXT_ATTENTION_PROJECTIONS,
    UND_Q_PROJECTIONS,
    inject_loop_lora,
    loop_trainable_names,
)
from qwen_latent_cot.bagel.loop_data import (
    LoopFlowCollator,
    LoopFlowCollatorConfig,
)


def test_semantic_token_trainer_is_removed():
    import qwen_latent_cot.bagel.loop as loop_mod
    import qwen_latent_cot.bagel as bagel_pkg

    assert not hasattr(loop_mod, "BagelCrossStepFlowModule")
    with pytest.raises(AttributeError):
        bagel_pkg.BagelLoopFlowModule


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


def test_generation_lora_is_off_in_read_mode_and_on_in_write_mode():
    layer = LoopLoRALinear(
        _linear(2.0), rank=1, alpha=1, read_enabled=False
    )
    with torch.no_grad():
        layer.lora_A.weight.fill_(1.0)
        layer.lora_B.weight.fill_(1.0)
    inputs = torch.tensor([[1.0, 2.0]])
    base = layer.base_layer(inputs)
    layer.set_loop_mode("read")
    q_read = layer(inputs)
    layer.set_loop_mode("write")
    q_write = layer(inputs)
    assert torch.equal(q_read, base)
    assert not torch.equal(q_write, base)
    assert torch.equal(q_read, torch.tensor([[2.0, 4.0]]))
    assert torch.equal(q_write, torch.tensor([[5.0, 7.0]]))


def test_und_q_lora_read_changes_memory_rows_only():
    layer = LoopLoRALinear(_linear(2.0), rank=1, alpha=1, read_enabled=True)
    with torch.no_grad():
        layer.lora_A.weight.fill_(1.0)
        layer.lora_B.weight.fill_(1.0)
    inputs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    base = layer.base_layer(inputs)
    mask = torch.tensor([False, True, False])
    layer.set_loop_mode("read")
    read = layer.forward_rows(inputs, row_mask=mask)
    assert torch.allclose(read[0], base[0])
    assert torch.allclose(read[2], base[2])
    assert not torch.allclose(read[1], base[1])
    unmasked = layer.forward_rows(inputs, row_mask=None)
    assert torch.equal(unmasked, base)


def test_und_memory_row_mask_selects_memory_indexes_in_und_pack():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import und_memory_row_mask

    und = torch.tensor([10, 11, 12, 13])
    mem = torch.tensor([11, 12])
    mask = und_memory_row_mask(und, mem)
    assert mask.tolist() == [False, True, True, False]
    empty = und_memory_row_mask(und, torch.tensor([], dtype=torch.long))
    assert empty.tolist() == [False, False, False, False]


def test_project_und_queries_scopes_adapter_to_memory_rows():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import project_und_queries

    layer = LoopLoRALinear(_linear(2.0), rank=1, alpha=1, read_enabled=True)
    with torch.no_grad():
        layer.lora_A.weight.fill_(1.0)
        layer.lora_B.weight.fill_(1.0)
    hidden = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    text = torch.tensor([0, 1, 2])
    mem = torch.tensor([1])
    base = layer.base_layer(hidden)
    layer.set_loop_mode("read")
    out = project_und_queries(layer, hidden, text, mem)
    assert torch.allclose(out[0], base[0])
    assert torch.allclose(out[2], base[2])
    assert not torch.allclose(out[1], base[1])
    layer.set_loop_mode("off")
    assert torch.equal(project_und_queries(layer, hidden, text, mem), base)


def test_und_q_lora_write_is_also_memory_rows_only():
    layer = LoopLoRALinear(_linear(2.0), rank=1, alpha=1, read_enabled=True)
    with torch.no_grad():
        layer.lora_A.weight.fill_(1.0)
        layer.lora_B.weight.fill_(1.0)
    inputs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    base = layer.base_layer(inputs)
    mask = torch.tensor([False, True, False])
    layer.set_loop_mode("write")
    written = layer.forward_rows(inputs, row_mask=mask)
    assert torch.allclose(written[0], base[0])
    assert torch.allclose(written[2], base[2])
    assert not torch.allclose(written[1], base[1])


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in UND_Q_PROJECTIONS + GENERATION_ATTENTION_PROJECTIONS + TEXT_ATTENTION_PROJECTIONS:
            if not hasattr(self, name):
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
    assert len(names) == 2 * 2 * 2
    assert all("layers.1." in name or "layers.2." in name for name in names)
    assert not any("layers.0." in name or "layers.3." in name for name in names)
    assert any(".q_proj." in name for name in names)
    assert any(".q_proj_moe_gen." in name for name in names)
    assert not any(".k_proj" in name for name in names)
    assert not any(".v_proj" in name for name in names)
    assert not any(".o_proj" in name for name in names)
    assert not any(".mlp" in name for name in names)


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


def test_optional_gen_o_and_kv_lora_are_explicit():
    model = _Injectable()
    for parameter in model.parameters():
        parameter.requires_grad = False
    inject_loop_lora(
        model,
        start_layer=1,
        end_layer=3,
        rank=2,
        alpha=4,
        gen_attention_o_lora=True,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
    names = loop_trainable_names(
        model, start_layer=1, end_layer=3, gen_attention_o_lora=True
    )
    assert any(".o_proj_moe_gen." in name for name in names)
    assert not any(".k_proj" in name for name in names)

    model_kv = _Injectable()
    for parameter in model_kv.parameters():
        parameter.requires_grad = False
    inject_loop_lora(
        model_kv,
        start_layer=1,
        end_layer=2,
        rank=2,
        alpha=4,
        k_v_lora=True,
    )
    for name, parameter in model_kv.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
    kv_names = loop_trainable_names(model_kv, start_layer=1, end_layer=2, k_v_lora=True)
    assert any(".k_proj." in name for name in kv_names)
    assert any(".v_proj." in name for name in kv_names)
    assert any(".k_proj_moe_gen." in name for name in kv_names)
    assert not any(".o_proj_moe_gen." in name for name in kv_names)


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


def test_memory_body_enables_loop_lora_and_keeps_gradients():
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
    out = model.forward_inference(
        packed_query_sequence=inputs,
        query_lens=torch.tensor([1]),
        packed_query_position_ids=torch.arange(1),
        packed_query_indexes=torch.arange(1),
        update_past_key_values=False,
        is_causal=False,
        packed_memory_token_indexes=torch.tensor([0]),
        memory_loop_repeat=2,
        memory_loop_start=1,
        memory_loop_end=2,
    )
    assert model.layers[1].scale.loop_enabled is False
    out.packed_query_sequence.sum().backward()
    assert model.layers[1].scale.lora_B.weight.grad is not None
    assert model.layers[1].scale.loop_enabled is False


def test_bagel_forward_flow_is_not_globally_no_grad_wrapped():
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel

    # Inference entry points own their no-grad context. A decorator here
    # disconnects every loop SFT/RL loss from its trainable adapters.
    assert not hasattr(Bagel._forward_flow, "__wrapped__")
