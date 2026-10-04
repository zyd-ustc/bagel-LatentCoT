"""Reproduce FlashAttention's custom-autograd view contract without CUDA."""

from dataclasses import replace

import pytest
import torch

from test_anchored_t2i_loop import tiny_model, flow_inputs
from qwen_latent_cot.bagel.anchored_loop import configure_stage1, direct_flow_loss
from qwen_latent_cot.bagel.modeling.bagel import qwen2_navit


class CustomAttentionView(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


@pytest.mark.parametrize("mode", ["gen_only", "gen_memory_anchored"])
def test_custom_attention_view_preserves_loop_velocity_and_gradients(monkeypatch, mode):
    model, config = tiny_model(slots=0 if mode == "gen_only" else 2, mode=mode, alpha=0.1)
    configure_stage1(model)
    model.language_model.model.gradient_checkpointing = True
    config = replace(config, runtime_loop_depth=2)
    inputs = flow_inputs(model, batch=2)
    target = torch.randn_like(inputs["x_t"])
    reference = model.forward_t2i_loop(loop_config=config, **inputs)
    reference_loss = direct_flow_loss(reference, target, config)
    reference_loss.backward()
    gradients = {name: parameter.grad.clone() if parameter.grad is not None else None
                 for name, parameter in model.named_parameters() if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    original = qwen2_navit._sdpa_varlen_inference

    def attention_view(**kwargs):
        return CustomAttentionView.apply(original(**kwargs))

    monkeypatch.setattr(qwen2_navit, "_sdpa_varlen_inference", attention_view)
    result = model.forward_t2i_loop(loop_config=config, **inputs)
    loss = direct_flow_loss(result, target, config)
    loss.backward()
    assert torch.equal(result.velocity, reference.velocity)
    assert torch.equal(loss, reference_loss)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
        elif gradients[name] is None:
            assert parameter.grad is None
        else:
            assert torch.isfinite(parameter.grad).all()
            assert torch.equal(parameter.grad, gradients[name])
