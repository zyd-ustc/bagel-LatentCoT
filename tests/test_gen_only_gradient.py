"""Regression for the eight-layer GEN-only gradient starvation on real BAGEL."""

from dataclasses import replace

import torch

from test_anchored_t2i_loop import flow_inputs, tiny_model
from qwen_latent_cot.bagel.anchored_loop import configure_stage1, direct_flow_loss
from qwen_latent_cot.bagel.loop_checkpoint import load_loop_checkpoint, save_loop_checkpoint


def test_eight_native_body_layers_preserve_entry_gradient_at_small_gate():
    model, config = tiny_model(0, "gen_only", alpha=0.01, body_layers=8)
    configure_stage1(model)
    model.language_model.model.gradient_checkpointing = True
    config = replace(config, runtime_loop_depth=1)
    inputs = flow_inputs(model)
    target = torch.randn_like(inputs["x_t"])
    norms = []
    for gate in (0.02, 1.0):
        with torch.no_grad():
            model.t2i_loop.gate_logits.fill_(torch.logit(torch.tensor(gate)))
        model.zero_grad(set_to_none=True)
        result = model.forward_t2i_loop(**inputs, loop_config=config)
        torch.testing.assert_close(result.velocity, result.base_velocity, rtol=0, atol=0)
        direct_flow_loss(result, target, config).backward()
        norms.append(model.t2i_loop.reentry.up.bias.grad.norm().item())
        assert all(p.grad is None for name, p in model.named_parameters()
                   if not name.startswith("t2i_loop."))
    # Compare the actual frozen MoT Jacobian, not just whether grad is nonzero.
    # The old reference-reset gate lost ~0.02**8 of the entry gradient.
    assert norms[0] > 1e-5
    assert norms[0] > 0.1 * norms[1]


def test_small_step_training_opens_gen_only_adapter_and_changes_velocity():
    model, config = tiny_model(0, "gen_only", alpha=0.01, body_layers=8)
    configure_stage1(model)
    model.language_model.model.gradient_checkpointing = True
    inputs = flow_inputs(model)
    target = torch.randn_like(inputs["x_t"])
    native = {name: p.detach().clone() for name, p in model.named_parameters()
              if not name.startswith("t2i_loop.")}
    initial_alpha = model.t2i_loop.output_alpha.detach().clone()
    initial_gates = model.t2i_loop.gate_logits.detach().clone()
    optimizer = torch.optim.AdamW(model.t2i_loop.parameters(), lr=1e-4, weight_decay=0)
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        runtime = replace(config, runtime_loop_depth=1 if step < 5 else 2)
        result = model.forward_t2i_loop(**inputs, loop_config=runtime)
        direct_flow_loss(result, target, runtime).backward()
        assert torch.nn.utils.clip_grad_norm_(model.t2i_loop.parameters(), 1).isfinite()
        optimizer.step()
    assert model.t2i_loop.reentry.up.weight.norm() > 1e-3
    assert model.t2i_loop.reentry.up.bias.abs().max() > 1e-3
    assert not torch.equal(model.t2i_loop.output_alpha, initial_alpha)
    assert not torch.equal(model.t2i_loop.gate_logits, initial_gates)
    assert result.stats[-1]["velocity_delta_ratio"] > 1e-5
    for name, p in model.named_parameters():
        if name in native:
            assert p.grad is None
            torch.testing.assert_close(p, native[name], rtol=0, atol=0)


def test_loop_checkpoint_preserves_fp32_updates_when_native_model_is_bf16(tmp_path):
    model, _ = tiny_model(0, "gen_only", alpha=0.01)
    configure_stage1(model)
    with torch.no_grad():
        model.t2i_loop.reentry.up.bias.fill_(0.000123456789)
        model.t2i_loop.gate_logits.add_(0.0001)
    save_loop_checkpoint(model, tmp_path, step=1, model_path="native")
    restored, _ = tiny_model(0, "gen_only", alpha=0.01)
    metadata = load_loop_checkpoint(restored, tmp_path)
    assert metadata["format"] == "umm-t2i-anchored-loop-v4"
    assert metadata["loop_parameter_dtype"] == "torch.float32"
    assert all(p.dtype == torch.float32 for p in restored.t2i_loop.parameters())
    for key, value in model.t2i_loop.state_dict().items():
        torch.testing.assert_close(restored.t2i_loop.state_dict()[key], value, rtol=0, atol=0)
