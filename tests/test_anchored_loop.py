"""Anchored dual-state contract. The heavy checks live in the package self-check."""
import os
from dataclasses import replace

import pytest
import torch

from qwen_latent_cot.anchored.adapter import GATE_WARMUP_SCALE, written_residual
from qwen_latent_cot.anchored.cli import main
from qwen_latent_cot.anchored.feedback import FeedbackStateUpdater
from qwen_latent_cot.anchored.loss import (
    anchored_loss,
    focal_flow_loss,
    improvement_margin_loss,
    native_error_weights,
)
from qwen_latent_cot.anchored.self_check import make_loop, pack_case, run_self_check, tiny_language_model


def test_anchored_contract():
    names = run_self_check()
    assert "exact_bypass_and_cache" in names
    assert "cfg_matches_bagel" in names
    assert "gradients_and_frozen_native" in names
    assert "trained_observation_is_an_independent_copy" in names
    assert "observation_source_mode_follows_the_training_config" in names
    assert "anchored_observation_reuses_native_pre_kv" in names
    assert "image_tokens_address_memory_slots" in names
    assert "inference_reads_the_trained_gate" in names
    assert "written_residual_respects_the_rms_cap" in names
    assert "observation_query_follows_the_verified_writer" in names
    assert "gen_only_does_not_carry_memory" in names
    assert "round_velocities_and_supervision" in names


def test_zero_feedback_state_keeps_the_observation_and_later_magnitude():
    torch.manual_seed(0)
    updater = FeedbackStateUpdater(8, 2, 4)
    observation = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        updater.up.weight.zero_()
    first = updater(updater.initial(2, observation, "cond", 0.4), observation)
    assert first.round_index == 1
    assert torch.equal(first.value, observation)
    carried = (first.value.float() * 5).to(dtype=first.value.dtype)
    second = updater(replace(first, value=carried), observation)
    assert torch.equal(second.value, carried)
    live_observation = torch.randn(2, 2, 8, requires_grad=True)
    live = FeedbackStateUpdater(8, 2, 4)
    output = live(live.initial(2, live_observation, "cond", 0.4), live_observation)
    output.value.sum().backward()
    assert live.up.weight.grad is not None and float(live.up.weight.grad.abs().sum()) > 0.0
    assert live_observation.grad is not None and float(live_observation.grad.abs().sum()) > 0.0


def test_negative_gate_writes_zero_and_does_not_flip_the_warmup():
    anchor = torch.randn(4, 8, dtype=torch.bfloat16)
    vae = torch.tensor([1, 2])
    delta = torch.randn(4, 8, requires_grad=True)
    negative = torch.tensor(-0.2, requires_grad=True)
    written = written_residual(delta, negative, anchor, 1.0, vae)
    assert torch.equal(written, torch.zeros_like(anchor))
    written.float().sum().backward()
    zero_delta = delta.detach().clone().requires_grad_(True)
    zero = torch.tensor(0.0, requires_grad=True)
    zero_written = written_residual(zero_delta, zero, anchor, 1.0, vae)
    assert torch.equal(zero_written, torch.zeros_like(anchor))
    zero_written.float().sum().backward()
    assert torch.allclose(delta.grad, zero_delta.grad)
    assert float(zero_delta.grad.abs().sum()) > 0.0
    assert float(zero.grad.abs()) > 0.0
    assert GATE_WARMUP_SCALE == 1e-3


def test_later_rounds_detach_adapter_hidden_and_keep_feedback_state():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=4)
    loop = make_loop(language_model, correction_rounds=2, max_delta_rms_ratio=1.0)
    seen = []
    real = loop.adapter.forward

    def spy(hidden, feedback, anchor, flow_time, branch):
        seen.append((hidden, feedback.round_index, feedback.value))
        return real(hidden, feedback, anchor, flow_time, branch)

    loop.adapter.forward = spy
    leaf = hidden.detach().clone().requires_grad_(True)
    loop.train()
    output = loop.forward_branch(leaf, branch, train=True)
    assert output.feedback.round_index == 2
    assert [item[1] for item in seen] == [1, 2]
    assert seen[0][0].requires_grad
    assert not seen[1][0].requires_grad
    assert seen[1][2] is not seen[0][2]
    seen.clear()
    loop.eval()
    loop.forward_branch(leaf, branch, train=True)
    assert seen[1][0].requires_grad
    assert [item[1] for item in seen] == [1, 2]


def test_focal_flow_weights_native_errors_and_protects_accurate_tokens():
    native = torch.zeros(4, 2)
    target = torch.zeros(4, 2)
    target[0] = 3.0
    target[1] = 1.0
    focal, protect = native_error_weights(native, target)
    assert float(focal.min()) >= 0.2 and float(focal.max()) <= 3.0
    assert float(protect.min()) >= 0.3 and float(protect.max()) <= 2.0
    assert float(focal[0]) > float(focal[1]) > float(focal[3])
    assert float(protect[0]) < float(protect[3])
    corrected = (target + 0.2).detach().requires_grad_(True)
    focal_flow_loss(corrected, target, native).backward()
    grad = corrected.grad.abs().sum(dim=-1)
    assert float(grad[0]) > float(grad[1]) > float(grad[3])
    anchored, parts = anchored_loss(
        corrected.detach(), target, native, None, None, None,
        lambda_anchor=1.0, lambda_entry=0.0,
    )
    assert parts["anchor"] > 0.0
    assert float(anchored) > float(focal_flow_loss(corrected.detach(), target, native))
    uniform_native = torch.ones(4, 2)
    uniform_target = torch.zeros(4, 2)
    prediction = torch.full((4, 2), 0.3)
    weighted = focal_flow_loss(prediction, uniform_target, uniform_native)
    token_err = (uniform_native - uniform_target).pow(2).mean(dim=-1, keepdim=True)
    scale = token_err / (token_err.mean() + 1e-6)
    expected = (scale * (prediction - uniform_target).pow(2)).mean()
    assert torch.equal(weighted, expected)


def test_improvement_margin_penalizes_only_tokens_that_miss_the_native_error():
    target = torch.zeros(2, 2)
    native = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    one = torch.tensor([[1.0, 0.0]])
    summed = improvement_margin_loss(one, torch.zeros(1, 2), torch.zeros(1, 2), 0.0)
    assert summed.item() == pytest.approx(1.0)
    tied = improvement_margin_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), 0.25)
    assert tied.item() == pytest.approx(0.25)
    worse = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
    loss = improvement_margin_loss(worse, target, native, 0.05)
    assert loss.item() == pytest.approx(1.525)
    loss.backward()
    assert native.grad is None
    assert float(worse.grad[0].abs().sum()) == 0.0
    assert worse.grad[1, 0].item() == pytest.approx(2.0)
    better = torch.zeros(2, 2, requires_grad=True)
    native_error = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    closed = improvement_margin_loss(better, target, native_error, 0.1)
    assert closed.detach().item() == 0.0
    closed.backward()
    assert float(better.grad.abs().sum()) == 0.0
    plain, _parts = anchored_loss(
        worse.detach(), target, native.detach(), None, None, None,
        lambda_anchor=0.0, lambda_entry=0.0, lambda_improve=0.0,
    )
    assert torch.equal(plain, focal_flow_loss(worse.detach(), target, native.detach()))
    hinged, parts = anchored_loss(
        worse.detach(), target, native.detach(), None, None, None,
        lambda_anchor=0.0, lambda_entry=0.0, lambda_improve=1.0, improvement_margin=0.05,
    )
    assert parts["improve"] == pytest.approx(1.525)
    assert float(hinged) == pytest.approx(float(plain) + 1.525)
    with pytest.raises(ValueError, match="margin"):
        improvement_margin_loss(worse.detach(), target, native.detach(), -0.1)
    with pytest.raises(ValueError, match="lambda_improve"):
        anchored_loss(
            worse.detach(), target, native.detach(), None, None, None,
            lambda_anchor=0.0, lambda_entry=0.0, lambda_improve=-1.0,
        )


def test_shuffled_donor_must_match_the_forward_time():
    from dataclasses import replace

    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=3, flow_time=0.9)
    loop = make_loop(language_model, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.gate.raw.fill_(1.0)
    first = loop.forward_branch(hidden, branch, flow_time=0.9, train=False)
    assert first.feedback is not None
    loop.config.feedback_mode = "shuffled"
    second = loop.forward_branch(
        hidden, branch, flow_time=0.9, train=False, feedback_donor=first.feedback,
    )
    assert second.corrected
    bad = replace(first.feedback, flow_time=0.2)
    with pytest.raises(ValueError, match="flow time"):
        loop.forward_branch(hidden, branch, flow_time=0.9, train=False, feedback_donor=bad)


def test_show_config():
    main(["show-config", "--config", "configs/anchored_dual_state_loop.json"])


def test_legacy_training_entries_are_locked(monkeypatch):
    monkeypatch.delenv("BAGEL_ALLOW_LEGACY_MEMORY_SFT", raising=False)
    from qwen_latent_cot.diagnostics.visual_cli import main as visual_main
    from qwen_latent_cot.interaction.cli import main as interaction_main
    with pytest.raises(RuntimeError, match="retired"):
        visual_main(["train", "--model", "unused", "--output", "unused"])
    with pytest.raises(RuntimeError, match="retired"):
        interaction_main(["sft", "--model", "unused", "--parent", "unused", "--train", "a", "--val", "b", "--output", "c"])
    monkeypatch.setenv("BAGEL_ALLOW_LEGACY_MEMORY_SFT", "1")
    from qwen_latent_cot.anchored.legacy import refuse_legacy_memory_sft
    refuse_legacy_memory_sft("visual-train")
    os.environ.pop("BAGEL_ALLOW_LEGACY_MEMORY_SFT", None)
