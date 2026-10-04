import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_anchored_t2i_loop import cache_copy, flow_inputs, tiny_model

from qwen_latent_cot.bagel.anchored_loop import (
    configure_stage1,
    direct_flow_loss,
    memory_slot_stats,
)
from qwen_latent_cot.bagel.flow_time import (
    sample_native_flow_timestep,
    shift_flow_timestep,
)
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache


def cfg_inputs(model, inputs, batch):
    for name, lens, ropes in [
        ("text", [0] * batch, [0] * batch),
        ("img", [3 + i for i in range(batch)], [5 + i for i in range(batch)]),
    ]:
        cfg = model.prepare_vae_latent_cfg(lens, ropes, [(16, 16)] * batch)
        inputs.update(
            {
                key.replace("cfg_", f"cfg_{name}_", 1): value
                for key, value in cfg.items()
            }
        )
        kv = NaiveCache(4)
        if name == "img":
            values = cache_copy(inputs["past_key_values"])
            kv.key_cache, kv.value_cache = values["key_cache"], values["value_cache"]
        inputs[f"cfg_{name}_past_key_values"] = kv


def test_legacy_velocity_matches_frozen_parent_same_weights_and_seed(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    with tarfile.open(fixtures / "current_memloop_parent.tar.gz") as archive:
        archive.extractall(tmp_path / "parent", filter="data")
    cases, actual = [], []
    # Include zero new gates/alpha: legacy must still execute the original loop.
    for slots, depth, batch, text_scale, img_scale, renorm, seed in [
        (8, 1, 1, 1.0, 1.0, "global", 11),
        (16, 2, 2, 4.0, 2.0, "global", 12),
        (8, 2, 2, 4.0, 2.0, "channel", 13),
        (16, 1, 2, 4.0, 2.0, "text_channel", 14),
        (8, 1, 2, 1.0, 2.0, "global", 15),
    ]:
        model, config = tiny_model(slots, "legacy_memory_only", alpha=0)
        model.t2i_loop.config = replace(
            config, runtime_loop_depth=depth, log_loop_stats=False
        )
        torch.manual_seed(seed)
        inputs = flow_inputs(model, batch)
        cfg_inputs(model, inputs, batch)
        inputs.update(
            cfg_text_scale=text_scale,
            cfg_img_scale=img_scale,
            cfg_renorm_type=renorm,
            cfg_renorm_min=0.5,
        )
        with torch.no_grad():
            model.t2i_loop.gate_logits.fill_(-torch.inf)
            result = model._forward_flow(**inputs)
            # Trained workspace and adapters cannot alter the historical control.
            model.t2i_loop.memory_init.fill_(9)
            model.t2i_loop.output_alpha.fill_(7)
            model.t2i_loop.reentry.up.bias.fill_(3)
            again = model.forward_t2i_loop(**inputs).velocity
        torch.testing.assert_close(again, result, atol=0, rtol=0)
        actual.append(result)
        cases.append(
            dict(
                seed=seed,
                slots=slots,
                depth=depth,
                batch=batch,
                llm_config=model.config.llm_config.to_dict(),
                weights={
                    k: v
                    for k, v in model.state_dict().items()
                    if not k.startswith("t2i_loop.")
                },
                x_t=inputs["x_t"],
                timestep=inputs["timestep"],
                cache=cache_copy(inputs["past_key_values"]),
                text_cache=cache_copy(inputs["cfg_text_past_key_values"]),
                img_cache=cache_copy(inputs["cfg_img_past_key_values"]),
                text_scale=text_scale,
                img_scale=img_scale,
                renorm=renorm,
            )
        )
    source, target = tmp_path / "cases.pt", tmp_path / "velocities.pt"
    torch.save(cases, source)
    run = subprocess.run(
        [
            sys.executable,
            str(fixtures / "run_current_memloop_parent.py"),
            str(tmp_path / "parent"),
            str(source),
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    expected = torch.load(target, weights_only=True)
    for compatibility, parent in zip(actual, expected):
        torch.testing.assert_close(compatibility, parent, atol=0, rtol=0)


@pytest.mark.parametrize("slots", [8, 16])
def test_loaded_boundary_memory_breaks_slot_symmetry_and_reports_rank(slots):
    model, config = tiny_model(slots)
    with torch.no_grad():
        # Exercise small embedding values with real BF16 rounding.
        model.language_model.model.embed_tokens.weight.mul_(0.02)
    modules = model.t2i_loop
    rng_before = torch.get_rng_state().clone()
    modules.initialize_memory_from_boundaries(
        model.language_model.model.embed_tokens.weight, [1, 2]
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    assert torch.unique(modules.memory_init, dim=0).shape[0] == slots
    stats = memory_slot_stats(modules.initial_memory(2))
    assert stats["effective_rank"] > 2
    assert stats["slot_variation_norm"] > 0
    initial = modules.memory_init.clone()
    modules.initialize_memory_from_boundaries(
        model.language_model.model.embed_tokens.weight, [1, 2]
    )
    torch.testing.assert_close(initial, modules.memory_init, rtol=0, atol=0)
    collapsed = torch.ones(2, slots, 32)
    assert memory_slot_stats(collapsed)["effective_rank"] == 0
    assert memory_slot_stats(collapsed)["mean_abs_pairwise_cosine"] == pytest.approx(1)
    with torch.no_grad():
        result = model.forward_t2i_loop(**flow_inputs(model))
    assert all(
        row["memory_slot_stats"]["slot_variation_norm"] > 0 for row in result.stats
    )


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_timestep_draw_is_exact_native_logit_normal_then_shift(shift):
    expected_raw = torch.randn(4096, generator=torch.Generator().manual_seed(91))
    expected = expected_raw.sigmoid()
    expected = shift * expected / (1 + (shift - 1) * expected)
    sampled = sample_native_flow_timestep(
        (4096,), generator=torch.Generator().manual_seed(91), timestep_shift=shift
    )
    torch.testing.assert_close(sampled, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        sample_native_flow_timestep(raw_t=expected_raw, timestep_shift=shift),
        expected,
        rtol=0,
        atol=0,
    )
    assert sampled.min() > 0 and sampled.max() < 1
    assert shift_flow_timestep(torch.tensor([0.0, 1.0]), shift).tolist() == [0.0, 1.0]


def test_gate_preserves_native_update_and_scales_only_gen_correction():
    model, config = tiny_model()
    native = torch.randn(2, 4, 32)
    correction = torch.randn_like(native)
    actual = model.t2i_loop.gate(0, native, native + correction)
    torch.testing.assert_close(
        actual, native + model.t2i_loop.gate_logits[0].sigmoid() * correction
    )
    with torch.no_grad():
        model.t2i_loop.gate_logits[0] = -torch.inf
    torch.testing.assert_close(
        model.t2i_loop.gate(0, native, native + correction), native, rtol=0, atol=0
    )


def test_gen_only_initial_velocity_is_native_but_zero_bias_can_train():
    model, config = tiny_model(0, "gen_only", alpha=0.01)
    configure_stage1(model)
    inputs = flow_inputs(model)
    result = model.forward_t2i_loop(**inputs)
    torch.testing.assert_close(result.velocity, result.base_velocity, rtol=0, atol=0)
    direct_flow_loss(result, torch.randn_like(inputs["x_t"]), config).backward()
    assert model.t2i_loop.reentry.up.bias.grad.abs().sum() > 0


def test_native_training_forward_uses_the_shared_time_transform(monkeypatch):
    model, _ = tiny_model()
    model.timestep_shift = 3.0
    captured = []
    handle = model.time_embedder.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].clone())
    )
    monkeypatch.setattr(
        model.language_model, "forward", lambda **kwargs: kwargs["packed_sequence"]
    )
    raw = torch.linspace(-3, 3, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward(
            sequence_length=18,
            packed_text_ids=torch.tensor([1, 2]),
            packed_text_indexes=torch.tensor([0, 17]),
            sample_lens=[18],
            packed_position_ids=torch.zeros(18, dtype=torch.long),
            nested_attention_masks=[torch.zeros(18, 18)],
            padded_latent=torch.randn(1, 2, 8, 8),
            patchified_vae_latent_shapes=[(4, 4)],
            packed_latent_position_ids=torch.arange(16),
            packed_vae_token_indexes=torch.arange(1, 17),
            packed_timesteps=raw,
            mse_loss_indexes=torch.arange(1, 17),
        )
    handle.remove()
    torch.testing.assert_close(
        captured[0],
        sample_native_flow_timestep(raw_t=raw, timestep_shift=3),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(output["mse"]).all()


def test_real_body_gates_only_gen_and_leaves_memory_full_update(monkeypatch):
    model, config = tiny_model(2, alpha=0.5)
    calls, gated_shapes = [], []
    layer = model.language_model.model.layers[2]
    original_layer, original_gate = layer.forward_inference, model.t2i_loop.gate

    def record_layer(*args, **kwargs):
        out = original_layer(*args, **kwargs)
        calls.append((kwargs["packed_query_sequence"].clone(), out[0].clone()))
        return out

    def record_gate(offset, reference, current):
        gated_shapes.append(reference.shape)
        return original_gate(offset, reference, current)

    monkeypatch.setattr(layer, "forward_inference", record_layer)
    monkeypatch.setattr(model.t2i_loop, "gate", record_gate)
    with torch.no_grad():
        result = model.forward_t2i_loop(
            **flow_inputs(model), loop_config=replace(config, runtime_loop_depth=1)
        )
    assert gated_shapes == [torch.Size([32, 32]), torch.Size([32, 32])]
    # Query order per sample is 18 original rows + two trailing memory slots.
    before, after = calls[-1]
    memory_indexes = torch.tensor([18, 19, 38, 39])
    assert not torch.equal(before[memory_indexes], after[memory_indexes])
    expected_norm = float(after[memory_indexes].float().norm())
    assert result.stats[0]["memory_norm"] == pytest.approx(expected_norm)
    assert result.stats[0]["layers"][0]["native_transform_ratio"] > 0


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_corrected_checkpoint_rejects_old_gate_or_per_depth_alpha(tmp_path, version):
    import json

    from qwen_latent_cot.bagel.loop_checkpoint import (
        checkpoint_config,
        save_loop_checkpoint,
    )

    model, _ = tiny_model()
    save_loop_checkpoint(model, tmp_path, step=0, model_path="native")
    path = tmp_path / "loop.json"
    metadata = json.loads(path.read_text())
    assert (
        metadata["gate_semantics"] == "native_gen_reference_plus_gated_loop_correction"
    )
    metadata["format"] = f"umm-t2i-anchored-loop-{version}"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible"):
        checkpoint_config(tmp_path)


def test_gen_only_learned_bias_changes_velocity_with_full_native_transform():
    model, config = tiny_model(0, "gen_only", alpha=1)
    inputs = flow_inputs(model)
    with torch.no_grad():
        model.t2i_loop.gate_logits.zero_()  # g=0.5 for a visible BF16 correction.
        model.t2i_loop.reentry.up.bias.fill_(0.5)
        result = model.forward_t2i_loop(**inputs)
    assert not torch.equal(result.velocity, result.base_velocity)
    assert result.stats[0]["layers"][0]["raw_gen_correction_ratio"] > 0
