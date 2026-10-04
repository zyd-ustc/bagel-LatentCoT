from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen_latent_cot.bagel.anchored_loop import (
    AnchorState,
    LoopConfig,
    LoopModules,
    LoopResult,
    configure_stage1,
    direct_flow_loss,
    run_anchored_loop,
)
from qwen_latent_cot.bagel.loop_checkpoint import (
    load_loop_checkpoint,
    save_loop_checkpoint,
)
from qwen_latent_cot.bagel.modeling import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import (
    NaiveCache,
    _sdpa_varlen_inference,
)
from qwen_latent_cot.data.t2i import patchify_latents, sample_flow_state


def tiny_model(slots=2, mode="gen_memory_anchored", alpha=0.0):
    torch.manual_seed(17)
    llm_config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
        pad_token_id=None,
    )
    loop = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=1,
        loop_end_layer=3,
        runtime_loop_depth=3,
        memory_slots=slots,
        loop_mode=mode,
        loop_output_alpha_init=alpha,
        log_loop_stats=True,
    )
    config = BagelConfig(
        llm_config=llm_config,
        visual_und=False,
        vae_config=SimpleNamespace(z_channels=2, downsample=2),
        max_latent_size=8,
        t2i_loop=loop.to_dict(),
    )
    model = Bagel(Qwen2ForCausalLM(llm_config), None, config).to(torch.bfloat16).eval()
    with torch.no_grad():
        nn.init.normal_(model.llm2vae.weight, std=0.1)
    return model, loop


def flow_inputs(model, batch=2):
    lengths = [3 + i for i in range(batch)]
    inputs = model.prepare_vae_latent(
        lengths,
        [5 + i for i in range(batch)],
        [(16, 16)] * batch,
        {"start_of_image": 1, "end_of_image": 2},
    )
    noise = inputs.pop("packed_init_noises").to(torch.bfloat16)
    cache = NaiveCache(4)
    for i in range(4):
        cache.key_cache[i] = torch.randn(sum(lengths), 2, 8, dtype=torch.bfloat16)
        cache.value_cache[i] = torch.randn(sum(lengths), 2, 8, dtype=torch.bfloat16)
    inputs.update(
        x_t=noise,
        timestep=torch.full((len(noise),), 0.6, dtype=torch.bfloat16),
        past_key_values=cache,
    )
    return inputs


def cache_copy(cache):
    return {
        name: {
            key: value.clone() if value is not None else None
            for key, value in getattr(cache, name).items()
        }
        for name in ["key_cache", "value_cache"]
    }


@pytest.mark.parametrize("control", ["disabled", "depth_zero", "alpha_zero"])
@pytest.mark.parametrize("slots", [0, 2, 8, 16])
def test_off_controls_match_native_velocity(control, slots):
    model, config = tiny_model(
        slots, "gen_only" if slots == 0 else "gen_memory_anchored"
    )
    inputs = flow_inputs(model)
    off = replace(config, enable_t2i_loop=False)
    with torch.no_grad():
        model.t2i_loop.config = off
        native = model._forward_flow(**inputs)
        runtime = (
            replace(config, enable_t2i_loop=False)
            if control == "disabled"
            else replace(config, runtime_loop_depth=0)
            if control == "depth_zero"
            else config
        )
        actual = model.forward_t2i_loop(**inputs, loop_config=runtime)
    torch.testing.assert_close(actual.velocity, native, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,slots",
    [
        ("memory_only", 2),
        ("gen_only", 0),
        ("gen_memory_anchored", 2),
        ("direct_native_memory", 2),
        ("direct_native_gen_only", 0),
    ],
)
def test_real_mot_recurrence_keeps_prompt_and_sampler_anchors(mode, slots):
    model, config = tiny_model(slots, mode, alpha=0.3)
    inputs = flow_inputs(model)
    original_cache = cache_copy(inputs["past_key_values"])
    original_x = inputs["x_t"].clone()
    with torch.no_grad():
        result = model.forward_t2i_loop(**inputs)
    assert len(result.velocities) == 3
    assert torch.isfinite(result.velocity).all()
    if mode == "gen_only":
        # With no memory and a zero adapter there is no recurrent correction.
        torch.testing.assert_close(
            result.velocity, result.base_velocity, rtol=0, atol=0
        )
    torch.testing.assert_close(inputs["x_t"], original_x, rtol=0, atol=0)
    for name, values in original_cache.items():
        for index, value in values.items():
            torch.testing.assert_close(
                getattr(inputs["past_key_values"], name)[index], value, rtol=0, atol=0
            )
    assert all(len(log["layers"]) == 2 for log in result.stats)


def test_shared_body_receives_anchored_delta_instead_of_native_exit():
    config = LoopConfig(
        enable_t2i_loop=True,
        loop_start_layer=0,
        loop_end_layer=1,
        runtime_loop_depth=3,
        loop_mode="gen_only",
        memory_slots=0,
        reentry_adapter_type="fixed",
        fixed_reentry_scale=0.1,
        loop_output_alpha_init=1,
    )
    modules = LoopModules(2, config)
    anchor = AnchorState(torch.ones(1, 2, 2), torch.full((1, 2, 2), 10.0))
    entries = []

    def body(entry, memory, modules):
        entries.append(entry.clone())
        return entry + 2, None, []

    run_anchored_loop(anchor, modules, config, body, lambda x: x)
    torch.testing.assert_close(entries[0], anchor.gen_entry)
    torch.testing.assert_close(entries[1], torch.full_like(entries[1], 0.3))
    assert not torch.equal(entries[1], entries[0] + 2)


def test_stage1_can_open_alpha_then_adapter_and_never_updates_native_weights():
    model, config = tiny_model(2)
    names = configure_stage1(model)
    assert names and all(name.startswith("t2i_loop.") for name in names)
    model.language_model.model.gradient_checkpointing = True
    inputs = flow_inputs(model)
    target = torch.randn_like(inputs["x_t"])
    before = {
        name: p.clone()
        for name, p in model.named_parameters()
        if not name.startswith("t2i_loop.")
    }
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.3
    )
    for step in range(3):
        optimizer.zero_grad()
        result = model.forward_t2i_loop(**inputs)
        loss = direct_flow_loss(result, target, config)
        loss.backward()
        assert model.t2i_loop.output_alpha.grad.abs().sum() > 0
        if step > 0:
            assert model.t2i_loop.gate_logits.grad.abs().sum() > 0
            assert model.t2i_loop.reentry.up.weight.grad.abs().sum() > 0
        optimizer.step()
    for name, p in model.named_parameters():
        if name in before:
            assert p.grad is None
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_cfg_branches_have_independent_workspace_and_same_depth(monkeypatch):
    model, config = tiny_model(2, alpha=0.2)
    inputs = flow_inputs(model)
    for name, lens, ropes in [("text", [0, 0], [0, 0]), ("img", [3, 4], [5, 6])]:
        cfg = model.prepare_vae_latent_cfg(lens, ropes, [(16, 16)] * 2)
        inputs.update(
            {
                key.replace("cfg_", f"cfg_{name}_", 1): value
                for key, value in cfg.items()
            }
        )
        cache = NaiveCache(4)
        if name == "img":
            for i in range(4):
                cache.key_cache[i] = inputs["past_key_values"].key_cache[i].clone()
                cache.value_cache[i] = inputs["past_key_values"].value_cache[i].clone()
        inputs[f"cfg_{name}_past_key_values"] = cache
    memories = []
    original = model.t2i_loop.initial_memory

    def record(*args):
        memory = original(*args)
        memories.append(memory)
        return memory

    monkeypatch.setattr(model.t2i_loop, "initial_memory", record)
    with torch.no_grad():
        result = model.forward_t2i_loop(**inputs, cfg_text_scale=4, cfg_img_scale=1.5)
    assert len(memories) == 3
    assert len({memory.data_ptr() for memory in memories}) == 3
    assert {item["branch"] for item in result.stats} == {
        "cond",
        "text_removed",
        "image_removed",
    }
    assert all(
        sum(item["branch"] == branch for item in result.stats) == 3
        for branch in {"cond", "text_removed", "image_removed"}
    )


def test_direct_flow_supervision_uses_clean_target_for_every_round():
    cfg = LoopConfig(loop_ds_weight=0.5)
    target = torch.zeros(2)
    predictions = [torch.full((2,), 1.0), torch.full((2,), 2.0), torch.full((2,), 3.0)]
    result = LoopResult(predictions[-1], torch.zeros(2), predictions, [])
    assert direct_flow_loss(result, target, cfg).item() == pytest.approx(
        9 + 0.5 * (1 + 4) / 2
    )


def test_checkpoint_is_strict_and_contains_only_new_modules(tmp_path):
    model, _ = tiny_model(2, alpha=0.1)
    save_loop_checkpoint(model, tmp_path, step=10, model_path=tmp_path / "base")
    expected = model.t2i_loop.output_alpha.clone()
    with torch.no_grad():
        model.t2i_loop.output_alpha.fill_(9)
    metadata = load_loop_checkpoint(model, tmp_path)
    assert metadata["step"] == 10
    torch.testing.assert_close(model.t2i_loop.output_alpha, expected)
    other, _ = tiny_model(0, "gen_only")
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_loop_checkpoint(other, tmp_path)


def test_flow_sign_and_native_latent_patch_order():
    clean = torch.arange(16).reshape(1, 1, 4, 4).float()
    patch = patchify_latents(clean, 2)
    torch.testing.assert_close(patch[0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
    torch.testing.assert_close(patch[-1], torch.tensor([10.0, 11.0, 14.0, 15.0]))
    noise = torch.ones_like(patch)
    x_t, velocity = sample_flow_state(patch, 0.4, noise)
    torch.testing.assert_close(x_t - 0.4 * velocity, patch)


def test_cached_sdpa_causal_mask_includes_entire_past_prefix():
    query = torch.zeros(1, 2, 4)
    key = torch.zeros(4, 1, 4)
    value = torch.arange(4).float().view(4, 1, 1).expand(-1, -1, 4)
    output = _sdpa_varlen_inference(
        query=query,
        key=key,
        value=value,
        query_lens=torch.tensor([1]),
        key_value_lens=torch.tensor([4]),
        causal=True,
    )
    torch.testing.assert_close(output, torch.full_like(output, 1.5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"runtime_loop_depth": -1},
        {"freeze_prompt_kv_in_loop": False},
        {"loop_mode": "gen_only", "memory_slots": 8},
    ],
)
def test_invalid_config_fails_before_running_backbone(kwargs):
    with pytest.raises(ValueError):
        LoopConfig(**kwargs)


def test_all_zero_loop_gates_restore_native_even_with_nonzero_alpha():
    model, config = tiny_model(2, alpha=0.5)
    with torch.no_grad():
        model.t2i_loop.gate_logits.fill_(-torch.inf)
        result = model.forward_t2i_loop(**flow_inputs(model))
    torch.testing.assert_close(result.velocity, result.base_velocity, rtol=0, atol=0)
    assert result.velocities == []


@pytest.mark.parametrize("renorm", ["global", "channel", "text_channel"])
def test_cfg_alpha_zero_matches_native_combination(renorm):
    model, config = tiny_model(2)
    inputs = flow_inputs(model)
    for branch, lens, ropes in [("text", [0, 0], [0, 0]), ("img", [3, 4], [5, 6])]:
        cfg = model.prepare_vae_latent_cfg(lens, ropes, [(16, 16)] * 2)
        inputs.update(
            {
                key.replace("cfg_", f"cfg_{branch}_", 1): value
                for key, value in cfg.items()
            }
        )
        cache = NaiveCache(4)
        if branch == "img":
            for i in range(4):
                cache.key_cache[i] = inputs["past_key_values"].key_cache[i].clone()
                cache.value_cache[i] = inputs["past_key_values"].value_cache[i].clone()
        inputs[f"cfg_{branch}_past_key_values"] = cache
    with torch.no_grad():
        model.t2i_loop.config = replace(config, enable_t2i_loop=False)
        native = model._forward_flow(
            **inputs, cfg_text_scale=4, cfg_img_scale=1.5, cfg_renorm_type=renorm
        )
        actual = model.forward_t2i_loop(
            **inputs,
            loop_config=config,
            cfg_text_scale=4,
            cfg_img_scale=1.5,
            cfg_renorm_type=renorm,
        )
    torch.testing.assert_close(actual.velocity, native, rtol=0, atol=0)


@pytest.mark.parametrize("control", ["zero", "frozen", "shuffled"])
def test_memory_controls_run_on_real_packed_mot(control):
    model, config = tiny_model(2, alpha=0.3)
    with torch.no_grad():
        result = model.forward_t2i_loop(
            **flow_inputs(model), loop_config=replace(config, memory_control=control)
        )
    assert len(result.velocities) == 3
    assert torch.isfinite(result.velocity).all()
    if control == "zero":
        assert all(item["memory_norm"] == 0 for item in result.stats)
    if control == "frozen":
        assert len({item["memory_norm"] for item in result.stats}) == 1


def test_single_sample_shuffled_control_is_rejected():
    model, config = tiny_model(2, alpha=0.3)
    with pytest.raises(ValueError, match="batch>=2"):
        model.forward_t2i_loop(
            **flow_inputs(model, batch=1),
            loop_config=replace(config, memory_control="shuffled"),
        )
