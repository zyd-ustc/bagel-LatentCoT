"""Contract checks for the anchored loop. These do not need a 7B checkpoint."""
from __future__ import annotations

import torch

from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache, Qwen2Config, Qwen2ForCausalLM
from qwen_latent_cot.anchored.adapter import bound_residual
from qwen_latent_cot.anchored.config import LoopConfig
from qwen_latent_cot.anchored.feedback import FeedbackState, intervene_feedback
from qwen_latent_cot.anchored.legacy import refuse_legacy_checkpoint
from qwen_latent_cot.anchored.loop import AnchoredDualStateLoop
from qwen_latent_cot.anchored.loss import anchored_loss
from qwen_latent_cot.anchored.native_path import BranchContext
from qwen_latent_cot.anchored.schedule import in_loop_schedule, selected_steps


def _exact(actual, expected, name: str) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        if actual.shape == expected.shape:
            gap = float((actual.detach().float() - expected.detach().float()).abs().max())
        else:
            gap = -1.0
        raise AssertionError(
            f"{name} diverged max_abs={gap} "
            f"shape={tuple(actual.shape)}/{tuple(expected.shape)} "
            f"dtype={actual.dtype}/{expected.dtype}"
        )


def tiny_language_model():
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, layer_module="Qwen2MoTDecoderLayer",
        qk_norm=True,
    )
    # Native GEN attention casts activations to bf16, so the tiny contract
    # model uses the same dtype as the existing MoT tests.
    return Qwen2ForCausalLM(config).to(torch.bfloat16).eval()


def loop_config(**overrides) -> LoopConfig:
    payload = dict(
        enabled=True, body_start=1, body_end=2, correction_rounds=1, schedule="early",
        schedule_fraction=0.35, entry_scale=0.0, max_delta_rms_ratio=0.02,
        feedback_mode="correct", state_mode="anchored", variant="dual", slots=4, rank=4,
        writer_position_start=2048, lambda_anchor=0.01, lambda_entry=0.001,
        log_intermediates=False, force_full_zero_path=False, verify_prompt_cache=True,
    )
    payload.update(overrides)
    return LoopConfig(**payload)


def make_loop(language_model=None, **overrides) -> AnchoredDualStateLoop:
    language_model = tiny_language_model() if language_model is None else language_model
    loop = AnchoredDualStateLoop(language_model, loop_config(**overrides))
    loop.eval()
    return loop


def pack_case(language_model, samples=1, prefix=0, seed=0, flow_time=0.9, name="cond"):
    torch.manual_seed(seed)
    decoder = language_model.model
    per = 6
    text, vae = [], []
    for sample in range(samples):
        base = sample * per
        text.extend((base, base + per - 1))
        vae.extend(range(base + 1, base + per - 1))
    hidden = torch.randn(samples * per, decoder.config.hidden_size, dtype=decoder.embed_tokens.weight.dtype)
    lengths = (per,) * samples
    positions = torch.cat([torch.arange(per) for _ in range(samples)]).long()
    query, past, offset = [], [], 0
    for length in lengths:
        past.extend(range(offset, offset + prefix))
        query.extend(range(offset + prefix, offset + prefix + length))
        offset += prefix + length
    cache = NaiveCache(len(decoder.layers))
    if prefix:
        prompt = torch.randn(samples * prefix, decoder.config.hidden_size, dtype=decoder.embed_tokens.weight.dtype)
        decoder.forward_inference(
            packed_query_sequence=prompt,
            query_lens=torch.tensor((prefix,) * samples, dtype=torch.int32),
            packed_query_position_ids=torch.arange(prefix).repeat(samples),
            packed_query_indexes=torch.arange(samples * prefix),
            past_key_values=cache,
            key_values_lens=torch.zeros(samples, dtype=torch.int32),
            packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
    branch = BranchContext(
        name=name,
        query_lens=torch.tensor(lengths, dtype=torch.int32),
        packed_query_position_ids=positions,
        packed_query_indexes=torch.tensor(query, dtype=torch.long),
        past_key_values=cache,
        key_values_lens=torch.tensor((prefix,) * samples, dtype=torch.int32),
        packed_key_value_indexes=torch.tensor(past, dtype=torch.long),
        packed_text_indexes=torch.tensor(text, dtype=torch.long),
        packed_vae_token_indexes=torch.tensor(vae, dtype=torch.long),
        flow_time=flow_time,
    )
    return hidden, branch


def native_velocity(language_model, hidden, branch):
    output = language_model.model.forward_inference(
        packed_query_sequence=hidden,
        query_lens=branch.query_lens,
        packed_query_position_ids=branch.packed_query_position_ids,
        packed_query_indexes=branch.packed_query_indexes,
        past_key_values=branch.past_key_values,
        key_values_lens=branch.key_values_lens,
        packed_key_value_indexes=branch.packed_key_value_indexes,
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=branch.packed_vae_token_indexes,
        packed_text_indexes=branch.packed_text_indexes,
    )
    return output.packed_query_sequence[branch.packed_vae_token_indexes]


def check_schedule():
    config = loop_config()
    # t decreases from noise to data. 0.9 is early; 0.1 is late.
    assert in_loop_schedule(config, flow_time=0.9)
    assert not in_loop_schedule(config, flow_time=0.1)
    assert in_loop_schedule(config, step_index=0, num_steps=10)
    assert not in_loop_schedule(config, step_index=9, num_steps=10)
    early = selected_steps("early", 0.35, 10)
    uniform = selected_steps("uniform", 0.35, 10)
    assert len(uniform) == len(early) and len(early) > 0
    assert selected_steps("late", 0.35, 10) != early
    try:
        in_loop_schedule(loop_config(schedule="uniform"), flow_time=0.9)
    except ValueError:
        return
    raise AssertionError("uniform schedule accepted a raw flow time")


def check_span_matches_native():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, prefix=2, seed=3)
    loop = make_loop(language_model, enabled=False)
    with torch.no_grad():
        output = loop.forward_branch(hidden, branch, train=False)
        expected = native_velocity(language_model, hidden, branch)
    _exact(output.velocity, expected, "disabled loop")
    _exact(output.native_velocity, expected, "native velocity")
    assert output.corrected is False
    assert loop.path.body_calls == 1


def check_exact_bypass_and_cache():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, prefix=2, seed=5)
    loop = make_loop(
        language_model, entry_scale=1.0, max_delta_rms_ratio=1.0, force_full_zero_path=True,
    )
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        before_keys = [branch.past_key_values.key_cache[i].clone() for i in range(3)]
        positions = branch.packed_query_position_ids.clone()
        lengths = branch.query_lens.clone()
        output = loop.forward_branch(hidden, branch, train=False)
        expected = native_velocity(language_model, hidden, branch)
    _exact(output.velocity, expected, "full zero path")
    _exact(output.entries[0], output.anchor, "zero residual reentry")
    assert not torch.equal(output.entries[0], output.hidden)
    assert torch.equal(branch.packed_query_position_ids, positions)
    assert torch.equal(branch.query_lens, lengths)
    for index, key in enumerate(before_keys):
        _exact(branch.past_key_values.key_cache[index], key, f"prompt key {index}")
    assert loop.path.body_calls == 2
    text = branch.packed_text_indexes
    _exact(output.entries[0][text], output.anchor[text], "text rows")


def check_fast_bypass_skips_observer():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=1)
    loop = make_loop(language_model, entry_scale=0.0)

    def fail_read(*args, **kwargs):
        raise AssertionError("fast bypass ran the observer")

    loop.observer.read = fail_read
    with torch.no_grad():
        output = loop.forward_branch(hidden, branch, train=False)
        expected = native_velocity(language_model, hidden, branch)
    _exact(output.velocity, expected, "fast bypass")
    assert loop.path.body_calls == 1


def check_schedule_direction_changes_velocity():
    language_model = tiny_language_model()
    early_hidden, early_branch = pack_case(language_model, seed=9, flow_time=0.95)
    late_hidden, late_branch = pack_case(language_model, seed=9, flow_time=0.1)
    loop = make_loop(language_model, entry_scale=1.0, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        early = loop.forward_branch(early_hidden, early_branch, train=False)
        late = loop.forward_branch(late_hidden, late_branch, train=False)
        native = native_velocity(language_model, early_hidden, early_branch)
    if torch.equal(early.velocity, native):
        raise AssertionError("early anchored correction did not change velocity")
    _exact(late.velocity, late.native_velocity, "late step must stay native")
    _exact(early.entries[0][early_branch.packed_text_indexes], early.anchor[early_branch.packed_text_indexes], "prompt-side rows")


def check_rounds_share_the_same_anchor():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=4)
    loop = make_loop(language_model, correction_rounds=2, force_full_zero_path=True, entry_scale=1.0)
    with torch.no_grad():
        output = loop.forward_branch(hidden, branch, train=False)
    assert len(output.entries) == 2
    _exact(output.entries[0], output.anchor, "round 1 anchor")
    _exact(output.entries[1], output.anchor, "round 2 anchor")
    _exact(output.velocity, output.native_velocity, "two zero rounds")


def check_direct_hidden_is_opt_in():
    assert loop_config().state_mode == "anchored"
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=6)
    direct = make_loop(language_model, state_mode="direct_hidden", entry_scale=1.0, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        output = direct.forward_branch(hidden, branch, train=False)
    if torch.equal(output.entries[0], output.anchor):
        raise AssertionError("direct-hidden ablation used the native anchor")


def check_feedback_content_and_gen_only():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, samples=2, seed=8)
    dual = make_loop(language_model, force_full_zero_path=True)
    alone = make_loop(language_model, variant="gen_only", force_full_zero_path=True)
    with torch.no_grad():
        first = dual.forward_branch(hidden, branch, train=False)
        shifted = hidden.clone()
        shifted[branch.packed_vae_token_indexes] += 1
        second = dual.forward_branch(shifted, branch, train=False)
        gen_first = alone.forward_branch(hidden, branch, train=False)
        gen_second = alone.forward_branch(shifted, branch, train=False)
    if torch.allclose(first.feedback.value, second.feedback.value):
        raise AssertionError("dual feedback ignored GEN hidden")
    _exact(gen_first.feedback.value, gen_second.feedback.value, "gen-only feedback")
    correct = make_loop(language_model, feedback_mode="correct", entry_scale=1.0, max_delta_rms_ratio=1.0)
    shuffled = make_loop(language_model, feedback_mode="shuffled", entry_scale=1.0, max_delta_rms_ratio=1.0)
    zeroed = make_loop(language_model, feedback_mode="zero", entry_scale=1.0, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        shuffled.load_state_dict(correct.state_dict())
        zeroed.load_state_dict(correct.state_dict())
        # Memory projection starts at 1e-3. On a bf16 body that scale is below one ulp.
        correct.adapter.m_down.weight.normal_(0, 1.0)
        correct.adapter.up.weight.normal_(0, 0.5)
        shuffled.adapter.m_down.weight.copy_(correct.adapter.m_down.weight)
        shuffled.adapter.up.weight.copy_(correct.adapter.up.weight)
        zeroed.adapter.m_down.weight.copy_(correct.adapter.m_down.weight)
        zeroed.adapter.up.weight.copy_(correct.adapter.up.weight)
        correct_out = correct.forward_branch(hidden, branch, train=False)
        shuffled_out = shuffled.forward_branch(hidden, branch, train=False)
        zero_out = zeroed.forward_branch(hidden, branch, train=False)
    _exact(shuffled_out.feedback.value, correct_out.feedback.value.roll(1, 0), "shuffle is a roll")
    assert shuffled_out.feedback.value.shape == correct_out.feedback.value.shape
    assert shuffled_out.feedback.value.dtype == correct_out.feedback.value.dtype
    _exact(zero_out.feedback.value, torch.zeros_like(zero_out.feedback.value), "zero feedback")
    if torch.equal(shuffled_out.velocity, correct_out.velocity):
        raise AssertionError("shuffled feedback did not change velocity")
    donor = FeedbackState(correct_out.feedback.value, "cfg_text", correct_out.feedback.round_index, correct_out.feedback.flow_time)
    try:
        intervene_feedback(correct_out.feedback, "shuffled", initial=correct_out.feedback, donor=donor)
    except ValueError:
        return
    raise AssertionError("cross-branch donor was accepted")


def check_gradients_and_frozen_native():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=11)
    loop = make_loop(language_model, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
    native = language_model.model.layers[1].self_attn.q_proj.weight.detach().clone()
    leaf = hidden.detach().clone().requires_grad_(True)
    output = loop.forward_branch(leaf, branch, train=True)
    loss, _ = anchored_loss(
        output.velocity, torch.randn_like(output.velocity), output.native_velocity, output.delta,
        output.anchor, branch.packed_vae_token_indexes, lambda_anchor=loop.config.lambda_anchor,
        lambda_entry=loop.config.lambda_entry,
    )
    loss.backward()
    if loop.gate.raw.grad is None or float(loop.gate.raw.grad.abs()) == 0.0:
        raise AssertionError("closed gate received no gradient")
    adapter_grad = loop.adapter.up.weight.grad
    if adapter_grad is not None and float(adapter_grad.abs().sum()) != 0.0:
        raise AssertionError("adapter received a gradient through a zero gate")
    assert language_model.model.layers[1].self_attn.q_proj.weight.grad is None
    optimizer = torch.optim.SGD(loop.trainable_parameters(), lr=1.0)
    optimizer.step()
    _exact(language_model.model.layers[1].self_attn.q_proj.weight, native, "native weight after optimizer")
    names = [name for name, parameter in loop.named_parameters() if parameter.requires_grad]
    if any(name.startswith("layers") or "moe_gen" in name for name in names):
        raise AssertionError("optimizer can see native weights")

    loop.zero_grad(set_to_none=True)
    with torch.no_grad():
        loop.gate.raw.fill_(0.1)
    leaf2 = hidden.detach().clone().requires_grad_(True)
    opened = loop.forward_branch(leaf2, branch, train=True)
    opened.velocity.float().sum().backward()
    if loop.adapter.up.weight.grad is None or float(loop.adapter.up.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("open gate did not train the adapter")
    if loop.updater.up.weight.grad is None or float(loop.updater.up.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("open gate did not train the feedback update")

    dual = make_loop(language_model, force_full_zero_path=False, entry_scale=1.0, max_delta_rms_ratio=1.0)
    leaf3 = hidden.detach().clone().requires_grad_(True)
    feedback = dual.forward_branch(leaf3, branch, train=True).feedback.value
    feedback.sum().backward()
    if leaf3.grad is None or float(leaf3.grad.abs().sum()) == 0.0:
        raise AssertionError("feedback does not depend on GEN hidden")

    gen_only = make_loop(language_model, variant="gen_only", entry_scale=1.0, max_delta_rms_ratio=1.0)
    leaf4 = hidden.detach().clone().requires_grad_(True)
    gen_feedback = gen_only.forward_branch(leaf4, branch, train=True).feedback.value
    gen_feedback.sum().backward()
    if leaf4.grad is not None and float(leaf4.grad.abs().sum()) != 0.0:
        raise AssertionError("gen-only feedback depends on GEN hidden")


def check_cfg_matches_bagel():
    from qwen_latent_cot.bagel.modeling.autoencoder import AutoEncoderParams
    from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel, BagelConfig

    language_model = tiny_language_model()
    vae = AutoEncoderParams(32, 3, 8, 32, 3, [1, 2], 1, 4, 0.3, 0.1)
    bagel = Bagel(
        language_model, None,
        BagelConfig(
            visual_gen=True, visual_und=False, llm_config=language_model.config, vit_config=None,
            vae_config=vae, latent_patch_size=1, max_latent_size=4,
        ),
    ).eval()
    # The real loader sets this. A bare Qwen2Model does not have the attribute.
    bagel.language_model.model.enable_taylorseer = False
    with torch.no_grad():
        bagel.llm2vae.weight.normal_(0, 0.02)
        bagel.llm2vae.bias.normal_(0, 0.02)
    # The decoder is bf16. The velocity head must accept that dtype.
    # The timestep embedder stays float32 because its sinusoid is float32.
    bagel.llm2vae.to(dtype=torch.bfloat16)
    depth = len(language_model.model.layers)
    text_ids = torch.tensor([1, 2])
    text_index = torch.tensor([0, 5])
    vae_index = torch.tensor([1, 2, 3, 4])
    vae_pos = torch.arange(4)
    seqlens = torch.tensor([6], dtype=torch.int32)
    x_t = torch.randn(4, 4)
    timestep = torch.full((4,), 0.8)
    caches = {name: NaiveCache(depth) for name in ("cond", "cfg_text", "cfg_img")}
    indexes = torch.arange(6)
    positions = torch.arange(6)
    empty = torch.zeros(0, dtype=torch.long)
    lens = torch.tensor([0], dtype=torch.int32)

    def branch(name):
        return BranchContext(
            name=name, query_lens=seqlens.clone(), packed_query_position_ids=positions.clone(),
            packed_query_indexes=indexes.clone(), past_key_values=caches[name],
            key_values_lens=lens.clone(), packed_key_value_indexes=empty.clone(),
            packed_text_indexes=text_index.clone(), packed_vae_token_indexes=vae_index.clone(),
            flow_time=0.8,
        )

    loop = make_loop(language_model, enabled=True, entry_scale=0.0)
    with torch.no_grad():
        ours = loop.predict_flow(
            bagel, x_t, timestep, text_ids, text_index, vae_index, vae_pos, seqlens,
            {"cond": branch("cond"), "cfg_text": branch("cfg_text"), "cfg_img": branch("cfg_img")},
            cfg_text_scale=2.0, cfg_img_scale=1.5, cfg_renorm_min=0.0, cfg_renorm_type="global",
            train=False,
        )
        native = bagel._forward_flow(
            x_t=x_t, timestep=timestep, packed_vae_token_indexes=vae_index,
            packed_vae_position_ids=vae_pos, packed_text_ids=text_ids, packed_text_indexes=text_index,
            packed_indexes=indexes, packed_position_ids=positions, packed_seqlens=seqlens,
            key_values_lens=lens, past_key_values=caches["cond"], packed_key_value_indexes=empty,
            cfg_renorm_min=0.0, cfg_renorm_type="global", cfg_text_scale=2.0,
            cfg_text_packed_position_ids=positions, cfg_text_packed_query_indexes=indexes,
            cfg_text_key_values_lens=lens, cfg_text_past_key_values=caches["cfg_text"],
            cfg_text_packed_key_value_indexes=empty, cfg_img_scale=1.5,
            cfg_img_packed_position_ids=positions, cfg_img_packed_query_indexes=indexes,
            cfg_img_key_values_lens=lens, cfg_img_past_key_values=caches["cfg_img"],
            cfg_img_packed_key_value_indexes=empty,
        )
    _exact(ours, native, "cfg bypass versus Bagel._forward_flow")
    try:
        shared = NaiveCache(depth)
        loop.predict_flow(
            bagel, x_t, timestep, text_ids, text_index, vae_index, vae_pos, seqlens,
            {"cond": branch("cond"), "cfg_text": replace_cache(branch("cfg_text"), shared),
             "cfg_img": replace_cache(branch("cfg_img"), shared)},
            cfg_text_scale=2.0, cfg_img_scale=1.5, train=False,
        )
    except ValueError:
        return
    raise AssertionError("two CFG branches shared one cache")


def replace_cache(branch, cache):
    return BranchContext(
        name=branch.name, query_lens=branch.query_lens, packed_query_position_ids=branch.packed_query_position_ids,
        packed_query_indexes=branch.packed_query_indexes, past_key_values=cache,
        key_values_lens=branch.key_values_lens, packed_key_value_indexes=branch.packed_key_value_indexes,
        packed_text_indexes=branch.packed_text_indexes, packed_vae_token_indexes=branch.packed_vae_token_indexes,
        flow_time=branch.flow_time,
    )


def check_checkpoint_and_bound():
    language_model = tiny_language_model()
    loop = make_loop(language_model)
    blob = loop.export_state()
    other = make_loop(language_model)
    with torch.no_grad():
        other.gate.raw.fill_(0.2)
    other.load_state(blob)
    _exact(other.gate.raw, loop.gate.raw, "reloaded gate")
    try:
        refuse_legacy_checkpoint({"experiment": "memory_sft", "model": {"layers.0.weight": torch.zeros(1)}})
    except ValueError:
        pass
    else:
        raise AssertionError("legacy checkpoint was accepted")
    anchor = torch.ones(2, 4)
    delta = torch.zeros_like(anchor)
    delta[1] = 10
    bounded = bound_residual(delta, anchor, 0.02, torch.tensor([1]))
    assert float(bounded[0].abs().sum()) == 0.0
    assert float(bounded[1].float().pow(2).mean().sqrt()) <= 0.02 * float(anchor[1].float().pow(2).mean().sqrt()) + 1e-5


CHECKS = [
    check_schedule,
    check_span_matches_native,
    check_exact_bypass_and_cache,
    check_fast_bypass_skips_observer,
    check_schedule_direction_changes_velocity,
    check_rounds_share_the_same_anchor,
    check_direct_hidden_is_opt_in,
    check_feedback_content_and_gen_only,
    check_gradients_and_frozen_native,
    check_cfg_matches_bagel,
    check_checkpoint_and_bound,
]


def run_self_check() -> list[str]:
    names = []
    for check in CHECKS:
        check()
        names.append(check.__name__.removeprefix("check_"))
    return names
