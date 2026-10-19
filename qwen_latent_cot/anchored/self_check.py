"""Contract checks for the anchored loop. These do not need a 7B checkpoint."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache, Qwen2Config, Qwen2ForCausalLM
from qwen_latent_cot.anchored.adapter import (
    GATE_WARMUP_SCALE,
    AnchoredReentryAdapter,
    EntryGate,
    bound_residual,
    slot_attention,
    written_residual,
)
from qwen_latent_cot.anchored.config import SLOT_HEADS, LoopConfig
from qwen_latent_cot.anchored.diagnostics import round_velocity_report
from qwen_latent_cot.anchored.feedback import (
    OUTPUT_INIT_STD,
    PROJECTION_INIT_STD,
    FeedbackState,
    FeedbackStateUpdater,
    intervene_feedback,
)
from qwen_latent_cot.anchored.legacy import refuse_legacy_checkpoint
from qwen_latent_cot.anchored.loop import AnchoredDualStateLoop
from qwen_latent_cot.anchored.loss import anchored_loss
from qwen_latent_cot.anchored.native_path import BranchContext
from qwen_latent_cot.anchored.observer import observation_slots
from qwen_latent_cot.anchored.schedule import in_loop_schedule, selected_steps
from qwen_latent_cot.diagnostics.visual import observation_readout_keys


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
        # Keep the early prefix here. Contract forwards use t=0.9, and the
        # shipped default interval leaves that noise step on the native pass.
        enabled=True, body_start=1, body_end=2, correction_rounds=1, schedule="early",
        schedule_fraction=0.35, schedule_start=0.15, schedule_end=0.65, max_delta_rms_ratio=0.02,
        observer="trained_observation", allow_unloaded_observer=True, seed_token_ids=(1, 2, 3, 4),
        feedback_mode="correct", state_mode="anchored", variant="dual", slots=4, rank=4,
        writer_position_start=2048, lambda_anchor=0.01, lambda_entry=0.001,
        log_intermediates=False, force_native_bypass=False, force_full_zero_path=False,
        verify_prompt_cache=True,
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


def check_default_interval_skips_pure_noise():
    """Shipped configs correct u in [0.15, 0.65]. t > 0.85 stays native."""
    bare = LoopConfig()
    if (bare.schedule, bare.schedule_start, bare.schedule_end) != ("interval", 0.15, 0.65):
        raise AssertionError("default schedule left the mid-early interval")
    if bare.body_start != 20 or bare.body_end != 28:
        raise AssertionError("default body span drifted")
    root = Path(__file__).resolve().parents[2]
    for name in ("anchored_dual_state_loop.json", "anchored_gen_only.json"):
        loaded = LoopConfig.from_json(root / "configs" / name)
        if (loaded.schedule, loaded.schedule_start, loaded.schedule_end) != ("interval", 0.15, 0.65):
            raise AssertionError(f"{name} left the mid-early interval")
        if loaded.body_start != 20 or loaded.body_end != 28:
            raise AssertionError(f"{name} body span drifted")
    window = LoopConfig(schedule="interval", schedule_start=0.15, schedule_end=0.65, enabled=True)
    assert in_loop_schedule(window, flow_time=0.85)
    assert in_loop_schedule(window, flow_time=0.6)
    assert in_loop_schedule(window, flow_time=0.35)
    assert not in_loop_schedule(window, flow_time=0.95)
    assert not in_loop_schedule(window, flow_time=0.2)
    assert in_loop_schedule(window, step_index=3, num_steps=21)
    assert in_loop_schedule(window, step_index=13, num_steps=21)
    assert not in_loop_schedule(window, step_index=0, num_steps=21)
    assert not in_loop_schedule(window, step_index=14, num_steps=21)
    try:
        LoopConfig(schedule_start=0.7, schedule_end=0.2, body_start=0, body_end=1).validate(1)
    except ValueError as error:
        if "interval" not in str(error):
            raise
    else:
        raise AssertionError("a reversed schedule interval was accepted")

    language_model = tiny_language_model()
    noise_hidden, noise_branch = pack_case(language_model, seed=9, flow_time=0.95)
    mid_hidden, mid_branch = pack_case(language_model, seed=9, flow_time=0.6)
    loop = make_loop(language_model, schedule="interval", max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        loop.gate.raw.fill_(1.0)
        noise = loop.forward_branch(noise_hidden, noise_branch, train=False)
        mid = loop.forward_branch(mid_hidden, mid_branch, train=False)
        native = native_velocity(language_model, mid_hidden, mid_branch)
    _exact(noise.velocity, noise.native_velocity, "pure noise stays on the native pass")
    if torch.equal(mid.velocity, native):
        raise AssertionError("mid-early interval did not run the correction")


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
        language_model, max_delta_rms_ratio=1.0, force_full_zero_path=True,
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
    loop = make_loop(language_model)

    def fail_read(*args, **kwargs):
        raise AssertionError("fast bypass ran the observer")

    loop.observer.read = fail_read
    with torch.no_grad():
        output = loop.forward_branch(hidden, branch, train=False)
        expected = native_velocity(language_model, hidden, branch)
    _exact(output.velocity, expected, "fast bypass")
    assert loop.path.body_calls == 1


def check_inference_reads_the_trained_gate():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=13)
    loop = make_loop(language_model, max_delta_rms_ratio=1.0)
    try:
        LoopConfig.from_dict({"entry_scale": 0.0, **loop.config.to_dict()})
    except ValueError as error:
        if "entry_scale" not in str(error):
            raise
    else:
        raise AssertionError("entry_scale was still accepted")
    try:
        make_loop(language_model, force_native_bypass=True, force_full_zero_path=True)
    except ValueError as error:
        if "force_native_bypass" not in str(error):
            raise
    else:
        raise AssertionError("native bypass and the full zero path were both accepted")

    skipped = make_loop(language_model, force_native_bypass=True, max_delta_rms_ratio=1.0)
    zero_path = make_loop(language_model, force_full_zero_path=True, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        loop.gate.raw.fill_(0.4)
        loop.path.body_calls = 0
        low = loop.forward_branch(hidden, branch, train=False)
        native = native_velocity(language_model, hidden, branch)
        if loop.path.body_calls != 2:
            raise AssertionError("inference with an open gate skipped the correction")
        if torch.equal(low.velocity, native):
            raise AssertionError("inference ignored a trained gate of 0.4")
        loop.gate.raw.fill_(0.8)
        high = loop.forward_branch(hidden, branch, train=False)
        if torch.equal(high.velocity, native) or torch.equal(low.velocity, high.velocity):
            raise AssertionError("inference gate scale did not change velocity")

        skipped.load_state_dict(loop.state_dict())
        skipped.path.body_calls = 0
        bypass = skipped.forward_branch(hidden, branch, train=False)
        train_skip = skipped.forward_branch(hidden, branch, train=True)
    _exact(bypass.velocity, native, "native bypass ignores the trained gate")
    _exact(train_skip.velocity, native, "native bypass during training")
    if skipped.path.body_calls != 2:
        raise AssertionError("force_native_bypass still ran a correction")
    with torch.no_grad():
        zero_path.load_state_dict(loop.state_dict())
        zero_path.path.body_calls = 0
        zeroed = zero_path.forward_branch(hidden, branch, train=False)
    _exact(zeroed.velocity, native, "full zero path ignores the trained gate")
    _exact(zeroed.entries[0], zeroed.anchor, "full zero path writes the anchor")
    if zero_path.path.body_calls != 2:
        raise AssertionError("force_full_zero_path skipped the second body")


def check_schedule_direction_changes_velocity():
    language_model = tiny_language_model()
    early_hidden, early_branch = pack_case(language_model, seed=9, flow_time=0.95)
    late_hidden, late_branch = pack_case(language_model, seed=9, flow_time=0.1)
    loop = make_loop(language_model, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        loop.gate.raw.fill_(1.0)
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
    loop = make_loop(language_model, correction_rounds=2, force_full_zero_path=True)
    with torch.no_grad():
        output = loop.forward_branch(hidden, branch, train=False)
    assert len(output.entries) == 2
    _exact(output.entries[0], output.anchor, "round 1 anchor")
    _exact(output.entries[1], output.anchor, "round 2 anchor")
    _exact(output.velocity, output.native_velocity, "two zero rounds")


def check_trained_observation_is_an_independent_copy():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=5)
    refused = AnchoredDualStateLoop(
        language_model, loop_config(allow_unloaded_observer=False, force_full_zero_path=True),
    )
    try:
        refused.forward_branch(hidden, branch, train=False)
    except RuntimeError as error:
        if "observation checkpoint" not in str(error):
            raise
    else:
        raise AssertionError("dual state read the native UND copy without a checkpoint")
    gen_only = AnchoredDualStateLoop(
        language_model, loop_config(allow_unloaded_observer=False, variant="gen_only", force_full_zero_path=True),
    )
    with torch.no_grad():
        gen_only.forward_branch(hidden, branch, train=False)

    loop = make_loop(language_model, force_full_zero_path=True)
    decoder = language_model.model
    observer = loop.observer
    native_q = decoder.layers[0].self_attn.q_proj.weight.detach().clone()
    if observer.layers[0].self_attn.q_proj.weight.data_ptr() == native_q.data_ptr():
        raise AssertionError("observation Q projection aliases the native decoder")
    if observer.layers[0].mlp.down_proj.weight.data_ptr() != decoder.layers[0].mlp.down_proj.weight.data_ptr():
        raise AssertionError("untrained MLP was copied away from the native decoder")
    embeddings = loop.path.rope(hidden, branch.packed_query_position_ids)
    captured = {}
    loop.path.pre(hidden, branch, embeddings, captured)
    loop.path.body(hidden, branch, embeddings, captured)
    loop.path.post(hidden, branch, embeddings, captured)
    with torch.no_grad():
        copied = observer.read(captured, branch)
        native = observation_slots(
            decoder.layers, decoder.embed_tokens, decoder, observer.seed_token_ids, captured,
            branch, observer.slots, observer.position_start,
        )
    _exact(copied, native, "unloaded copy matches the native readout")

    und = {}
    for key in observation_readout_keys(observer.layers):
        layer_index = int(key.split(".")[1])
        local = key[len(f"layers.{layer_index}."):]
        und[key] = observer.layers[layer_index].get_parameter(local).detach().cpu().float() + 0.25
    path = Path("/tmp/anchored_observation_format4.pt")
    # C2-observe forces the observe task, so the endpoint weight below is unused.
    torch.save(_observation_blob(
        und, stage="C2-observe", depth=len(decoder.layers),
        config=_training_config("C2-observe", len(decoder.layers), task_mix={
            "observe": 0.15, "endpoint": 0.85, "generation": 0.0,
            "trajectory": 0.0, "conditional": 0.0,
        }),
    ), path)
    bad = dict(torch.load(path, map_location="cpu", weights_only=True))
    bad["und"] = dict(bad["und"])
    bad["und"]["layers.0.self_attn.q_proj_moe_gen.weight"] = und["layers.0.self_attn.q_proj.weight"]
    bad_path = Path("/tmp/anchored_observation_moe.pt")
    torch.save(bad, bad_path)
    try:
        observer.load_checkpoint(bad_path)
    except ValueError as error:
        if "readout" not in str(error):
            raise
    else:
        raise AssertionError("observation loader accepted a GEN expert tensor")
    observer.load_checkpoint(path)
    status = observer.status()
    if status["source_mode"] != "gen_full_query" or status["trained_sources"] != ["gen_full_query"]:
        raise AssertionError("C2-observe training tasks did not resolve to gen_full_query")
    if not status["sha256"] or status["checkpoint"] != str(path):
        raise AssertionError("observation provenance omitted the checkpoint path or hash")
    _exact(decoder.layers[0].self_attn.q_proj.weight, native_q, "native Q after observation load")
    if torch.equal(observer.layers[0].self_attn.q_proj.weight, native_q):
        raise AssertionError("observation load left the copied projection unchanged")
    with torch.no_grad():
        loaded = observer.read(captured, branch)
    if torch.equal(loaded, native):
        raise AssertionError("loaded readout still matches the native UND weights")
    clone = copy.deepcopy(language_model).eval()
    with torch.no_grad():
        for key, tensor in und.items():
            parameter = clone.model.get_parameter(key)
            parameter.copy_(tensor.to(dtype=parameter.dtype))
        matched = observation_slots(
            clone.model.layers, clone.model.embed_tokens, clone.model, observer.seed_token_ids, captured,
            branch, observer.slots, observer.position_start,
        )
    _exact(loaded, matched, "loaded readout matches a decoder carrying the same UND copy")
    if any(parameter.requires_grad for parameter in observer.parameters()):
        raise AssertionError("observation copy is trainable")
    exported = loop.export_state()["model"]
    if any(key.startswith(("observer", "layers")) or "self_attn" in key for key in exported):
        raise AssertionError("ADSL checkpoint stored the observation readout")


def _training_config(stage, depth, **overrides):
    config = {
        "stage": stage,
        "visual_source_policy": "full_query",
        "source_layers": list(range(depth)),
        "loss_weights": {"sem": 1, "teacher": 0, "flow": 0},
        "task_mix": {"observe": 1.0},
    }
    config.update(overrides)
    return config


def _observation_blob(und, *, stage, depth, config, **extra):
    blob = {
        "format_version": 4, "stage": stage, "topology": "C", "slots": 4,
        "depth": depth, "seed_ids": [1, 2, 3, 4], "und": und, "config": config,
    }
    blob.update(extra)
    return blob


def _expect_load_refusal(observer, path, text):
    before = observer.layers[0].self_attn.q_proj.weight.detach().clone()
    try:
        observer.load_checkpoint(path)
    except ValueError as error:
        if text not in str(error):
            raise
    else:
        raise AssertionError("observation loader accepted " + text)
    _exact(observer.layers[0].self_attn.q_proj.weight, before, "reader weights after refused load")


def check_observation_source_mode_follows_the_training_config():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=13)
    loop = make_loop(language_model, force_full_zero_path=True)
    observer = loop.observer
    depth = len(language_model.model.layers)
    und = {}
    for key in observation_readout_keys(observer.layers):
        layer_index = int(key.split(".")[1])
        local = key[len(f"layers.{layer_index}."):]
        und[key] = observer.layers[layer_index].get_parameter(local).detach().cpu().float() + 0.5
    native_q = language_model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    def save(name, **kwargs):
        path = Path("/tmp") / name
        torch.save(_observation_blob(und, depth=depth, **kwargs), path)
        return path

    missing = save("anchored_observation_noconfig.pt", stage="C2-observe", config=None)
    missing_blob = torch.load(missing, map_location="cpu", weights_only=True)
    del missing_blob["config"]
    torch.save(missing_blob, missing)
    _expect_load_refusal(observer, missing, "training config")

    mismatched = save(
        "anchored_observation_stage_mismatch.pt", stage="C2-observe",
        config=_training_config("C2-feedback", depth),
    )
    _expect_load_refusal(observer, mismatched, "stage")

    image_only = save(
        "anchored_observation_image_only.pt", stage="C2-observe",
        config=_training_config("C2-observe", depth, visual_source_policy="image_only"),
    )
    _expect_load_refusal(observer, image_only, "full_query")

    partial_layers = save(
        "anchored_observation_partial_layers.pt", stage="C2-observe",
        config=_training_config("C2-observe", depth, source_layers=[0, 1]),
    )
    _expect_load_refusal(observer, partial_layers, "every layer")

    feedback = save(
        "anchored_observation_feedback_mix.pt", stage="C2-feedback",
        config=_training_config("C2-feedback", depth, task_mix={
            "observe": 0.15, "endpoint": 0.85, "generation": 0.0,
            "trajectory": 0.0, "conditional": 0.0,
        }),
    )
    _expect_load_refusal(observer, feedback, "prompt_and_gen")

    looped = save(
        "anchored_observation_c3_mix.pt", stage="C3-loop",
        config=_training_config(
            "C3-loop", depth,
            loss_weights={"sem": 0.1, "teacher": 1, "flow": 1},
            task_mix={
                "observe": 0.15, "endpoint": 0.4, "generation": 0.3,
                "trajectory": 0.1, "conditional": 0.05,
            },
        ),
    )
    _expect_load_refusal(observer, looped, "prompt_and_gen")
    inert = save(
        "anchored_observation_inert_generation.pt", stage="C2-feedback",
        config=_training_config("C2-feedback", depth, task_mix={
            "observe": 0.0, "endpoint": 0.0, "generation": 1.0,
        }),
    )
    _expect_load_refusal(observer, inert, "no writer task")
    if observer.loaded:
        raise AssertionError("a refused checkpoint marked the reader loaded")
    _exact(language_model.model.layers[0].self_attn.q_proj.weight, native_q, "native Q after refused loads")

    prompt_path = save(
        "anchored_observation_prompt_request.pt", stage="C2-observe",
        config=_training_config("C2-observe", depth),
    )
    try:
        observer.load_checkpoint(prompt_path, source_mode="prompt_and_gen")
    except ValueError as error:
        if "not wired" not in str(error):
            raise
    else:
        raise AssertionError("prompt_and_gen loaded a checkpoint")

    reserved = make_loop(
        language_model, observation_source_mode="prompt_and_gen", force_full_zero_path=True,
    )
    try:
        reserved.forward_branch(hidden, branch, train=False)
    except RuntimeError as error:
        if "not wired" not in str(error):
            raise
    else:
        raise AssertionError("prompt_and_gen ran the gen_full_query reader")
    try:
        AnchoredDualStateLoop(language_model, loop_config(observation_source_mode="image_only"))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown observation source mode was accepted")

    feedback_observe = save(
        "anchored_observation_feedback_observe_only.pt", stage="C2-feedback",
        config=_training_config("C2-feedback", depth, task_mix={
            "observe": 1.0, "endpoint": 0.0, "generation": 0.0,
            "trajectory": 0.0, "conditional": 0.0,
        }),
        parent="runs/other-parent.pt",
    )
    observer.load_checkpoint(feedback_observe)
    status = observer.status()
    if status["stage"] != "C2-feedback" or status["trained_sources"] != ["gen_full_query"]:
        raise AssertionError("an observe-only C2-feedback config was not read as gen_full_query")
    if status["parent"] != "runs/other-parent.pt" or status["source_mode"] != "gen_full_query":
        raise AssertionError("observation provenance dropped the parent or resolved mode")
    copied = observer.layers[0].self_attn.q_proj.weight.detach().clone()
    clobber = Path("/tmp/anchored_observation_feedback_clobber.pt")
    torch.save(_observation_blob(
        {key: torch.zeros_like(value) for key, value in und.items()},
        stage="C2-feedback", depth=depth,
        config=_training_config("C2-feedback", depth, task_mix={
            "observe": 0.15, "endpoint": 0.85,
        }),
    ), clobber)
    _expect_load_refusal(observer, clobber, "prompt_and_gen")
    _exact(observer.layers[0].self_attn.q_proj.weight, copied, "refused load kept the accepted reader")
    if not observer.loaded:
        raise AssertionError("a later refusal cleared an accepted reader")

    c3_observe = save(
        "anchored_observation_c3_observe_only.pt", stage="C3-loop",
        config=_training_config("C3-loop", depth, task_mix={"observe": 1.0}),
    )
    observer.load_checkpoint(c3_observe)
    if observer.status()["stage"] != "C3-loop" or observer.status()["trained_sources"] != ["gen_full_query"]:
        raise AssertionError("an observe-only C3 config was not read as gen_full_query")
    _exact(language_model.model.layers[0].self_attn.q_proj.weight, native_q, "native Q after source-mode loads")


def check_anchored_observation_reuses_native_pre_kv():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=12)
    loop = make_loop(
        language_model, correction_rounds=2, force_full_zero_path=True, log_intermediates=True,
    )
    with torch.no_grad():
        loop.forward_branch(hidden, branch, train=False)
    trace = loop.observation_trace
    if len(trace) != 2:
        raise AssertionError("two correction rounds did not record two observations")
    first, second = trace
    depth = len(language_model.model.layers)
    if first["kind"] != "full_native" or second["kind"] != "anchored":
        raise AssertionError("round 0 is the native forward and the next read is anchored")
    if first["layers"] != list(range(depth)) or second["layers"] != list(range(depth)):
        raise AssertionError("observation capture missed a decoder layer")
    if first["experts"] != ["mixed_und_gen"] * depth:
        raise AssertionError("full-query capture did not keep the mixed GEN expert")
    for roles in first["roles"]:
        if "boundary" not in roles or "image" not in roles or "prompt" in roles:
            raise AssertionError("full-query roles must include boundary and image tokens")
    if not first["pre_ptrs"] or first["pre_ptrs"] != second["pre_ptrs"]:
        raise AssertionError("anchored observation replaced the native Pre KV")
    if first["_kept"][0][0] is not second["_kept"][0][0]:
        raise AssertionError("anchored observation copied Pre KV instead of reusing it")
    if first["body_ptrs"] == second["body_ptrs"] or first["post_ptrs"] == second["post_ptrs"]:
        raise AssertionError("anchored observation reused Body or Post KV from the native forward")
    if first["_kept"][1][0] is second["_kept"][1][0] or first["_kept"][2][0] is second["_kept"][2][0]:
        raise AssertionError("Body or Post KV object was shared across rounds")

    through_end = make_loop(
        language_model, body_start=1, body_end=depth, correction_rounds=2,
        force_full_zero_path=True, log_intermediates=True,
    )
    with torch.no_grad():
        through_end.forward_branch(hidden, branch, train=False)
    spanned = through_end.observation_trace
    if spanned[0]["layers"] != list(range(depth)) or spanned[0]["post_ptrs"] or spanned[1]["post_ptrs"]:
        raise AssertionError("a body that ends at the last layer still produced Post KV")
    if spanned[1]["kind"] != "anchored" or spanned[0]["pre_ptrs"] != spanned[1]["pre_ptrs"]:
        raise AssertionError("empty-post anchored observation dropped the native Pre KV")
    if len(spanned[0]["body_ptrs"]) != depth - 1 or spanned[0]["body_ptrs"] == spanned[1]["body_ptrs"]:
        raise AssertionError("empty-post anchored observation did not recompute Body KV")

    with torch.no_grad():
        loop.forward_branch(hidden, branch, train=False)
    if len(loop.observation_trace) != 2:
        raise AssertionError("observation trace accumulated across forwards")
    late_hidden, late_branch = pack_case(language_model, seed=12, flow_time=0.1)
    with torch.no_grad():
        loop.forward_branch(late_hidden, late_branch, train=False)
    if loop.observation_trace:
        raise AssertionError("an out-of-schedule step kept an observation trace")
    gen_only = make_loop(
        language_model, variant="gen_only", correction_rounds=2, force_full_zero_path=True,
    )
    with torch.no_grad():
        gen_only.forward_branch(hidden, branch, train=False)
    if gen_only.observation_trace:
        raise AssertionError("gen-only recorded an UND observation")


def check_anchored_reader_reuses_the_prefix_hidden():
    """The second read skips Pre layers and matches a full rerun on the stitch."""
    import qwen_latent_cot.anchored.observer as observer_module
    from dataclasses import replace as replace_kv

    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=12)
    loop = make_loop(language_model, correction_rounds=2, force_full_zero_path=True)
    seen = []
    real = observer_module.und_block

    def wrapped(layer, *args, **kwargs):
        seen.append(layer.self_attn.layer_idx)
        return real(layer, *args, **kwargs)

    observer_module.und_block = wrapped
    try:
        with torch.no_grad():
            loop.forward_branch(hidden, branch, train=False)
    finally:
        observer_module.und_block = real
    depth = len(language_model.model.layers)
    boundary = loop.config.body_start
    if seen != list(range(depth)) + list(range(boundary, depth)):
        raise AssertionError(f"second read reran the prefix layers: {seen}")
    first, second = loop.observation_trace
    if second.get("boundary_layer") != boundary or first.get("boundary_entropy") != second.get("boundary_entropy_native"):
        raise AssertionError("boundary entropy was not recorded from the cached prefix")
    if second["boundary_entropy_delta"] != 0.0:
        raise AssertionError("zero residual changed the boundary attention entropy")

    decoder = language_model.model
    observer = loop.observer
    embeddings = loop.path.rope(hidden, branch.packed_query_position_ids)
    captured = {}
    loop.path.pre(hidden, branch, embeddings, captured)
    loop.path.body(hidden, branch, embeddings, captured)
    loop.path.post(hidden, branch, embeddings, captured)
    with torch.no_grad():
        slots, prefix = observer.read_with_boundary(captured, branch, boundary)
        prefix = replace_kv(prefix, pre_ptrs=tuple(item.keys.data_ptr() for item in (captured[i] for i in range(boundary))))
        full = observer.read(captured, branch)
    _exact(slots, full, "prefix snapshot matches the full first read")
    stitched = dict(captured)
    body = stitched[boundary]
    stitched[boundary] = replace_kv(body, keys=body.keys + 0.5, values=body.values - 0.25)
    with torch.no_grad():
        resumed = observer.read_from_boundary(stitched, branch, prefix, prefix.pre_ptrs)
        rerun = observer.read(stitched, branch)
    _exact(resumed, rerun, "cached prefix matches a full rerun on stitched KV")
    try:
        observer.read_from_boundary(stitched, branch, prefix, tuple(range(boundary)))
    except RuntimeError as error:
        if "prefix KV" not in str(error):
            raise
    else:
        raise AssertionError("a changed Pre KV pointer reused the cached reader state")


def check_direct_hidden_is_opt_in():
    assert loop_config().state_mode == "anchored"
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=6)
    direct = make_loop(language_model, state_mode="direct_hidden", max_delta_rms_ratio=1.0)
    with torch.no_grad():
        direct.gate.raw.fill_(1.0)
        output = direct.forward_branch(hidden, branch, train=False)
    if torch.equal(output.entries[0], output.anchor):
        raise AssertionError("direct-hidden ablation used the native anchor")


def check_feedback_content_and_gen_only():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, samples=2, seed=8)
    dual = make_loop(language_model, force_full_zero_path=True)
    with torch.no_grad():
        first = dual.forward_branch(hidden, branch, train=False)
        shifted = hidden.clone()
        shifted[branch.packed_vae_token_indexes] += 1
        second = dual.forward_branch(shifted, branch, train=False)
    if torch.allclose(first.feedback.value, second.feedback.value):
        raise AssertionError("dual feedback ignored GEN hidden")
    correct = make_loop(language_model, feedback_mode="correct", max_delta_rms_ratio=1.0)
    shuffled = make_loop(language_model, feedback_mode="shuffled", max_delta_rms_ratio=1.0)
    zeroed = make_loop(language_model, feedback_mode="zero", max_delta_rms_ratio=1.0)
    with torch.no_grad():
        shuffled.load_state_dict(correct.state_dict())
        zeroed.load_state_dict(correct.state_dict())
        # The updater and the slot attention start near 1e-3. Widen them so a
        # sample roll is visible in bf16 after the frozen body.
        correct.updater.down.weight.normal_(0, 0.5)
        correct.updater.up.weight.normal_(0, 0.5)
        for name in ("q_proj", "k_proj", "v_proj", "mem_out"):
            getattr(correct.adapter, name).weight.normal_(0, 0.5)
        correct.adapter.up.weight.normal_(0, 0.5)
        correct.gate.raw.fill_(1.0)
        shuffled.load_state_dict(correct.state_dict())
        zeroed.load_state_dict(correct.state_dict())
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
    _exact(output.delta, torch.zeros_like(output.delta), "zero gate writes bf16 zero while training")
    if loop.gate.raw.grad is None or float(loop.gate.raw.grad.abs()) == 0.0:
        raise AssertionError("closed gate received no gradient")
    adapter_grad = loop.adapter.up.weight.grad
    memory_grad = loop.adapter.mem_out.weight.grad
    updater_grad = loop.updater.up.weight.grad
    if adapter_grad is None or float(adapter_grad.abs().sum()) == 0.0:
        raise AssertionError("zero-gate warmup did not train the adapter")
    if memory_grad is None or float(memory_grad.abs().sum()) == 0.0:
        raise AssertionError("zero-gate warmup did not train the memory attention")
    if updater_grad is None or float(updater_grad.abs().sum()) == 0.0:
        raise AssertionError("zero-gate warmup did not train the feedback update")
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
    if loop.adapter.mem_out.weight.grad is None or float(loop.adapter.mem_out.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("open gate did not train the memory attention")
    if loop.updater.up.weight.grad is None or float(loop.updater.up.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("open gate did not train the feedback update")

    dual = make_loop(language_model, force_full_zero_path=False, max_delta_rms_ratio=1.0)
    leaf3 = hidden.detach().clone().requires_grad_(True)
    feedback = dual.forward_branch(leaf3, branch, train=True).feedback.value
    feedback.sum().backward()
    if leaf3.grad is None or float(leaf3.grad.abs().sum()) == 0.0:
        raise AssertionError("feedback does not depend on GEN hidden")


def check_gen_only_does_not_carry_memory():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, samples=2, seed=16)
    try:
        make_loop(language_model, variant="gen_only", feedback_mode="shuffled")
    except ValueError as error:
        if "memory" not in str(error):
            raise
    else:
        raise AssertionError("gen-only accepted a memory intervention")

    loop = make_loop(language_model, variant="gen_only", correction_rounds=2, max_delta_rms_ratio=1.0)

    def fail_read(*_args, **_kwargs):
        raise AssertionError("gen-only read UND")

    def fail_update(*_args, **_kwargs):
        raise AssertionError("gen-only updated memory")

    def fail_memory_adapter(*_args, **_kwargs):
        raise AssertionError("gen-only called the memory adapter")

    loop.observer.read = fail_read
    loop.updater.forward = fail_update
    loop.adapter.forward = fail_memory_adapter
    other = make_loop(language_model, variant="gen_only", correction_rounds=2, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        loop.gate.raw.fill_(1.0)
        loop.path.body_calls = 0
        output = loop.forward_branch(hidden, branch, train=False)
        other.load_state_dict(loop.state_dict())
        other.adapter.mem_out.weight.normal_(0, 0.5)
        other.adapter.q_proj.weight.normal_(0, 0.5)
        other.updater.up.weight.normal_(0, 0.5)
        shifted = other.forward_branch(hidden, branch, train=False)
    if output.feedback is not None or shifted.feedback is not None:
        raise AssertionError("gen-only returned a memory state")
    if loop.observation_trace or other.observation_trace:
        raise AssertionError("gen-only recorded an UND observation")
    if loop.path.body_calls != 1 + loop.config.correction_rounds:
        raise AssertionError("gen-only did not repeat the body")
    if torch.equal(output.velocity, output.native_velocity):
        raise AssertionError("state-free correction did not change velocity")
    _exact(shifted.velocity, output.velocity, "memory weights changed gen-only velocity")
    names = [name for name, _parameter in loop.named_parameters() if _parameter.requires_grad and name.startswith((
        "updater.", "adapter.mem_out.", "adapter.q_proj.", "adapter.q_norm.", "adapter.k_norm.",
    ))]
    trained = {name for name, _parameter in loop.named_parameters() if any(parameter is _parameter for parameter in loop.trainable_parameters())}
    if any(name in trained for name in names):
        raise AssertionError("gen-only training includes memory parameters")
    leaf = hidden.detach().clone().requires_grad_(True)
    opened = loop.forward_branch(leaf, branch, train=True)
    opened.velocity.float().sum().backward()
    if loop.adapter.up.weight.grad is None or float(loop.adapter.up.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("gen-only open gate did not train the state-free residual")
    if loop.adapter.mem_out.weight.grad is not None or loop.updater.up.weight.grad is not None:
        raise AssertionError("gen-only trained memory")


def _grad_mass(value) -> float:
    if value is None:
        return 0.0
    return float(value.detach().abs().sum())


def check_round_velocities_and_supervision():
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=17)
    try:
        make_loop(language_model, correction_rounds=2, round_loss_weights=(1.0,))
    except ValueError as error:
        if "one weight" not in str(error):
            raise
    else:
        raise AssertionError("a short round weight tuple was accepted")
    try:
        make_loop(language_model, correction_rounds=2, round_loss_weights=(1.0, -0.1))
    except ValueError as error:
        if "non-negative" not in str(error):
            raise
    else:
        raise AssertionError("a negative round weight was accepted")
    payload = loop_config(correction_rounds=2).to_dict()
    payload["round_loss_weights"] = [1.0, 0.25]
    uneven = LoopConfig.from_dict(payload)
    uneven.validate(len(language_model.model.layers))
    if uneven.round_loss_weights != (1.0, 0.25):
        raise AssertionError("unequal round weights were not kept")

    zero_path = make_loop(
        language_model, correction_rounds=2, force_full_zero_path=True, log_intermediates=True,
    )
    with torch.no_grad():
        zeroed = zero_path.forward_branch(hidden, branch, train=False)
    if (
        len(zeroed.round_velocities) != 3
        or len(zeroed.round_deltas) != 2
        or len(zeroed.round_observations) != 2
    ):
        raise AssertionError("two corrections did not return v0, v1, and v2")
    for index, velocity in enumerate(zeroed.round_velocities):
        _exact(velocity, zeroed.native_velocity, f"zero-path velocity {index}")
    _exact(zeroed.velocity, zeroed.round_velocities[-1], "sampler velocity is the last round")
    observation_shape = (
        int(branch.query_lens.shape[0]), zero_path.config.slots, zeroed.anchor.shape[-1],
    )
    for observation in zeroed.round_observations:
        if not isinstance(observation, torch.Tensor) or tuple(observation.shape) != observation_shape:
            raise AssertionError("dual round observation is missing")
    for delta in zeroed.round_deltas:
        _exact(delta, torch.zeros_like(zeroed.anchor), "zero-path written residual")
    drifts = [record["velocity_drift"] for record in zeroed.records if "round" in record]
    if drifts != [0.0, 0.0]:
        raise AssertionError(f"logged round drift was {drifts}")

    bypass = make_loop(language_model)
    with torch.no_grad():
        skipped = bypass.forward_branch(hidden, branch, train=False)
    if len(skipped.round_velocities) != 1 or skipped.round_deltas or skipped.round_observations:
        raise AssertionError("fast bypass did not keep only the native velocity")
    _exact(skipped.round_velocities[0], skipped.native_velocity, "bypass v0")

    gen = make_loop(
        language_model, variant="gen_only", correction_rounds=2, force_full_zero_path=True,
    )

    def fail_read(*_args, **_kwargs):
        raise AssertionError("gen-only read UND while recording observations")

    gen.observer.read = fail_read
    with torch.no_grad():
        gen_out = gen.forward_branch(hidden, branch, train=False)
    if gen_out.round_observations != [None, None] or len(gen_out.round_velocities) != 3:
        raise AssertionError("gen-only round records were incomplete")
    for velocity in gen_out.round_velocities:
        _exact(velocity, gen_out.native_velocity, "gen-only zero-path velocity")

    far = torch.full((4, 4), 3.0)
    near = torch.ones(4, 4)
    base = torch.zeros(4, 4)
    improved = round_velocity_report([far, near, base], base)
    if not all(item["error_fell"] and not item["error_rose"] for item in improved["corrections"]):
        raise AssertionError("a round that moves toward the target was marked as a rise")
    amplified = round_velocity_report([base, near, far], base)
    if any(item["error_fell"] for item in amplified["corrections"]) or not amplified["corrections"][1]["error_rose"]:
        raise AssertionError("a round that moves away from the target was marked as a correction")
    if not (amplified["drift"][2] > amplified["drift"][1] > amplified["drift"][0]):
        raise AssertionError("drift did not grow across amplifying rounds")

    loop = make_loop(language_model, correction_rounds=2, max_delta_rms_ratio=1.0)
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        loop.updater.up.weight.normal_(0, 0.5)
        loop.gate.raw.fill_(1.0)

    def supervise(weights):
        loop.zero_grad(set_to_none=True)
        leaf = hidden.detach().clone().requires_grad_(True)
        output = loop.forward_branch(leaf, branch, train=True)
        for velocity in output.round_velocities:
            velocity.retain_grad()
        loss, parts = anchored_loss(
            output.velocity, target, output.native_velocity, output.delta, output.anchor,
            branch.packed_vae_token_indexes, lambda_anchor=0.0, lambda_entry=0.0,
            round_velocities=output.round_velocities, round_loss_weights=weights,
        )
        loss.backward()
        return output, loss, parts

    target = torch.randn_like(hidden[branch.packed_vae_token_indexes])
    first, first_loss, _parts = supervise((1.0, 0.0))
    if torch.equal(first.round_velocities[1], first.round_velocities[0]):
        raise AssertionError("first correction matched the native velocity")
    if torch.equal(first.round_velocities[2], first.round_velocities[1]):
        raise AssertionError("second correction matched the first velocity")
    _exact(first.velocity, first.round_velocities[2], "final velocity")
    _exact(first.delta, first.round_deltas[-1], "last written residual")
    if not first.round_velocities[1].requires_grad or not first.round_deltas[0].requires_grad:
        raise AssertionError("round tensors left the training graph")
    moved = round_velocity_report(first.round_velocities, first.round_velocities[1].detach())
    if not moved["corrections"][0]["error_fell"] or not moved["corrections"][1]["error_rose"]:
        raise AssertionError("round 2 versus round 1 was not visible in the velocity report")
    mse1 = (first.round_velocities[1].float() - target.float()).pow(2).mean()
    if not torch.equal(first_loss, mse1):
        raise AssertionError("a weight of (1, 0) did not keep only the first correction")
    if _grad_mass(first.round_velocities[1].grad) == 0.0 or first.round_velocities[2].grad is not None:
        raise AssertionError("the first-round weight supervised another velocity")

    second, second_loss, parts = supervise((0.0, 1.0))
    mse2 = (second.round_velocities[2].float() - target.float()).pow(2).mean()
    if not torch.equal(second_loss, mse2):
        raise AssertionError("a weight of (0, 1) did not keep only the last correction")
    if second.round_velocities[1].grad is not None or _grad_mass(second.round_velocities[2].grad) == 0.0:
        raise AssertionError("the last-round weight supervised another velocity")
    if "mse_0" not in parts or "error_fell_2" not in parts or "error_rose_1" not in parts:
        raise AssertionError("per-round error flags were missing from the loss report")

    mixed, mixed_loss, _parts = supervise((1.0, 0.25))
    expected = (
        (mixed.round_velocities[1].float() - target.float()).pow(2).mean()
        + 0.25 * (mixed.round_velocities[2].float() - target.float()).pow(2).mean()
    )
    if not torch.equal(mixed_loss, expected):
        raise AssertionError("unequal round weights did not match the weighted sum")
    try:
        anchored_loss(
            mixed.velocity, target, mixed.native_velocity, mixed.delta, mixed.anchor,
            branch.packed_vae_token_indexes, lambda_anchor=0.0, lambda_entry=0.0,
            round_velocities=mixed.round_velocities[:2], round_loss_weights=(1.0, 0.0),
        )
    except ValueError as error:
        if "round_loss_weights" not in str(error):
            raise
    else:
        raise AssertionError("a short velocity list was accepted")

    loop.zero_grad(set_to_none=True)
    leaf = hidden.detach().clone().requires_grad_(True)
    output = loop.forward_branch(leaf, branch, train=True)
    for velocity in output.round_velocities:
        velocity.retain_grad()
    default_loss, _parts = anchored_loss(
        output.velocity, target, output.native_velocity, output.delta, output.anchor,
        branch.packed_vae_token_indexes, lambda_anchor=0.0, lambda_entry=0.0,
    )
    default_loss.backward()
    final_mse = (output.velocity.float() - target.float()).pow(2).mean()
    if not torch.equal(default_loss, final_mse) or output.round_velocities[1].grad is not None:
        raise AssertionError("the default flow loss supervised an earlier round")


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

    loop = make_loop(language_model, enabled=True)
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


def check_image_tokens_address_memory_slots():
    width = 8
    adapter = AnchoredReentryAdapter(width, width).eval()
    eye = torch.eye(width)
    with torch.no_grad():
        adapter.q_proj.weight.copy_(eye * 4)
        adapter.k_proj.weight.copy_(eye * 4)
        adapter.v_proj.weight.copy_(eye)
        adapter.mem_out.weight.copy_(eye)
        adapter.h_down.weight.zero_()
        adapter.p_down.weight.zero_()
        adapter.time_down.weight.zero_()
        adapter.up.weight.zero_()

    def packed(samples):
        per = 3
        text, vae = [], []
        for sample in range(samples):
            base = sample * per
            text.append(base)
            vae.extend((base + 1, base + 2))
        return BranchContext(
            name="cond",
            query_lens=torch.tensor((per,) * samples, dtype=torch.int32),
            packed_query_position_ids=torch.arange(per).repeat(samples),
            packed_query_indexes=torch.arange(samples * per),
            past_key_values=object(),
            key_values_lens=torch.zeros(samples, dtype=torch.int32),
            packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
            packed_text_indexes=torch.tensor(text, dtype=torch.long),
            packed_vae_token_indexes=torch.tensor(vae, dtype=torch.long),
            flow_time=0.9,
        )

    hidden = torch.zeros(3, width)
    hidden[1, 0] = 1.0
    hidden[2, 1] = 1.0
    anchor = torch.zeros_like(hidden)
    slots = torch.zeros(1, 2, width)
    slots[0, 0, 0] = 1.0
    slots[0, 1, 1] = 1.0
    branch = packed(1)
    state = FeedbackState(slots, "cond", 1, 0.9)
    with torch.no_grad():
        base = adapter(hidden, state, anchor, 0.9, branch)
        if torch.allclose(base[1], base[2]):
            raise AssertionError("two image tokens read the same memory mixture")
        if float(base[0].abs().sum()) != 0.0:
            raise AssertionError("a text token received a memory residual")
        moved = slots.clone()
        # Stay inside the head that separates the two tokens. A new coordinate
        # in an unqueried head is mixed uniformly into every image token.
        moved[0, 0, 1] = 4.0
        shifted = adapter(hidden, FeedbackState(moved, "cond", 1, 0.9), anchor, 0.9, branch)
        first = float((shifted[1] - base[1]).abs().sum())
        second = float((shifted[2] - base[2]).abs().sum())
        if first <= second * 2:
            raise AssertionError(
                f"slot 0 moved token 0 by {first} and token 1 by {second}"
            )

        hidden2 = hidden.repeat(2, 1)
        anchor2 = torch.zeros_like(hidden2)
        both = torch.zeros(2, 2, width)
        both[0].copy_(slots[0])
        both[1, 0, 3] = 5.0
        both[1, 1, 4] = 5.0
        branch2 = packed(2)
        mixed = adapter(hidden2, FeedbackState(both, "cond", 1, 0.9), anchor2, 0.9, branch2)
        cleared = both.clone()
        cleared[1].zero_()
        isolated = adapter(hidden2, FeedbackState(cleared, "cond", 1, 0.9), anchor2, 0.9, branch2)
        _exact(mixed[:3], isolated[:3], "sample 0 memory read")


def check_slot_heads_keep_separate_slot_weights():
    """Four heads share the rank width. QK RMSNorm keeps each head's slot mix soft."""
    rank = 8
    hidden = 16
    head_dim = rank // SLOT_HEADS
    adapter = AnchoredReentryAdapter(hidden, rank).eval()
    if adapter.num_heads != SLOT_HEADS or adapter.head_dim != head_dim:
        raise AssertionError("slot attention did not split rank into 4 heads")
    if adapter.q_proj.weight.shape != (rank, hidden) or adapter.mem_out.weight.shape != (hidden, rank):
        raise AssertionError("multi-head projections changed the total width")
    if adapter.q_norm.weight.numel() != head_dim or adapter.k_norm.weight.numel() != head_dim:
        raise AssertionError("Q/K RMSNorm is not the head dimension")
    try:
        AnchoredReentryAdapter(hidden, rank + 2)
    except ValueError as error:
        if "divisible" not in str(error):
            raise
    else:
        raise AssertionError("rank not divisible by the head count was accepted")
    try:
        LoopConfig(rank=rank + 2, body_start=0, body_end=1).validate(1)
    except ValueError as error:
        if "divisible" not in str(error):
            raise
    else:
        raise AssertionError("config accepted a rank the slot heads cannot split")

    query = torch.zeros(1, rank)
    key = torch.zeros(2, rank)
    value = torch.zeros(2, rank)
    query[0, 0] = 1.0
    query[0, head_dim] = 1.0
    key[0, 0] = 1.0
    key[1, head_dim] = 1.0
    value[0, 0] = 3.0
    value[1, head_dim] = 5.0
    grown = 50.0
    with torch.no_grad():
        read = slot_attention(
            query, key, value, num_heads=adapter.num_heads, q_norm=adapter.q_norm, k_norm=adapter.k_norm,
        )
        scaled = slot_attention(
            query * grown, key * grown, value,
            num_heads=adapter.num_heads, q_norm=adapter.q_norm, k_norm=adapter.k_norm,
        )
        q = adapter.q_norm(query.reshape(1, SLOT_HEADS, head_dim)).permute(1, 0, 2)
        k = adapter.k_norm(key.reshape(2, SLOT_HEADS, head_dim)).permute(1, 0, 2)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5), dim=-1)
        raw_q = (query * grown).reshape(1, SLOT_HEADS, head_dim).permute(1, 0, 2)
        raw_k = (key * grown).reshape(2, SLOT_HEADS, head_dim).permute(1, 0, 2)
        raw = torch.softmax(torch.matmul(raw_q, raw_k.transpose(-1, -2)) * (head_dim ** -0.5), dim=-1)
    if not torch.allclose(scaled, read, rtol=1e-4, atol=1e-4):
        raise AssertionError("QK RMSNorm changed the read when projections grew")
    if float(weights[0, 0, 0]) <= float(weights[0, 0, 1]) or float(weights[1, 0, 1]) <= float(weights[1, 0, 0]):
        raise AssertionError("heads did not attend different slots")
    if torch.allclose(weights[0, 0], weights[1, 0]):
        raise AssertionError("two heads used the same slot mixture")
    if float(weights.max()) > 0.95:
        raise AssertionError("QK RMSNorm left a near one-hot slot route")
    if float(raw[0, 0].max()) < 0.99 or float(raw[1, 0].max()) < 0.99:
        raise AssertionError("the unnormalized scores did not collapse")
    if abs(float(read[0, 0]) - float(weights[0, 0, 0]) * 3.0) > 1e-4:
        raise AssertionError("head 0 did not carry slot 0")
    if abs(float(read[0, head_dim]) - float(weights[1, 0, 1]) * 5.0) > 1e-4:
        raise AssertionError("head 1 did not carry slot 1")
    leaf = query.detach().clone().requires_grad_(True)
    trained = slot_attention(
        leaf, key, value, num_heads=adapter.num_heads, q_norm=adapter.q_norm, k_norm=adapter.k_norm,
    )
    trained.sum().backward()
    if adapter.q_norm.weight.grad is None or float(adapter.q_norm.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("Q RMSNorm received no gradient")
    if adapter.k_norm.weight.grad is None or float(adapter.k_norm.weight.grad.abs().sum()) == 0.0:
        raise AssertionError("K RMSNorm received no gradient")


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


def _rms_ratio(residual: torch.Tensor, reference: torch.Tensor) -> float:
    residual = residual.float()
    reference = reference.float()
    numerator = residual.pow(2).mean(dim=-1).sqrt()
    denominator = reference.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    return float((numerator / denominator).max())


def check_written_residual_respects_the_rms_cap():
    """The cap applies to gamma * delta. Widened weights make bf16 movement visible.

    The default initialization is checked for the cap and for entering the loop.
    A matching native velocity there is a bf16 rounding fact.
    """
    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=14)
    ratio = 0.02
    loop = make_loop(language_model, max_delta_rms_ratio=ratio)
    seen = {}
    original = loop.adapter.forward

    def record(hidden, feedback, anchor, flow_time, branch):
        delta = original(hidden, feedback, anchor, flow_time, branch)
        seen["delta"] = delta.detach().clone()
        return delta

    loop.adapter.forward = record
    with torch.no_grad():
        loop.adapter.up.weight.normal_(0, 0.5)
        loop.gate.raw.fill_(4.0)
        output = loop.forward_branch(hidden, branch, train=False)
    vae = branch.packed_vae_token_indexes
    raw = seen["delta"]
    capped = written_residual(raw, loop.gate.raw.detach(), output.anchor, ratio, vae)
    _exact(output.delta, capped, "loop writes Bound(gamma * delta)")
    _exact(output.entries[0], output.anchor + output.delta, "entry is anchor plus the written residual")
    written_ratio = _rms_ratio(output.delta[vae], output.anchor[vae])
    if written_ratio > ratio * 1.05:
        raise AssertionError(f"written residual ratio {written_ratio} exceeded {ratio}")
    before_gate = bound_residual(raw.float(), output.anchor.float(), ratio, vae)
    old = loop.gate.raw.detach().float() * before_gate.float()
    old_ratio = _rms_ratio(old[vae], output.anchor[vae].float())
    if old_ratio <= ratio * 2:
        raise AssertionError("bounding before the gate stayed inside the cap, so this fixture misses the order")
    zero = written_residual(raw, torch.zeros((), dtype=torch.float32), output.anchor, ratio, vae)
    _exact(zero, torch.zeros_like(output.anchor), "zero gate writes bf16 zero")
    _exact(output.anchor + zero, output.anchor, "zero gate add")

    plain = make_loop(language_model, max_delta_rms_ratio=ratio)
    with torch.no_grad():
        plain.gate.raw.fill_(1.0)
        plain.path.body_calls = 0
        plain_out = plain.forward_branch(hidden, branch, train=False)
    if plain.path.body_calls != 2:
        raise AssertionError("default-init open gate skipped the correction")
    plain_ratio = _rms_ratio(plain_out.delta[vae], plain_out.anchor[vae])
    if plain_ratio > ratio * 1.05:
        raise AssertionError(f"default-init written ratio {plain_ratio} exceeded {ratio}")


def check_observation_query_follows_the_verified_writer():
    from qwen_latent_cot.anchored import observer as observer_module
    from qwen_latent_cot.diagnostics.memory import neutral_seeds, rope_for, und_block

    language_model = tiny_language_model()
    hidden, branch = pack_case(language_model, seed=15)
    loop = make_loop(language_model, force_full_zero_path=True)
    decoder = language_model.model
    observer = loop.observer
    embeddings = loop.path.rope(hidden, branch.packed_query_position_ids)
    captured = {}
    with torch.no_grad():
        loop.path.pre(hidden, branch, embeddings, captured)
        loop.path.body(hidden, branch, embeddings, captured)
        loop.path.post(hidden, branch, embeddings, captured)
    calls = []
    real = observer_module.und_block

    def spy(layer, queries, rope, lengths, source=None, *, causal=False):
        calls.append((None if source is None else source.layer, queries.detach().clone()))
        return real(layer, queries, rope, lengths, source, causal=causal)

    observer_module.und_block = spy
    try:
        with torch.no_grad():
            slots = observation_slots(
                observer.layers, observer.embed, decoder, observer.seed_token_ids, captured,
                branch, observer.slots, observer.position_start,
            )
    finally:
        observer_module.und_block = real
    depth = len(decoder.layers)
    if [layer for layer, _ in calls] != list(range(depth)):
        raise AssertionError("observation query did not enter every UND layer")
    if len(calls) == loop.config.body_end - loop.config.body_start:
        raise AssertionError("observation query used the GEN body span")
    ids = torch.tensor(observer.seed_token_ids, dtype=torch.long)
    with torch.no_grad():
        seeds = neutral_seeds(decoder, ids)
        _exact(calls[0][1], seeds[0], "observation query is the neutral seed embedding")
        reference = seeds[0]
        positions = torch.arange(observer.position_start, observer.position_start + observer.slots)
        lengths = (observer.slots,)
        for index, layer in enumerate(observer.layers):
            rope = rope_for(decoder, reference, positions)
            reference = und_block(layer, reference, rope, lengths, captured[index], causal=False)
    _exact(slots, reference.view(1, observer.slots, -1), "observation recurrence matches the C2 writer")
    body_only = {index: captured[index] for index in range(loop.config.body_start, loop.config.body_end)}
    try:
        observation_slots(
            observer.layers, observer.embed, decoder, observer.seed_token_ids, body_only,
            branch, observer.slots, observer.position_start,
        )
    except ValueError as error:
        if "every layer" not in str(error):
            raise
    else:
        raise AssertionError("body-layer KV was accepted as the observation source")


def check_zero_gate_warmup_keeps_a_float32_residual():
    """A subnormal-bf16 delta still trains the gate, and the forward write stays zero."""
    anchor = torch.randn(4, 8, dtype=torch.bfloat16)
    # Below the bf16 subnormal floor, still nonzero in the float32 product.
    delta = torch.full((4, 8), 1e-41, requires_grad=True)
    if float(delta.detach().abs().sum()) == 0.0 or torch.count_nonzero(delta.detach().to(torch.bfloat16)) != 0:
        raise AssertionError("fixture must stay nonzero in float32 and cast to bf16 zero")
    gate = EntryGate()
    vae = torch.tensor([1, 2])
    target = torch.randn(2, 8)
    written = written_residual(delta, gate(), anchor, 1.0, vae)
    _exact(written, torch.zeros_like(anchor), "warmup forward is bf16 zero")
    (written[vae].float() - target).pow(2).mean().backward()
    if gate.raw.grad is None or float(gate.raw.grad.abs()) == 0.0:
        raise AssertionError("bf16-underflow delta gave the gate no gradient")
    if delta.grad is None or float(delta.grad.abs().sum()) == 0.0:
        raise AssertionError("zero-gate warmup gave the residual no gradient")

    frozen = delta.detach().clone().requires_grad_(True)
    plain = written_residual(frozen, torch.zeros(()), anchor, 1.0, vae)
    (plain[vae].float() - target).pow(2).mean().backward()
    if frozen.grad is None or float(frozen.grad.abs().sum()) != 0.0:
        raise AssertionError("detached zero warmed the residual")

    opened_delta = torch.randn(4, 8, requires_grad=True)
    opened_gate = EntryGate()
    with torch.no_grad():
        opened_gate.raw.fill_(0.2)
    opened = written_residual(opened_delta, opened_gate(), anchor, 1.0, vae)
    manual_delta = opened_delta.detach().clone().requires_grad_(True)
    manual_scale = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    manual = bound_residual(manual_scale * manual_delta, anchor.float(), 1.0, vae).to(dtype=anchor.dtype)
    _exact(opened, manual, "warmup matches the true product once the gate has opened")
    (opened[vae].float() - target).pow(2).mean().backward()
    (manual[vae].float() - target).pow(2).mean().backward()
    if not torch.allclose(opened_delta.grad, manual_delta.grad):
        raise AssertionError("open-gate residual gradient differed from the true product")
    if not torch.allclose(opened_gate.raw.grad, manual_scale.grad):
        raise AssertionError("open-gate gradient differed from the true product")

    torch.manual_seed(0)
    adapter = AnchoredReentryAdapter(128, 32)
    updater = FeedbackStateUpdater(128, 4, 32)
    output_layers = (adapter.up.weight, adapter.mem_out.weight, updater.up.weight)
    early_layers = (adapter.h_down.weight, adapter.q_proj.weight, updater.down.weight)
    with torch.no_grad():
        for weight in output_layers:
            if abs(float(weight.std()) - OUTPUT_INIT_STD) > 2e-3:
                raise AssertionError("output projection init left std 1e-2")
        for weight in early_layers:
            if abs(float(weight.std()) - PROJECTION_INIT_STD) > 2e-4:
                raise AssertionError("early projection init left std 1e-3")
    if float(gate.raw.detach()) != 0.0 or GATE_WARMUP_SCALE != 1e-3:
        raise AssertionError("gate warmup constants drifted")


CHECKS = [
    check_schedule,
    check_default_interval_skips_pure_noise,
    check_span_matches_native,
    check_exact_bypass_and_cache,
    check_fast_bypass_skips_observer,
    check_inference_reads_the_trained_gate,
    check_schedule_direction_changes_velocity,
    check_rounds_share_the_same_anchor,
    check_trained_observation_is_an_independent_copy,
    check_observation_query_follows_the_verified_writer,
    check_observation_source_mode_follows_the_training_config,
    check_anchored_observation_reuses_native_pre_kv,
    check_anchored_reader_reuses_the_prefix_hidden,
    check_direct_hidden_is_opt_in,
    check_feedback_content_and_gen_only,
    check_gen_only_does_not_carry_memory,
    check_round_velocities_and_supervision,
    check_gradients_and_frozen_native,
    check_zero_gate_warmup_keeps_a_float32_residual,
    check_written_residual_respects_the_rms_cap,
    check_cfg_matches_bagel,
    check_image_tokens_address_memory_slots,
    check_slot_heads_keep_separate_slot_weights,
    check_checkpoint_and_bound,
]


def run_self_check() -> list[str]:
    names = []
    for check in CHECKS:
        check()
        names.append(check.__name__.removeprefix("check_"))
    return names
