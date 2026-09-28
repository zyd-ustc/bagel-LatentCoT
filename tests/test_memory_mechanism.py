from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qwen_latent_cot.bagel.memory_mechanism import (
    MODES, append_attention_stats, run_decoder, select_write_state, zero_memory_qkv,
)
from qwen_latent_cot.bagel.mechanism_inference import (
    MemoryMechanismEngine, PAIRS, generate_trajectory, probe_base_trajectory,
)
from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel
from qwen_latent_cot.bagel.modeling.bagel import qwen2_navit as navit


def run_fake(mode):
    seq = torch.arange(16.0).reshape(8, 2)
    memory = torch.tensor([1, 5]) if mode != "native" else None
    calls, diagnostics = [], {}

    def layer(i, hidden, **kwargs):
        calls.append((i, hidden.clone(), kwargs))
        return hidden + (i + 1), None

    out = run_decoder(
        seq, mode=mode, indexes=memory, gen_indexes=torch.tensor([2, 6]),
        query_lens=torch.tensor([4, 4]), body_start=1, body_end=3,
        num_layers=4, run_layer=layer, normalize=lambda x: x, diagnostics=diagnostics,
    )
    return seq, out, calls, diagnostics


@pytest.mark.parametrize("mode", MODES)
def test_schedule_and_nonmemory_reset(mode):
    seq, output, calls, diag = run_fake(mode)
    assert [c[0] for c in calls] == ([0, 1, 2, 3] if mode == "native" else [0, 1, 2, 1, 2, 3])
    if mode == "native":
        assert torch.equal(output, seq + 10)
        return
    assert [c[2]["block_gen_reads_memory"] for c in calls] == [True, True, True, False, False, False]
    nonmemory = torch.tensor([0, 2, 3, 4, 6, 7])
    assert torch.equal(calls[1][1][nonmemory], calls[3][1][nonmemory])
    assert torch.equal(diag["memory_write_input"], calls[3][1][[1, 5]])


def test_null_is_clamped_at_every_layer_and_frozen_never_evolves_after_read():
    _, null, calls, diag = run_fake("static_null")
    assert all(c[2]["force_memory_qkv_zero"] for c in calls)
    assert all(torch.count_nonzero(c[1][[1, 5]]) == 0 for c in calls)
    assert torch.count_nonzero(null[[1, 5]]) == 0
    assert all(row["max_abs"] == 0 for row in diag["memory_hidden_max"])
    _, frozen, calls, diag = run_fake("frozen_correct")
    for _, entry, _ in calls[3:]:
        assert torch.equal(entry[[1, 5]], diag["memory_read"])
    assert torch.equal(frozen[[1, 5]], diag["memory_read"])
    _, zero, calls, diag = run_fake("zero_dynamic")
    assert torch.count_nonzero(diag["memory_write_input"]) == 0
    assert torch.count_nonzero(zero[[1, 5]]) > 0


def test_shuffled_read_swaps_whole_samples_without_mutating_read():
    read = torch.arange(32.0).reshape(16, 2)
    original = read.clone()
    shuffled = select_write_state(read, "shuffled_dynamic", 2)
    assert torch.equal(shuffled[:8], read[8:])
    assert torch.equal(shuffled[8:], read[:8])
    assert torch.equal(original, read)
    with pytest.raises(ValueError, match="complete paired"):
        select_write_state(read, "shuffled_dynamic", 1)


def test_attention_mass_denominator_and_read_mask():
    query = torch.zeros(3, 2, 2)
    key = torch.zeros(5, 1, 2)  # 2 prompt + boundary + memory + GEN.
    common = dict(query=query, key=key, query_lens=torch.tensor([3]),
                  key_lens=torch.tensor([5]), memory_indexes=torch.tensor([1]),
                  gen_indexes=torch.tensor([2]), layer=0, stage="write", chunk_size=1)
    trace = []
    append_attention_stats(trace, blocked_slices=None, **common)
    assert trace[0]["gen_to_prompt"] == pytest.approx(2 / 5)
    assert trace[0]["gen_to_memory"] == pytest.approx(1 / 5)
    trace = []
    append_attention_stats(trace, blocked_slices=[(torch.tensor([0, 2]), torch.tensor([3]))], **common)
    assert trace[0]["gen_to_memory"] == 0
    assert trace[0]["gen_to_prompt"] == pytest.approx(2 / 4)
    assert trace[0]["memory_to_prompt"] == pytest.approx(2 / 5)


def tiny_model():
    torch.manual_seed(11)
    cfg = navit.Qwen2Config(
        vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, layer_module="Qwen2MoTDecoderLayer", qk_norm=True,
    )
    return navit.Qwen2Model(cfg).to(torch.bfloat16).eval()


def real_kwargs(memory=True):
    width = 4 if memory else 3
    seq = torch.randn(width * 2, 8, dtype=torch.bfloat16)
    mem = torch.tensor([1, width + 1]) if memory else torch.empty(0, dtype=torch.long)
    gen = torch.tensor([width - 2, 2 * width - 2])
    text = torch.tensor([i for i in range(2 * width) if i not in gen.tolist()])
    cache = navit.NaiveCache(3)
    for i in range(3):
        cache.key_cache[i] = torch.randn(2, 1, 4, dtype=torch.bfloat16)
        cache.value_cache[i] = torch.randn(2, 1, 4, dtype=torch.bfloat16)
    return dict(
        packed_query_sequence=seq, query_lens=torch.tensor([width, width], dtype=torch.int),
        packed_query_position_ids=torch.zeros(2 * width, dtype=torch.long),
        packed_query_indexes=torch.tensor(list(range(1, width + 1)) + list(range(width + 2, 2 * width + 2))),
        past_key_values=cache, key_values_lens=torch.tensor([1, 1], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([0, width + 1]),
        update_past_key_values=False, is_causal=False, mode="gen",
        packed_text_indexes=text, packed_vae_token_indexes=gen, packed_memory_token_indexes=mem,
    )


@torch.inference_mode()
def test_native_matches_original_real_mot_and_cache_is_unchanged(monkeypatch):
    # An H200 environment may have FlashAttention installed, but CPU unit tests
    # must still select SDPA rather than trying to run a CUDA-only kernel.
    monkeypatch.setattr(navit, "flash_attn_varlen_func", lambda **_: pytest.fail("CPU called CUDA FlashAttention"))
    model = tiny_model()
    kwargs = real_kwargs(memory=False)
    snapshots = [v.clone() for v in kwargs["past_key_values"].value_cache.values()]
    original = model.forward_inference(**kwargs).packed_query_sequence
    diag = {}
    actual = model.forward_inference(**kwargs, memory_control_mode="native",
                                     memory_loop_start=1, memory_loop_end=2,
                                     mechanism_diagnostics=diag).packed_query_sequence
    assert torch.equal(original, actual)
    assert diag["gen_body_hidden"].shape == (2, 8)
    assert len(diag["attention"]) == 2
    for before, after in zip(snapshots, kwargs["past_key_values"].value_cache.values()):
        assert torch.equal(before, after)


@torch.inference_mode()
def test_real_mot_strict_null_overrides_projection_bias_and_rope(monkeypatch):
    model = tiny_model()
    for layer in model.layers:
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            projection.bias.fill_(3)
    kwargs = real_kwargs()
    seen = []
    original = navit._sdpa_varlen_inference

    def capture(**values):
        seen.append(values)
        assert torch.count_nonzero(values["query"][[1, 5]]) == 0
        assert torch.count_nonzero(values["key"][[2, 7]]) == 0
        assert torch.count_nonzero(values["value"][[2, 7]]) == 0
        return original(**values)

    monkeypatch.setattr(navit, "flash_attn_varlen_func", None)
    monkeypatch.setattr(navit, "_sdpa_varlen_inference", capture)
    diag = {}
    result = model.forward_inference(**kwargs, memory_control_mode="static_null",
                                     memory_loop_start=1, memory_loop_end=2,
                                     mechanism_diagnostics=diag)
    assert len(seen) == 4
    assert torch.count_nonzero(result.packed_query_sequence[[1, 5]]) == 0
    assert all(r["gen_to_memory"] > 0 for r in diag["attention"] if r["stage"] in ("write", "suffix"))


@torch.inference_mode()
def test_normal_matches_legacy_strict_read_loop():
    model = tiny_model()
    kwargs = real_kwargs()
    old = model.forward_inference(**kwargs, memory_loop_start=1, memory_loop_end=2,
                                  memory_loop_repeat=2).packed_query_sequence
    new = model.forward_inference(**kwargs, memory_control_mode="normal",
                                  memory_loop_start=1, memory_loop_end=2).packed_query_sequence
    assert torch.equal(old, new)


def test_probe_uses_only_base_states_and_generation_has_independent_states():
    seen = []
    class Engine:
        noise = torch.ones(2, 1)
        lengths = [1, 1]
        model = SimpleNamespace(image_euler_step=lambda x, v, dt: x - v * dt)

        def velocity(self, state, t, mode, diagnostics=False):
            seen.append((float(t), mode, state.clone()))
            value = MODES.index(mode) + 1.0
            velocity = torch.full_like(state, value)
            return velocity, {"gen_body_hidden": velocity, "gen_suffix_hidden": velocity}

    rows = []
    final = probe_base_trajectory(Engine(), [1.0, 0.5], [0.5, 0.5], lambda *args: rows.extend(args[3]))
    for t, mode, state in seen:
        assert torch.equal(state, torch.full_like(state, 1 if t == 1 else 0.5))
    assert torch.count_nonzero(final) == 0
    assert len(rows) == 4
    assert rows[0]["relative_topology"] == pytest.approx(1)
    assert rows[0]["cos_D_vs_C"] == pytest.approx(1)
    result = generate_trajectory(Engine(), [1.0, 0.5], [0.5, 0.5], "normal")
    assert torch.equal(result, torch.full_like(result, -3))


@torch.inference_mode()
def test_legacy_controls_cannot_be_mixed_with_mechanism_mode():
    model = tiny_model()
    with pytest.raises(ValueError, match="without legacy"):
        model.forward_inference(**real_kwargs(), memory_control_mode="normal",
                                memory_loop_start=1, memory_loop_end=2,
                                prompt_kv_mask_scope="all_generation")


@torch.inference_mode()
@pytest.mark.parametrize("rounds", [2, 4, 6, 8])
def test_real_velocity_engine_native_parity_cfg_and_fresh_memory(rounds):
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = navit.Qwen2Config(
                vocab_size=16, hidden_size=8, intermediate_size=16,
                num_hidden_layers=20, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=32, layer_module="Qwen2MoTDecoderLayer", qk_norm=True,
            )
            self.model = navit.Qwen2Model(config)
            self.calls = []

        def forward_inference(self, **kwargs):
            self.calls.append(kwargs)
            return self.model.forward_inference(**kwargs)

    class Time(torch.nn.Module):
        def forward(self, t):
            return t[:, None].expand(-1, 8)

    class Model(torch.nn.Module):
        prepare_vae_latent = Bagel.prepare_vae_latent
        prepare_vae_latent_cfg = Bagel.prepare_vae_latent_cfg
        mot_und_route_indexes = staticmethod(Bagel.mot_und_route_indexes)
        _combine_cfg_velocities = Bagel._combine_cfg_velocities
        _forward_flow = Bagel._forward_flow

        def __init__(self):
            super().__init__()
            self.language_model = LM()
            self.hidden_size, self.use_moe = 8, True
            self.latent_downsample, self.latent_channel, self.latent_patch_size = 16, 1, 1
            self.max_latent_size = 64
            self.config = SimpleNamespace(num_loop_tokens=8, llm_config=SimpleNamespace(num_hidden_layers=20))
            self.loop_memory = torch.nn.Parameter(torch.randn(8, 8))
            self.vae2llm, self.llm2vae = torch.nn.Linear(1, 8), torch.nn.Linear(8, 1)
            self.time_embedder, self.latent_pos_embed = Time(), torch.nn.Embedding(16, 8)

        def get_flattened_position_ids(self, *args, **kwargs):
            return torch.tensor([0])

        def prepare_prompts(self, **kwargs):
            return {}, [1, 1], [1, 1]

        def forward_cache_update_text(self, cache):
            for i in range(20):
                cache.key_cache[i] = torch.randn(2, 1, 4, dtype=torch.bfloat16)
                cache.value_cache[i] = torch.randn(2, 1, 4, dtype=torch.bfloat16)
            return cache

    torch.manual_seed(17)
    model = Model().to(torch.bfloat16).eval().requires_grad_(False)
    inferencer = SimpleNamespace(model=model, device=torch.device("cpu"), tokenizer=None,
                                 new_token_ids={"start_of_image": 1, "end_of_image": 2})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        engine = MemoryMechanismEngine(inferencer, ["first", "second"], [torch.ones(1, 1)] * 2, (16, 16))
        original_kwargs = {key: value for key, value in engine.layouts[0].items()
                           if key not in ("packed_init_noises", "packed_vae_seqlens", "packed_loop_token_indexes")}
        expected = model._forward_flow(x_t=engine.noise, timestep=torch.full((2,), 0.2),
                                       past_key_values=engine.cache, **original_kwargs)
        actual, _ = engine.velocity(engine.noise, 0.2, "native")
        assert torch.equal(expected, actual)
        model.language_model.calls.clear()
        first, _ = engine.velocity(engine.noise, 1., "normal", rounds=rounds)
        again, _ = engine.velocity(engine.noise, 1., "normal", rounds=rounds)
    calls = model.language_model.calls
    assert torch.equal(first, again)
    assert len(calls) == 4
    assert [c["past_key_values"] is engine.cache for c in calls] == [True, False, True, False]
    assert all(c["memory_control_mode"] == "normal" for c in calls)
    assert all(c["memory_control_rounds"] == rounds for c in calls)
    assert all(c["update_past_key_values"] is False for c in calls)
    for c in calls:
        assert torch.equal(c["packed_query_sequence"][c["packed_memory_token_indexes"]], model.loop_memory.repeat(2, 1))
