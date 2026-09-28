from types import SimpleNamespace

import pytest
import torch

from qwen_latent_cot.bagel.memory_mechanism import run_decoder
from qwen_latent_cot.bagel.normal_r_inference import (
    NormalRoundEngine, MODES, ROUNDS, probe_reference_trajectory, generate_trajectory,
)
from qwen_latent_cot.bagel.mechanism_inference import MemoryMechanismEngine
from qwen_latent_cot.bagel.modeling.bagel import qwen2_navit as navit
from test_memory_mechanism import tiny_model, real_kwargs


@pytest.mark.parametrize("rounds", ROUNDS)
def test_each_write_resets_nonmemory_recycles_memory_and_suffix_once(rounds):
    seq = torch.arange(16.).reshape(8, 2)
    mem = torch.tensor([1, 5]); nonmem = torch.tensor([0, 2, 3, 4, 6, 7])
    calls, diag = [], {}
    def layer(index, h, **kwargs):
        calls.append((index, h.clone(), kwargs))
        return h + index + 1, None
    out = run_decoder(seq, mode="normal", rounds=rounds, indexes=mem,
                      gen_indexes=torch.tensor([2, 6]), query_lens=torch.tensor([4, 4]),
                      body_start=1, body_end=3, num_layers=4, run_layer=layer,
                      normalize=lambda h: h, diagnostics=diag)
    assert [c[0] for c in calls] == [0] + [1, 2] * rounds + [3]
    assert [c[2]["block_gen_reads_memory"] for c in calls] == [True] * 3 + [False] * (2 * (rounds - 1) + 1)
    for write in range(1, rounds):
        entry = calls[1 + 2 * write][1]
        assert torch.equal(entry[nonmem], (seq + 1)[nonmem])
        assert torch.equal(entry[mem], (seq + 1 + 5 * write)[mem])
    assert len(diag["memory_write_inputs"]) == rounds - 1
    assert [x["round"] for x in diag["memory_hidden_max"] if x["stage"] == "write"] == [r for r in range(1, rounds) for _ in range(2)]
    assert torch.equal(out[nonmem], (seq + 10)[nonmem])
    assert torch.equal(out[mem], (seq + 5 * rounds + 5)[mem])


@pytest.mark.parametrize("rounds", ROUNDS)
@torch.inference_mode()
def test_real_decoder_matches_legacy_normal_round_recurrence(rounds):
    model = tiny_model(); kwargs = real_kwargs()
    expected = model.forward_inference(**kwargs, memory_loop_start=1,
                                      memory_loop_end=2, memory_loop_repeat=rounds).packed_query_sequence
    diag = {}
    actual = model.forward_inference(**kwargs, memory_loop_start=1, memory_loop_end=2,
                                    memory_control_mode="normal", memory_control_rounds=rounds,
                                    mechanism_diagnostics=diag).packed_query_sequence
    assert torch.equal(expected, actual)
    assert {x["round"] for x in diag["attention"] if x["stage"] == "write"} == set(range(1, rounds))


def test_causal_lm_wrapper_forwards_round_count(monkeypatch):
    model = navit.Qwen2ForCausalLM(tiny_model().config)
    seen = {}
    def forward(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(packed_query_sequence=kwargs["packed_query_sequence"])
    monkeypatch.setattr(model.model, "forward_inference", forward)
    model.forward_inference(**real_kwargs(), memory_control_mode="normal", memory_control_rounds=8)
    assert seen["memory_control_rounds"] == 8


@pytest.mark.parametrize("invalid", [0, 1, -1, 2.5, True])
def test_reject_invalid_rounds(invalid):
    with pytest.raises(ValueError, match="integer >= 2"):
        run_decoder(torch.zeros(2, 2), mode="normal", rounds=invalid, indexes=torch.tensor([0]),
                    gen_indexes=torch.tensor([1]), query_lens=torch.tensor([2]), body_start=0,
                    body_end=1, num_layers=1, run_layer=None, normalize=None)


def test_normal_adapter_rejects_removed_arms_and_passes_R(monkeypatch):
    calls = []
    def velocity(self, x, t, mode, **kwargs):
        calls.append((mode, kwargs)); return x, None
    monkeypatch.setattr(MemoryMechanismEngine, "velocity", velocity)
    engine = object.__new__(NormalRoundEngine)
    for mode in MODES: engine.velocity(torch.ones(1), 1., mode)
    assert [k["rounds"] for m, k in calls] == list(ROUNDS)
    assert all(m == "normal" for m, _ in calls)
    for removed in ["native", "static_null", "zero_dynamic", "shuffled_dynamic", "frozen_correct"]:
        with pytest.raises(ValueError, match="accepts only"):
            engine.velocity(torch.ones(1), 1., removed)


def test_probe_only_R2_advances_and_independent_trajectories_start_fresh():
    calls, rows = [], []
    class Engine:
        noise = torch.ones(2, 1)
        lengths = [1, 1]
        model = SimpleNamespace(image_euler_step=lambda x, v, dt: x - v * dt)
        def velocity(self, x, t, mode, diagnostics=False):
            calls.append((mode, float(t), x.clone()))
            v = torch.full_like(x, MODES.index(mode) + 1)
            return v, {"gen_body_hidden": v, "gen_suffix_hidden": v}
    engine = Engine()
    result = probe_reference_trajectory(engine, [1., .5], [.5, .5], lambda *args: rows.extend(args[3]))
    assert result.eq(0).all()
    assert len(calls) == 8
    for mode, t, x in calls: assert x.eq(1 if t == 1 else .5).all()
    assert rows[0]["relative_r4_vs_r2"] == pytest.approx(1)
    assert rows[0]["relative_r8_vs_r2"] == pytest.approx(3)
    assert rows[0]["arms"]["normal_r2"]["relative_velocity_vs_r2"] == 0
    for mode in MODES:
        result = generate_trajectory(engine, [1., .5], [.5, .5], mode)
        assert result.eq(-MODES.index(mode)).all()
    assert engine.noise.eq(1).all()
