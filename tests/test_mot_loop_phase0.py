from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel


def _packer(num_loop_tokens: int = 0):
    obj = Bagel.__new__(Bagel)
    obj.latent_downsample = 16
    obj.latent_channel = 4
    obj.latent_patch_size = 2
    obj.max_latent_size = 64
    obj.config = SimpleNamespace(num_loop_tokens=num_loop_tokens)

    def positions(height, width, downsample, max_num_patches_per_side):
        rows = height // downsample
        cols = width // downsample
        return torch.arange(rows * cols)

    obj.get_flattened_position_ids = positions
    return obj


def test_k0_layout_matches_vanilla_query_width():
    packer = _packer(0)
    out = Bagel.prepare_vae_latent(
        packer,
        curr_kvlens=[3],
        curr_rope=[0],
        image_sizes=[(32, 32)],
        new_token_ids={"start_of_image": 1, "end_of_image": 2},
    )
    # 32/16=2 -> 4 VAE tokens + start + end
    assert int(out["packed_seqlens"][0]) == 6
    assert int(out["packed_loop_token_indexes"].numel()) == 0
    assert int(out["packed_vae_token_indexes"].numel()) == 4
    assert list(out["packed_text_indexes"].tolist()) == [0, 5]


def test_k8_inserts_und_route_slots_without_vae_pos_or_text_ids():
    packer = _packer(8)
    out = Bagel.prepare_vae_latent(
        packer,
        curr_kvlens=[3],
        curr_rope=[0],
        image_sizes=[(32, 32)],
        new_token_ids={"start_of_image": 1, "end_of_image": 2},
        num_loop_tokens=8,
    )
    loop_idx = out["packed_loop_token_indexes"]
    text_idx = out["packed_text_indexes"]
    vae_idx = out["packed_vae_token_indexes"]
    assert int(out["packed_seqlens"][0]) == 6 + 8
    assert list(loop_idx.tolist()) == list(range(1, 9))
    assert set(loop_idx.tolist()).isdisjoint(text_idx.tolist())
    assert set(loop_idx.tolist()).isdisjoint(vae_idx.tolist())
    assert int(out["packed_text_ids"].numel()) == 2
    # memory uses the image-block LLM position, not a new 2D VAE grid
    image_pos = int(out["packed_position_ids"][0])
    assert torch.equal(
        out["packed_position_ids"][loop_idx],
        torch.full((8,), image_pos),
    )


def test_cfg_query_layout_matches_cond_seqlens():
    packer = _packer(8)
    cond = Bagel.prepare_vae_latent(
        packer,
        curr_kvlens=[5],
        curr_rope=[2],
        image_sizes=[(32, 32)],
        new_token_ids={"start_of_image": 1, "end_of_image": 2},
        num_loop_tokens=8,
    )
    cfg = Bagel.prepare_vae_latent_cfg(
        packer,
        curr_kvlens=[5],
        curr_rope=[2],
        image_sizes=[(32, 32)],
        num_loop_tokens=8,
    )
    assert int(cfg["cfg_packed_position_ids"].numel()) == int(cond["packed_seqlens"][0])
    assert torch.equal(cfg["cfg_packed_position_ids"], cond["packed_position_ids"])
    assert torch.equal(cfg["cfg_packed_query_indexes"], cond["packed_indexes"])


def test_und_route_indexes_concat_text_then_loop():
    text = torch.tensor([0, 9], dtype=torch.long)
    loop = torch.tensor([1, 2, 3], dtype=torch.long)
    routed = Bagel.mot_und_route_indexes(text, loop)
    assert list(routed.tolist()) == [0, 9, 1, 2, 3]
    assert torch.equal(Bagel.mot_und_route_indexes(text, text.new_empty(0)), text)


def test_k0_and_r1_defaults_do_not_enable_memory_loop(monkeypatch):
    calls = []

    class Dummy(Bagel):
        def __init__(self):
            self.config = SimpleNamespace(num_loop_tokens=0, loop_depth=1)
            self.loop_memory = None
            self.last_loop_diagnostics = []
            self.language_model = SimpleNamespace(model=SimpleNamespace(enable_taylorseer=False))

        def prepare_image_schedule(self, num_timesteps, timestep_shift, device):
            t = torch.tensor([1.0, 0.5], device=device)
            return t, torch.tensor([0.5, 0.5], device=device)

        def predict_image_velocity(self, **kwargs):
            calls.append("vanilla")
            return kwargs["x_t"]

        def _forward_flow_loop(self, **kwargs):
            calls.append("loop")
            raise AssertionError("K=0 must not call _forward_flow_loop")

        def image_euler_step(self, x_t, velocity, dt):
            return x_t

    dummy = Dummy.__new__(Dummy)
    Dummy.__init__(dummy)
    dummy.generate_image = Bagel.generate_image.__get__(dummy, Dummy)
    kwargs = {
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.tensor([0, 1]),
        "packed_init_noises": torch.zeros(4, 8),
        "packed_vae_position_ids": torch.zeros(4, dtype=torch.long),
        "packed_vae_token_indexes": torch.arange(4),
        "packed_vae_seqlens": torch.tensor([4], dtype=torch.int),
        "packed_boundary_token_indexes": torch.tensor([0, 1]),
        "packed_seqlens": torch.tensor([2], dtype=torch.int),
        "packed_position_ids": torch.zeros(2, dtype=torch.long),
        "packed_indexes": torch.arange(2),
        "past_key_values": None,
        "key_values_lens": torch.tensor([0], dtype=torch.int),
        "packed_key_value_indexes": torch.tensor([], dtype=torch.long),
        "num_timesteps": 2,
        "timestep_shift": 1.0,
        "cfg_interval": (0.0, 1.0),
        "enable_taylorseer": False,
    }
    dummy.generate_image(**kwargs)
    assert calls == ["vanilla", "vanilla"]
    assert dummy.last_loop_diagnostics == []


def test_r2_calls_inner_loop_twice_but_euler_once_per_timestep():
    velocity_calls = []
    euler_calls = []

    class Dummy(Bagel):
        def __init__(self):
            self.config = SimpleNamespace(
                num_loop_tokens=2,
                loop_depth=2,
                loop_uncond_memory="m0",
                loop_recycle_mode="full_depth",
                loop_memory_persist=True,
                memory_loop_start_layer=1,
                memory_loop_end_layer=2,
            )
            self.loop_memory = torch.ones(2, 4)
            self.loop_memory_persist = True
            self.last_loop_diagnostics = []
            self.language_model = SimpleNamespace(
                model=SimpleNamespace(enable_taylorseer=False)
            )

        def prepare_image_schedule(self, num_timesteps, timestep_shift, device):
            t = torch.tensor([1.0, 0.4], device=device)
            return t, torch.tensor([0.6, 0.4], device=device)

        def predict_image_velocity(self, **kwargs):
            raise AssertionError("K>0 must not use vanilla _forward_flow")

        def _forward_flow_loop(self, **kwargs):
            velocity_calls.append(float(kwargs["timestep"].reshape(-1)[0]))
            memory = kwargs["loop_memory"] + 1
            diag = {"memory_rms": 1.0, "vae_hidden_rms": 1.0, "velocity_norm": 1.0}
            return kwargs["x_t"], memory, memory, memory, diag

        def image_euler_step(self, x_t, velocity, dt):
            euler_calls.append(float(dt))
            return x_t

    dummy = Dummy.__new__(Dummy)
    Dummy.__init__(dummy)
    dummy.generate_image = Bagel.generate_image.__get__(dummy, Dummy)
    dummy.generate_image(
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0, 3]),
        packed_init_noises=torch.zeros(2, 4),
        packed_vae_position_ids=torch.zeros(2, dtype=torch.long),
        packed_vae_token_indexes=torch.tensor([1, 2]),
        packed_vae_seqlens=torch.tensor([2], dtype=torch.int),
        packed_boundary_token_indexes=torch.tensor([0, 3]),
        packed_seqlens=torch.tensor([4], dtype=torch.int),
        packed_position_ids=torch.zeros(4, dtype=torch.long),
        packed_indexes=torch.arange(4),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        packed_loop_token_indexes=torch.tensor([1, 2]),
        num_timesteps=2,
        timestep_shift=1.0,
        cfg_interval=(0.0, 1.0),
        enable_taylorseer=False,
        loop_depth=2,
        loop_recycle_mode="full_depth",
    )
    assert len(velocity_calls) == 4  # 2 timesteps * R=2
    assert len(euler_calls) == 2  # outer clock only
    assert [row["r"] for row in dummy.last_loop_diagnostics] == [0, 1, 0, 1]


def test_same_depth_runs_one_forward_per_timestep():
    velocity_calls = []
    euler_calls = []

    class Dummy(Bagel):
        def __init__(self):
            self.config = SimpleNamespace(
                num_loop_tokens=2,
                loop_depth=2,
                loop_uncond_memory="m0",
                loop_recycle_mode="same_depth",
                loop_memory_persist=True,
                memory_loop_start_layer=1,
                memory_loop_end_layer=2,
            )
            self.loop_memory = torch.ones(2, 4)
            self.loop_memory_persist = True
            self.last_loop_diagnostics = []
            self.language_model = SimpleNamespace(
                model=SimpleNamespace(enable_taylorseer=False)
            )

        def prepare_image_schedule(self, num_timesteps, timestep_shift, device):
            t = torch.tensor([1.0, 0.4], device=device)
            return t, torch.tensor([0.6, 0.4], device=device)

        def predict_image_velocity(self, **kwargs):
            raise AssertionError("K>0 must not use vanilla _forward_flow")

        def _forward_flow_loop(self, **kwargs):
            velocity_calls.append(kwargs["memory_loop_repeat"])
            assert kwargs["recycle_mode"] == "same_depth"
            memory = torch.ones(2, 4)
            diag = {"memory_rms": 1.0, "vae_hidden_rms": 1.0, "velocity_norm": 1.0}
            return kwargs["x_t"], memory, memory, memory, diag

        def image_euler_step(self, x_t, velocity, dt):
            euler_calls.append(float(dt))
            return x_t

    dummy = Dummy.__new__(Dummy)
    Dummy.__init__(dummy)
    dummy.generate_image = Bagel.generate_image.__get__(dummy, Dummy)
    dummy.generate_image(
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0, 3]),
        packed_init_noises=torch.zeros(2, 4),
        packed_vae_position_ids=torch.zeros(2, dtype=torch.long),
        packed_vae_token_indexes=torch.tensor([1, 2]),
        packed_vae_seqlens=torch.tensor([2], dtype=torch.int),
        packed_boundary_token_indexes=torch.tensor([0, 3]),
        packed_seqlens=torch.tensor([4], dtype=torch.int),
        packed_position_ids=torch.zeros(4, dtype=torch.long),
        packed_indexes=torch.arange(4),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        packed_loop_token_indexes=torch.tensor([1, 2]),
        num_timesteps=2,
        timestep_shift=1.0,
        cfg_interval=(0.0, 1.0),
        enable_taylorseer=False,
        loop_depth=2,
        loop_recycle_mode="same_depth",
        memory_loop_start=1,
        memory_loop_end=2,
    )
    assert velocity_calls == [2, 2]
    assert len(euler_calls) == 2


def test_memory_slot_stats_detect_collapse():
    collapsed = torch.ones(8, 4)
    stats = Bagel.memory_slot_stats(collapsed)
    assert stats["mean_abs_pairwise_cosine"] > 0.99
    assert stats["effective_rank"] < 1.1
    diverse = torch.eye(4, 4)
    diverse_stats = Bagel.memory_slot_stats(diverse)
    assert diverse_stats["effective_rank"] == pytest.approx(3.0, abs=1e-4)
    anti = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    anti_stats = Bagel.memory_slot_stats(anti)
    assert anti_stats["mean_abs_pairwise_cosine"] == pytest.approx(1.0, abs=1e-5)


def test_forward_flow_loop_is_trainable_path_not_no_grad():
    import inspect

    assert not getattr(Bagel._forward_flow_loop, "_is_generator", False)
    source = inspect.getsource(Bagel._forward_flow_loop)
    assert "@torch.no_grad" not in source.split("def _forward_flow_loop", 1)[0][-80:]
    assert Bagel._forward_flow_loop.__dict__.get("_orig_mod", None) is None


def test_bagel_config_defaults_match_plan():
    from qwen_latent_cot.bagel.modeling.bagel.bagel import BagelConfig

    cfg = BagelConfig()
    assert cfg.num_loop_tokens == 8
    assert cfg.loop_depth == 2
    assert cfg.loop_recycle_mode == "same_depth"
    assert cfg.loop_memory_persist is False
    assert cfg.memory_loop_start_layer == 16
    assert cfg.memory_loop_end_layer == 24
    assert cfg.round0_gen_reads_memory is False


def test_bagel_config_exposes_phase0_fields():
    from qwen_latent_cot.bagel.modeling.bagel.bagel import BagelConfig

    cfg = BagelConfig(
        num_loop_tokens=8,
        loop_depth=2,
        loop_uncond_memory="zero",
        loop_recycle_mode="same_depth",
        loop_memory_persist=False,
    )
    assert cfg.num_loop_tokens == 8
    assert cfg.loop_depth == 2
    assert cfg.loop_uncond_memory == "zero"
    assert cfg.loop_recycle_mode == "same_depth"
    assert cfg.loop_memory_persist is False


_DIAG = {
    "memory_rms": 1.0,
    "vae_hidden_rms": 1.0,
    "velocity_norm": 1.0,
    "delta_m": [0.2],
    "delta_g": [0.1],
    "delta_v": [0.05],
}


def _generate_kwargs(**overrides):
    kwargs = {
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.tensor([0, 3]),
        "packed_init_noises": torch.zeros(2, 4),
        "packed_vae_position_ids": torch.zeros(2, dtype=torch.long),
        "packed_vae_token_indexes": torch.tensor([1, 2]),
        "packed_vae_seqlens": torch.tensor([2], dtype=torch.int),
        "packed_boundary_token_indexes": torch.tensor([0, 3]),
        "packed_seqlens": torch.tensor([4], dtype=torch.int),
        "packed_position_ids": torch.zeros(4, dtype=torch.long),
        "packed_indexes": torch.arange(4),
        "past_key_values": object(),
        "key_values_lens": torch.tensor([0], dtype=torch.int),
        "packed_key_value_indexes": torch.tensor([], dtype=torch.long),
        "packed_loop_token_indexes": torch.tensor([1, 2]),
        "num_timesteps": 2,
        "timestep_shift": 1.0,
        "cfg_interval": (0.0, 1.0),
        "enable_taylorseer": False,
        "loop_depth": 2,
        "loop_recycle_mode": "same_depth",
        "memory_loop_start": 1,
        "memory_loop_end": 2,
        "return_trajectory": True,
        "sde_step_indices": (0, 1),
        "sde_noise_level": 0.0,
    }
    kwargs.update(overrides)
    return kwargs


def _make_memory_dummy(*, persist: bool, recycle_mode: str, forward):
    class Dummy(Bagel):
        def __init__(self):
            self.config = SimpleNamespace(
                num_loop_tokens=2,
                loop_depth=2,
                loop_uncond_memory="m0",
                loop_recycle_mode=recycle_mode,
                loop_memory_persist=persist,
                memory_loop_start_layer=1,
                memory_loop_end_layer=2,
            )
            self.loop_memory = torch.zeros(2, 4)
            self.loop_memory_persist = persist
            self.last_loop_diagnostics = []
            self.language_model = SimpleNamespace(
                model=SimpleNamespace(enable_taylorseer=False)
            )

        def prepare_image_schedule(self, num_timesteps, timestep_shift, device):
            t = torch.tensor([1.0, 0.4], device=device)
            return t, torch.tensor([0.6, 0.4], device=device)

        def predict_image_velocity(self, **kwargs):
            raise AssertionError("K>0 must not use vanilla _forward_flow")

        def _forward_flow_loop(self, **kwargs):
            return forward(kwargs)

        def image_euler_step(self, x_t, velocity, dt):
            return x_t

    dummy = Dummy.__new__(Dummy)
    Dummy.__init__(dummy)
    dummy.generate_image = Bagel.generate_image.__get__(dummy, Dummy)
    return dummy


def test_same_depth_body_recurrence_preserves_nonmemory_and_recycles_memory():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Model

    class FakeLayer:
        def __init__(self, idx: int):
            self.idx = idx
            self.inputs = []
            self.outputs = []

        def forward_inference(self, packed_query_sequence, **kwargs):
            self.inputs.append(packed_query_sequence.detach().clone())
            out = packed_query_sequence + float(self.idx + 1)
            self.outputs.append(out.detach().clone())
            return out, kwargs.get("past_key_values")

    class FakeNavit:
        def __init__(self):
            self.layers = [FakeLayer(i) for i in range(4)]
            self.use_moe = False
            self.norm = lambda hidden: hidden
            self.gradient_checkpointing = False
            self.training = False
            self.enable_taylorseer = False

        def rotary_emb(self, seq, pos):
            zeros = torch.zeros(1, seq.shape[0], seq.shape[1], dtype=seq.dtype)
            ones = torch.ones(1, seq.shape[0], seq.shape[1], dtype=seq.dtype)
            return ones, zeros

    navit = FakeNavit()
    seq = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
        ]
    )
    mem = torch.tensor([1], dtype=torch.long)
    nonmem = torch.tensor([0, 2, 3], dtype=torch.long)
    out = Qwen2Model.forward_inference(
        navit,
        packed_query_sequence=seq.clone(),
        query_lens=torch.tensor([4], dtype=torch.int),
        packed_query_position_ids=torch.arange(4),
        packed_query_indexes=torch.arange(4),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        update_past_key_values=False,
        is_causal=False,
        packed_memory_token_indexes=mem,
        memory_loop_repeat=2,
        memory_loop_start=1,
        memory_loop_end=3,
    )
    counts = [len(layer.inputs) for layer in navit.layers]
    assert counts == [1, 2, 2, 2]
    h_base = navit.layers[0].outputs[0]
    round1_body_out = navit.layers[2].outputs[0]
    round2_body_in = navit.layers[1].inputs[1]
    round2_body_out = navit.layers[2].outputs[1]
    assert torch.equal(round2_body_in[nonmem], h_base[nonmem])
    assert torch.equal(round2_body_in[mem], round1_body_out[mem])
    assert torch.equal(out.memory_body_out, round2_body_out[mem])
    assert not torch.equal(out.packed_query_sequence[mem], out.memory_body_out)

    custom = torch.tensor([[7.0, 8.0]])
    navit2 = FakeNavit()
    out2 = Qwen2Model.forward_inference(
        navit2,
        packed_query_sequence=seq.clone(),
        query_lens=torch.tensor([4], dtype=torch.int),
        packed_query_position_ids=torch.arange(4),
        packed_query_indexes=torch.arange(4),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        update_past_key_values=False,
        is_causal=False,
        packed_memory_token_indexes=mem,
        memory_loop_repeat=2,
        memory_loop_start=1,
        memory_loop_end=3,
        memory_body_in=custom,
    )
    round0_body_in = navit2.layers[1].inputs[0]
    assert torch.equal(round0_body_in[nonmem], navit2.layers[0].outputs[0][nonmem])
    assert torch.equal(round0_body_in[mem], custom)
    assert out2.memory_body_out is not None


def test_cfg_branches_keep_independent_recurrent_memory():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import BaseNavitOutputWithPast

    branch_calls = []

    class Dummy(Bagel):
        def __init__(self):
            self.hidden_size = 4
            self.use_moe = True
            self.config = SimpleNamespace(num_loop_tokens=1, loop_depth=2)

            def embed_tokens(ids):
                return torch.ones(int(ids.numel()), 4)

            def forward_inference(**kwargs):
                branch_calls.append(
                    {
                        "kv": kwargs["past_key_values"],
                        "body_in": (
                            None
                            if kwargs.get("memory_body_in") is None
                            else kwargs["memory_body_in"].detach().clone()
                        ),
                        "loop_embed": kwargs["packed_query_sequence"][
                            kwargs["packed_memory_token_indexes"]
                        ].detach().clone(),
                    }
                )
                seq = kwargs["packed_query_sequence"].clone()
                loop_idx = kwargs["packed_memory_token_indexes"]
                body_in = kwargs.get("memory_body_in")
                memory_out = (
                    seq[loop_idx] + 1
                    if body_in is None
                    else body_in + 1
                )
                seq[loop_idx] = memory_out + 50
                return BaseNavitOutputWithPast(
                    packed_query_sequence=seq,
                    past_key_values=kwargs["past_key_values"],
                    memory_body_out=memory_out,
                )

            self.language_model = SimpleNamespace(
                model=SimpleNamespace(embed_tokens=embed_tokens),
                forward_inference=forward_inference,
            )
            self.latent_pos_embed = lambda pos: torch.zeros(int(pos.numel()), 4)
            self.time_embedder = lambda t: torch.zeros(int(t.numel()), 4)
            self.vae2llm = lambda x: torch.zeros(x.shape[0], 4)
            self.llm2vae = lambda h: torch.zeros(h.shape[0], 4)

    dummy = Dummy.__new__(Dummy)
    Dummy.__init__(dummy)
    dummy._forward_flow_loop = Bagel._forward_flow_loop.__get__(dummy, Dummy)
    dummy._combine_cfg_velocities = Bagel._combine_cfg_velocities.__get__(
        dummy, Dummy
    )
    dummy.mot_und_route_indexes = staticmethod(Bagel.mot_und_route_indexes)
    dummy.memory_slot_stats = staticmethod(Bagel.memory_slot_stats)

    kv_full, kv_text, kv_img = object(), object(), object()
    m_full_in = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    m_text_in = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    m_img_in = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    embed = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    _, m_full, m_text, m_img, _ = dummy._forward_flow_loop(
        x_t=torch.zeros(1, 4),
        timestep=torch.tensor([0.7]),
        packed_vae_token_indexes=torch.tensor([2]),
        packed_vae_position_ids=torch.zeros(1, dtype=torch.long),
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0, 3]),
        packed_indexes=torch.arange(4),
        packed_position_ids=torch.zeros(4, dtype=torch.long),
        packed_seqlens=torch.tensor([4], dtype=torch.int),
        key_values_lens=torch.tensor([0], dtype=torch.int),
        past_key_values=kv_full,
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        packed_loop_token_indexes=torch.tensor([1]),
        loop_memory=embed,
        loop_memory_text=embed,
        loop_memory_img=embed,
        packed_boundary_token_indexes=torch.tensor([0, 3]),
        cfg_text_scale=2.0,
        cfg_img_scale=2.0,
        cfg_text_packed_position_ids=torch.zeros(4, dtype=torch.long),
        cfg_text_packed_query_indexes=torch.arange(4),
        cfg_text_key_values_lens=torch.tensor([0], dtype=torch.int),
        cfg_text_past_key_values=kv_text,
        cfg_text_packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        cfg_img_packed_position_ids=torch.zeros(4, dtype=torch.long),
        cfg_img_packed_query_indexes=torch.arange(4),
        cfg_img_key_values_lens=torch.tensor([0], dtype=torch.int),
        cfg_img_past_key_values=kv_img,
        cfg_img_packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        recycle_mode="same_depth",
        memory_loop_repeat=2,
        memory_loop_start=1,
        memory_loop_end=3,
        memory_body_in=m_full_in,
        memory_body_in_text=m_text_in,
        memory_body_in_img=m_img_in,
        embed_memory=embed,
    )
    assert [row["kv"] for row in branch_calls] == [kv_full, kv_text, kv_img]
    assert torch.equal(branch_calls[0]["body_in"], m_full_in)
    assert torch.equal(branch_calls[1]["body_in"], m_text_in)
    assert torch.equal(branch_calls[2]["body_in"], m_img_in)
    assert not torch.equal(m_full, m_text)
    assert not torch.equal(m_full, m_img)
    assert not torch.equal(m_text, m_img)
    assert torch.equal(m_full, m_full_in + 1)
    assert torch.equal(m_text, m_text_in + 1)
    assert torch.equal(m_img, m_img_in + 1)
    assert m_full.data_ptr() != m_text.data_ptr()
    assert m_full.data_ptr() != m_img.data_ptr()
    assert m_text.data_ptr() != m_img.data_ptr()
    m_full.add_(1)
    assert torch.equal(m_text, m_text_in + 1)
    assert torch.equal(m_img, m_img_in + 1)


def test_persist_true_carries_m_out_to_next_m_in_and_keeps_cfg_memories_apart():
    recorded = []
    step = [0]

    def forward(kwargs):
        m_full = torch.full((2, 4), float(step[0] + 1))
        m_text = torch.full((2, 4), float(step[0] + 10))
        m_img = torch.full((2, 4), float(step[0] + 100))
        recorded.append(
            {
                "body": kwargs["memory_body_in"],
                "text": kwargs["memory_body_in_text"],
                "img": kwargs["memory_body_in_img"],
            }
        )
        step[0] += 1
        return kwargs["x_t"], m_full, m_text, m_img, dict(_DIAG)

    dummy = _make_memory_dummy(persist=True, recycle_mode="same_depth", forward=forward)
    _, traj = dummy.generate_image(**_generate_kwargs(loop_memory_persist=True))
    assert recorded[0]["body"] is None
    assert torch.equal(recorded[1]["body"], torch.full((2, 4), 1.0))
    assert torch.equal(recorded[1]["text"], torch.full((2, 4), 10.0))
    assert torch.equal(recorded[1]["img"], torch.full((2, 4), 100.0))
    assert not torch.equal(recorded[1]["body"], recorded[1]["text"])
    assert not torch.equal(recorded[1]["text"], recorded[1]["img"])
    assert torch.equal(traj[0]["m_out"], traj[1]["m_in"])
    assert traj[1]["m_in"] is not None


def test_persist_false_resets_next_step_but_logs_current_m_out():
    recorded = []
    step = [0]

    def forward(kwargs):
        m_full = torch.full((2, 4), float(step[0] + 1))
        m_text = torch.full((2, 4), float(step[0] + 10))
        m_img = torch.full((2, 4), float(step[0] + 100))
        recorded.append(kwargs["memory_body_in"])
        step[0] += 1
        return kwargs["x_t"], m_full, m_text, m_img, dict(_DIAG)

    dummy = _make_memory_dummy(
        persist=False, recycle_mode="same_depth", forward=forward
    )
    _, traj = dummy.generate_image(**_generate_kwargs(loop_memory_persist=False))
    assert recorded[0] is None
    assert recorded[1] is None
    assert traj[0]["m_out"] is not None
    assert torch.equal(traj[0]["m_out"], torch.full((2, 4), 1.0))
    assert traj[1]["m_in"] is None
    assert traj[1]["m_out"] is not None
    assert torch.equal(traj[1]["m_out"], torch.full((2, 4), 2.0))
    assert dummy.last_loop_diagnostics
    for row in dummy.last_loop_diagnostics:
        assert "delta_m" in row
        assert "delta_g" in row
        assert "delta_v" in row


class TinySdpaLayer:
    def __init__(self):
        self.blocks = []

    def forward_inference(
        self,
        packed_query_sequence,
        query_lens,
        past_key_values=None,
        packed_vae_token_indexes=None,
        packed_memory_token_indexes=None,
        block_gen_reads_memory=False,
        **kwargs,
    ):
        from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import (
            _sdpa_varlen_inference,
            round0_blocked_slices,
        )

        self.blocks.append(bool(block_gen_reads_memory))
        qkv = packed_query_sequence.unsqueeze(1)
        kv_lens = query_lens
        blocked = None
        if bool(block_gen_reads_memory):
            blocked = round0_blocked_slices(
                query_lens,
                kv_lens,
                packed_vae_token_indexes,
                packed_memory_token_indexes,
            )
        attn = _sdpa_varlen_inference(
            query=qkv,
            key=qkv,
            value=qkv,
            query_lens=query_lens,
            key_value_lens=kv_lens,
            causal=False,
            blocked_slices=blocked,
        )
        return attn.squeeze(1), past_key_values


class TinySdpaNavit:
    def __init__(self, n_layers: int = 3):
        self.layers = [TinySdpaLayer() for _ in range(n_layers)]
        self.use_moe = True
        self.gradient_checkpointing = False
        self.training = False
        self.enable_taylorseer = False

    def rotary_emb(self, seq, pos):
        zeros = torch.zeros(1, seq.shape[0], seq.shape[1], dtype=seq.dtype)
        ones = torch.ones(1, seq.shape[0], seq.shape[1], dtype=seq.dtype)
        return ones, zeros

    def norm(self, hidden):
        return hidden

    def norm_moe_gen(self, hidden):
        return hidden


def _sdpa_layout():
    mem = torch.tensor([1], dtype=torch.long)
    gen = torch.tensor([2], dtype=torch.long)
    text = torch.tensor([0, 1, 3], dtype=torch.long)
    seq = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return seq, mem, gen, text


def _run_tiny_sdpa(seq, *, block: bool, repeat: int = 1, body: bool = True, n_layers: int = 3):
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Model

    navit = TinySdpaNavit(n_layers)
    _, mem, gen, text = _sdpa_layout()
    kwargs = dict(
        packed_query_sequence=seq.clone(),
        query_lens=torch.tensor([4], dtype=torch.int),
        packed_query_position_ids=torch.arange(4),
        packed_query_indexes=torch.arange(4),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=gen,
        packed_text_indexes=text,
        packed_memory_token_indexes=mem,
        block_gen_reads_memory=block,
    )
    if body:
        kwargs.update(
            memory_loop_repeat=repeat,
            memory_loop_start=0,
            memory_loop_end=1,
        )
    return Qwen2Model.forward_inference(navit, **kwargs), navit


def test_round0_blocked_slices_accounts_for_past_and_batch():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import round0_blocked_slices

    query_lens = torch.tensor([4, 4], dtype=torch.int)
    key_lens = torch.tensor([7, 6], dtype=torch.int)
    vae = torch.tensor([2, 6], dtype=torch.long)
    mem = torch.tensor([1, 5], dtype=torch.long)
    slices = round0_blocked_slices(query_lens, key_lens, vae, mem)
    gen0, mem0 = slices[0]
    gen1, mem1 = slices[1]
    assert list(gen0.tolist()) == [2]
    assert list(mem0.tolist()) == [3 + 1]
    assert list(gen1.tolist()) == [2]
    assert list(mem1.tolist()) == [2 + 1]


def test_sdpa_round0_block_and_read():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import (
        _sdpa_varlen_inference,
        round0_blocked_slices,
    )

    query_lens = torch.tensor([4], dtype=torch.int)
    mem = torch.tensor([1], dtype=torch.long)
    gen = torch.tensor([2], dtype=torch.long)
    hidden = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )
    qk = hidden.unsqueeze(1)
    v = hidden.unsqueeze(1)
    blocked = round0_blocked_slices(query_lens, query_lens, gen, mem)
    gen_local, mem_keys = blocked[0]
    assert list(mem_keys.tolist()) == [1]
    v_mem = v.clone()
    v_mem[1] = torch.tensor([[0.0, -1.0]])
    v_gen = v.clone()
    v_gen[2] = torch.tensor([[-1.0, 1.0]])
    out_block = _sdpa_varlen_inference(
        query=qk,
        key=qk,
        value=v,
        query_lens=query_lens,
        key_value_lens=query_lens,
        causal=False,
        blocked_slices=blocked,
    ).squeeze(1)
    out_block_mem = _sdpa_varlen_inference(
        query=qk,
        key=qk,
        value=v_mem,
        query_lens=query_lens,
        key_value_lens=query_lens,
        causal=False,
        blocked_slices=blocked,
    ).squeeze(1)
    out_block_gen = _sdpa_varlen_inference(
        query=qk,
        key=qk,
        value=v_gen,
        query_lens=query_lens,
        key_value_lens=query_lens,
        causal=False,
        blocked_slices=blocked,
    ).squeeze(1)
    assert torch.allclose(out_block[2], out_block_mem[2], atol=1e-5)
    assert not torch.allclose(out_block[1], out_block_gen[1], atol=1e-4)
    out_open = _sdpa_varlen_inference(
        query=qk,
        key=qk,
        value=v,
        query_lens=query_lens,
        key_value_lens=query_lens,
        causal=False,
        blocked_slices=None,
    ).squeeze(1)
    out_open_mem = _sdpa_varlen_inference(
        query=qk,
        key=qk,
        value=v_mem,
        query_lens=query_lens,
        key_value_lens=query_lens,
        causal=False,
        blocked_slices=None,
    ).squeeze(1)
    assert not torch.allclose(out_open[2], out_open_mem[2], atol=1e-4)

    past = 3
    key_lens = torch.tensor([4 + past], dtype=torch.int)
    past_kv = torch.ones(past, 1, 2)
    key = torch.cat([past_kv, qk], dim=0)
    value = torch.cat([past_kv, v], dim=0)
    slices = round0_blocked_slices(query_lens, key_lens, gen, mem)
    assert list(slices[0][1].tolist()) == [past + 1]
    out_past = _sdpa_varlen_inference(
        query=qk,
        key=key,
        value=value,
        query_lens=query_lens,
        key_value_lens=key_lens,
        causal=False,
        blocked_slices=slices,
    ).squeeze(1)
    value_mem = value.clone()
    value_mem[past + 1] = torch.tensor([[0.0, -1.0]])
    out_past_mem = _sdpa_varlen_inference(
        query=qk,
        key=key,
        value=value_mem,
        query_lens=query_lens,
        key_value_lens=key_lens,
        causal=False,
        blocked_slices=slices,
    ).squeeze(1)
    assert torch.allclose(out_past[2], out_past_mem[2], atol=1e-5)
    value_gen = value.clone()
    value_gen[past + 2] = torch.tensor([[-1.0, 1.0]])
    out_past_gen = _sdpa_varlen_inference(
        query=qk,
        key=key,
        value=value_gen,
        query_lens=query_lens,
        key_value_lens=key_lens,
        causal=False,
        blocked_slices=slices,
    ).squeeze(1)
    assert not torch.allclose(out_past[1], out_past_gen[1], atol=1e-4)


def test_same_depth_round0_block_round1_write_and_read():
    seq, _, gen, _ = _sdpa_layout()
    blocked, navit_b = _run_tiny_sdpa(seq, block=True, repeat=2, body=True, n_layers=2)
    assert navit_b.layers[0].blocks == [True, False]
    seq_mem = seq.clone()
    seq_mem[1] = torch.tensor([0.0, -1.0])
    seq_gen = seq.clone()
    seq_gen[2] = torch.tensor([-1.0, 1.0])
    blocked_mem, _ = _run_tiny_sdpa(seq_mem, block=True, repeat=2, body=True, n_layers=2)
    blocked_gen, _ = _run_tiny_sdpa(seq_gen, block=True, repeat=2, body=True, n_layers=2)
    assert torch.allclose(
        blocked.gen_round_hiddens[0][0],
        blocked_mem.gen_round_hiddens[0][0],
        atol=1e-4,
    )
    assert not torch.equal(blocked.memory_body_out, blocked_gen.memory_body_out)
    opened, _ = _run_tiny_sdpa(seq, block=False, repeat=2, body=True, n_layers=2)
    opened_mem, _ = _run_tiny_sdpa(seq_mem, block=False, repeat=2, body=True, n_layers=2)
    assert not torch.allclose(
        opened.gen_round_hiddens[0][0],
        opened_mem.gen_round_hiddens[0][0],
        atol=1e-4,
    )
    assert not torch.equal(blocked.gen_round_hiddens[0], blocked.gen_round_hiddens[1])
    assert blocked.gen_suffix_round_hiddens is not None
    assert len(blocked.gen_suffix_round_hiddens) == 2


def test_full_depth_block_flag_uses_sequential_path():
    seq, _, gen, _ = _sdpa_layout()
    blocked, navit_b = _run_tiny_sdpa(seq, block=True, body=False, n_layers=1)
    assert all(layer.blocks == [True] for layer in navit_b.layers)
    seq_mem = seq.clone()
    seq_mem[1] = torch.tensor([0.0, -1.0])
    seq_gen = seq.clone()
    seq_gen[2] = torch.tensor([-1.0, 1.0])
    blocked_mem, _ = _run_tiny_sdpa(seq_mem, block=True, body=False, n_layers=1)
    blocked_gen, _ = _run_tiny_sdpa(seq_gen, block=True, body=False, n_layers=1)
    assert torch.allclose(
        blocked.packed_query_sequence[gen],
        blocked_mem.packed_query_sequence[gen],
        atol=1e-4,
    )
    mem = torch.tensor([1], dtype=torch.long)
    assert not torch.equal(
        blocked.packed_query_sequence[mem],
        blocked_gen.packed_query_sequence[mem],
    )
    opened, navit_o = _run_tiny_sdpa(seq, block=False, body=False, n_layers=1)
    assert all(layer.blocks == [False] for layer in navit_o.layers)
    opened_mem, _ = _run_tiny_sdpa(seq_mem, block=False, body=False, n_layers=1)
    assert not torch.allclose(
        opened.packed_query_sequence[gen],
        opened_mem.packed_query_sequence[gen],
        atol=1e-4,
    )


def test_diagnostics_populate_deltas_when_r_ge_2():
    from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import BaseNavitOutputWithPast

    class Dummy(Bagel):
        def __init__(self):
            self.hidden_size = 4
            self.use_moe = True
            self.config = SimpleNamespace(num_loop_tokens=1, loop_depth=2)

            def embed_tokens(ids):
                return torch.ones(int(ids.numel()), 4)

            def forward_inference(**kwargs):
                seq = kwargs["packed_query_sequence"].clone()
                loop_idx = kwargs["packed_memory_token_indexes"]
                gen_idx = kwargs["packed_vae_token_indexes"]
                m0 = seq[loop_idx].clone()
                g0 = seq[gen_idx].clone()
                m1 = m0 + 1
                g1 = g0 + 2
                seq[loop_idx] = m1
                seq[gen_idx] = g1
                return BaseNavitOutputWithPast(
                    packed_query_sequence=seq,
                    past_key_values=kwargs["past_key_values"],
                    memory_body_out=m1,
                    memory_round_hiddens=(m0, m1),
                    gen_round_hiddens=(g0, g1),
                    gen_suffix_round_hiddens=(g0, g1),
                )

            self.language_model = SimpleNamespace(
                model=SimpleNamespace(embed_tokens=embed_tokens),
                forward_inference=forward_inference,
            )
            self.latent_pos_embed = lambda pos: torch.zeros(int(pos.numel()), 4)
            self.time_embedder = lambda t: torch.zeros(int(t.numel()), 4)
            self.vae2llm = lambda x: torch.zeros(x.shape[0], 4)
            self.llm2vae = lambda h: h

    dummy = Dummy.__new__(Dummy)
    Dummy.__init__(dummy)
    dummy._forward_flow_loop = Bagel._forward_flow_loop.__get__(dummy, Dummy)
    dummy._combine_cfg_velocities = Bagel._combine_cfg_velocities.__get__(
        dummy, Dummy
    )
    dummy.mot_und_route_indexes = staticmethod(Bagel.mot_und_route_indexes)
    dummy.memory_slot_stats = staticmethod(Bagel.memory_slot_stats)
    dummy.relative_l2 = staticmethod(Bagel.relative_l2)
    _, _, _, _, diag = dummy._forward_flow_loop(
        x_t=torch.zeros(1, 4),
        timestep=torch.tensor([0.7]),
        packed_vae_token_indexes=torch.tensor([2]),
        packed_vae_position_ids=torch.zeros(1, dtype=torch.long),
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0, 3]),
        packed_indexes=torch.arange(4),
        packed_position_ids=torch.zeros(4, dtype=torch.long),
        packed_seqlens=torch.tensor([4], dtype=torch.int),
        key_values_lens=torch.tensor([0], dtype=torch.int),
        past_key_values=object(),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        packed_loop_token_indexes=torch.tensor([1]),
        loop_memory=torch.ones(1, 4),
        recycle_mode="same_depth",
        memory_loop_repeat=2,
        memory_loop_start=1,
        memory_loop_end=3,
        embed_memory=torch.ones(1, 4),
        round0_gen_reads_memory=False,
    )
    assert diag["delta_m"]
    assert diag["delta_g"]
    assert diag["delta_v"]
    assert diag["delta_m"][0] > 0
    assert diag["delta_g"][0] > 0
    assert diag["delta_v"][0] > 0
    base = torch.ones(1, 4)
    assert Bagel.relative_l2(base + 1, base) == pytest.approx(
        float(torch.linalg.vector_norm(torch.ones(1, 4)) / torch.linalg.vector_norm(base)),
        abs=1e-6,
    )


def test_k0_diagnostics_empty():
    dummy = DummyK0()
    dummy.generate_image(
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0, 1]),
        packed_init_noises=torch.zeros(4, 8),
        packed_vae_position_ids=torch.zeros(4, dtype=torch.long),
        packed_vae_token_indexes=torch.arange(4),
        packed_vae_seqlens=torch.tensor([4], dtype=torch.int),
        packed_boundary_token_indexes=torch.tensor([0, 1]),
        packed_seqlens=torch.tensor([2], dtype=torch.int),
        packed_position_ids=torch.zeros(2, dtype=torch.long),
        packed_indexes=torch.arange(2),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        num_timesteps=2,
        timestep_shift=1.0,
        cfg_interval=(0.0, 1.0),
        enable_taylorseer=False,
    )
    assert dummy.last_loop_diagnostics == []


class DummyK0(Bagel):
    def __init__(self):
        self.config = SimpleNamespace(num_loop_tokens=0, loop_depth=1)
        self.loop_memory = None
        self.last_loop_diagnostics = []
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(enable_taylorseer=False)
        )

    def prepare_image_schedule(self, num_timesteps, timestep_shift, device):
        t = torch.tensor([1.0, 0.5], device=device)
        return t, torch.tensor([0.5, 0.5], device=device)

    def predict_image_velocity(self, **kwargs):
        return kwargs["x_t"]

    def image_euler_step(self, x_t, velocity, dt):
        return x_t


def test_full_depth_control_still_runs():
    velocity_calls = []

    def forward(kwargs):
        velocity_calls.append(kwargs.get("round0_gen_reads_memory"))
        memory = torch.ones(2, 4)
        return kwargs["x_t"], memory, memory, memory, dict(_DIAG)

    dummy = _make_memory_dummy(
        persist=False, recycle_mode="full_depth", forward=forward
    )
    dummy.generate_image(
        **_generate_kwargs(
            loop_recycle_mode="full_depth",
            loop_memory_persist=False,
            return_trajectory=False,
            sde_step_indices=(),
        )
    )
    assert len(velocity_calls) == 4
    assert velocity_calls == [False, True, False, True]


def test_generate_image_rejects_removed_loop_state_kwargs():
    dummy = DummyK0()
    kwargs = dict(
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0, 1]),
        packed_init_noises=torch.zeros(4, 8),
        packed_vae_position_ids=torch.zeros(4, dtype=torch.long),
        packed_vae_token_indexes=torch.arange(4),
        packed_vae_seqlens=torch.tensor([4], dtype=torch.int),
        packed_boundary_token_indexes=torch.tensor([0, 1]),
        packed_seqlens=torch.tensor([2], dtype=torch.int),
        packed_position_ids=torch.zeros(2, dtype=torch.long),
        packed_indexes=torch.arange(2),
        past_key_values=None,
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        num_timesteps=2,
        timestep_shift=1.0,
        cfg_interval=(0.0, 1.0),
        enable_taylorseer=False,
    )
    with pytest.raises(TypeError):
        dummy.generate_image(**kwargs, loop_state_scale=0.2)
    with pytest.raises(TypeError):
        dummy.generate_image(**kwargs, loop_state_mode="semantic_token")


def test_filter_old_prompt_keeps_edit_instruction():
    from PIL import Image
    from qwen_latent_cot.bagel.inferencer import filter_old_prompt

    image = Image.new("RGB", (8, 8))
    items = ["old prompt", image, "make it blue"]
    assert filter_old_prompt(items, True) == [image, "make it blue"]
    assert filter_old_prompt(items, False) == items

