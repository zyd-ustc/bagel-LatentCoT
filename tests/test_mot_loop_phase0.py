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
        "packed_loop_semantic_token_indexes": torch.tensor([], dtype=torch.long),
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
        packed_loop_semantic_token_indexes=torch.tensor([], dtype=torch.long),
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
        packed_loop_semantic_token_indexes=torch.tensor([], dtype=torch.long),
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
    assert stats["pairwise_cosine"] > 0.99
    assert stats["effective_rank"] < 1.1
    diverse = torch.eye(4, 4)
    diverse_stats = Bagel.memory_slot_stats(diverse)
    assert diverse_stats["effective_rank"] > 3.0


def test_forward_flow_loop_is_trainable_path_not_no_grad():
    import inspect

    assert not getattr(Bagel._forward_flow_loop, "_is_generator", False)
    source = inspect.getsource(Bagel._forward_flow_loop)
    assert "@torch.no_grad" not in source.split("def _forward_flow_loop", 1)[0][-80:]
    assert Bagel._forward_flow_loop.__dict__.get("_orig_mod", None) is None


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
