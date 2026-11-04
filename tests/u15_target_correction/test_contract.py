"""Contracts that do not load SenseNova: rho=0, history rollback, surrogate value, tiers."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from qwen_latent_cot.u15_target_correction import KIND, TRACE_KIND
from qwen_latent_cot.u15_target_correction.data import Corpus, Sampler
from qwen_latent_cot.u15_target_correction.field import (
    coarse_solve, protocol, rollback, target_rollout, total_velocity, validate_config,
)
from qwen_latent_cot.u15_target_correction.loss import weighted

ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / "configs/u15_proposal_correction/target_stage1.json"
STAGE2 = ROOT / "configs/u15_proposal_correction/target_stage2.json"


def _cfg():
    cfg = json.loads(STAGE1.read_text())
    cfg["window_steps"] = 1
    cfg["tail_steps"] = 1
    return cfg


class Runtime:
    identity = "unit-native"
    width = 8
    height = 8
    device = "cpu"

    def noise_scale(self):
        return 1.0

    def times(self):
        return [i / 10 for i in range(11)]


class Config(unittest.TestCase):
    def test_production_configs_are_deterministic_history(self):
        for path, stage, steps, batch in (
            (STAGE1, 1, 6000, 256),
            (STAGE2, 2, 4000, 128),
        ):
            cfg = json.loads(path.read_text())
            validate_config(cfg)
            self.assertEqual(cfg["kind"], KIND)
            self.assertEqual(cfg["stage"], stage)
            self.assertEqual(cfg["steps"], steps)
            self.assertEqual(cfg["global_batch"], batch)
            self.assertEqual(cfg["rho_deep"], 0)
            self.assertEqual(cfg["rho_shallow"], 0)
            self.assertEqual(cfg["lora_rank"], 64)
            self.assertEqual(cfg["freeze_gen_tail"], 6)
            self.assertEqual((cfg["window_steps"], cfg["tail_steps"]), (5, 3))

    def test_nonzero_rho_and_foreign_kind_are_rejected(self):
        cfg = _cfg()
        cfg["rho_shallow"] = 0.55
        with self.assertRaisesRegex(ValueError, "deterministic history"):
            validate_config(cfg)
        cfg = _cfg()
        cfg["kind"] = "u15_sc_soar_v4"
        with self.assertRaisesRegex(ValueError, "deterministic history"):
            validate_config(cfg)

    def test_protocol_binds_history_identity(self):
        cfg = _cfg()
        bound = protocol(Runtime(), cfg)
        self.assertEqual(bound["kind"], KIND)
        self.assertEqual(bound["rollback"], "history_identity")
        self.assertEqual(bound["bridge"], "target_driven")
        self.assertEqual(bound["prefix"], "image_first")
        self.assertEqual(bound["rho_deep"], 0)
        self.assertEqual(bound["native"], "unit-native")


class _Velocity:
    config = {"t_eps": 0.05}

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, z, t, tau, sketch, proposal):
        return self.fn(z, t, tau, sketch, proposal)


class Field(unittest.TestCase):
    def test_rollback_is_the_real_history_state(self):
        history = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2).requires_grad_()
        noise = torch.randn_like(history)
        out = rollback(None, history, noise, 0.07, "deep", 99, _cfg())
        self.assertFalse(out.requires_grad)
        self.assertTrue(torch.equal(out, history.detach()))
        out[0, 0, 0, 0] = -1
        self.assertEqual(float(history[0, 0, 0, 0]), 0)

    def test_handoff_drops_the_residual_and_detaches_the_proposal(self):
        seen = {}

        def native(z, t):
            return torch.ones_like(z) * 2

        def body(z, t, tau, sketch, proposal):
            seen["proposal"] = proposal
            return torch.ones_like(z) * 3

        corrector = _Velocity(body)
        z = torch.zeros(1, 1, 2, 2, requires_grad=True)
        at_handoff = total_velocity(native, corrector, z, z.new_tensor(0.5), 0.07, 0.5, None)
        self.assertEqual(float(at_handoff.reshape(-1)[0]), 2)
        self.assertNotIn("proposal", seen)
        below = total_velocity(native, corrector, z, z.new_tensor(0.07), 0.07, 0.5, None)
        self.assertEqual(float(below.reshape(-1)[0]), 5)
        self.assertFalse(seen["proposal"].requires_grad)

    def test_coarse_solve_is_euler(self):
        z = torch.zeros((), requires_grad=True)
        out = coarse_solve(z, 0.0, 1.0, 4, lambda x, t: torch.ones_like(x))
        self.assertTrue(torch.allclose(out, torch.ones(())))
        out.sum().backward()
        self.assertEqual(float(z.grad), 1)

    def test_rollout_value_matches_deployment_and_grad_matches_the_coarse_window(self):
        cfg = _cfg()
        cfg["deep_time"] = 0.1
        cfg["shallow_time"] = 0.2
        cfg["read_time"] = 0.3
        cfg["handoff_time"] = 0.5
        validate_config(cfg)
        scale = torch.tensor(0.25, requires_grad=True)
        anchor = torch.zeros(1, 1, 2, 2)

        def native(z, t):
            return torch.zeros_like(z)

        corrector = _Velocity(lambda z, t, tau, sketch, proposal: torch.ones_like(z) * scale)
        result, penalty = target_rollout(Runtime(), cfg, anchor, 0.1, native, corrector, None,
                                          torch.ones(1, 1, 1, 1))
        self.assertTrue(torch.isfinite(penalty))
        from qwen_latent_cot.u15_target_correction.field import integrate, phase_grid, segment_times
        velocity = lambda z, t: total_velocity(native, corrector, z, t, 0.1, 0.5, None)
        with torch.no_grad():
            fine = integrate(anchor, segment_times(phase_grid(Runtime(), cfg), 0.1, 1.0), velocity)
        self.assertTrue(torch.allclose(result, fine))
        result.sum().backward()
        self.assertTrue(torch.allclose(scale.grad, torch.tensor((0.5 - 0.1) * anchor.numel())))


class LossAndSampler(unittest.TestCase):
    def test_weighted_mean_uses_channel_count(self):
        error = torch.ones(1, 2, 2, 2)
        weight = torch.ones(1, 1, 2, 2)
        self.assertEqual(float(weighted(error, weight)), 1)

    def test_sampler_refuses_a_positive_empty_tier_and_is_reproducible(self):
        class Bag:
            rows = [{"tier": "A", "split": "train"}, {"tier": "B", "split": "val"}]

        cfg = {"seed": 17, "tier_weights": {"A": 1.0, "B": 0.0, "zero": 0.0}}
        first = Sampler(Bag(), cfg)
        second = Sampler(Bag(), cfg)
        self.assertEqual(first.take(6), second.take(6))
        self.assertTrue(all(index == 0 for index, seed, fraction in first.take(8)))
        cfg["tier_weights"] = {"A": 0.2, "B": 0.7, "zero": 0.1}
        with self.assertRaisesRegex(ValueError, "Missing required training tier"):
            Sampler(Bag(), cfg)

    def _write_pair(self, root):
        Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "src.png")
        Image.new("RGB", (8, 8), (0, 255, 0)).save(root / "edit.png")

    def test_corpus_rejects_bad_rows_and_keeps_offline_instructions_out_of_the_condition(self):
        cfg = _cfg()
        rt = Runtime()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "records": [
                    {"uid": "a", "family": "f", "split": "train", "approved": True, "tier": "A"},
                    {"uid": "b", "family": "f", "split": "val", "approved": True, "tier": "A"},
                ]
            }
            path = root / "corpus.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                Corpus(path, rt, cfg)
            manifest["records"][0]["approved"] = False
            manifest["records"].pop()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "approved"):
                Corpus(path, rt, cfg)

            self._write_pair(root)
            bound = protocol(rt, cfg)
            tau = cfg["deep_time"]
            trace = {
                "kind": TRACE_KIND,
                "record": {"protocol": bound, "raw_prompt": "a red square", "condition": "ONLINE"},
                "history": {str(tau): torch.zeros(1, 3, 8, 8)},
                "preview": torch.zeros(1, 3, 8, 8),
                "final": torch.ones(1, 3, 8, 8),
                "valid": torch.ones(1, 1, 8, 8),
            }
            torch.save(trace, root / "00000.pt")
            Image.new("RGB", (8, 8), (0, 0, 255)).save(root / "target.png")
            records = {
                "records": [{
                    "uid": "b1", "family": "p", "split": "train", "approved": True, "tier": "B",
                    "trace": "00000.pt", "arm": "deep", "target_image": "target.png",
                    "raw_prompt": "a red square", "target_edit_instruction": "OFFLINE",
                }, {
                    "uid": "z1", "family": "q", "split": "train", "approved": True, "tier": "zero",
                    "trace": "00000.pt", "arm": "deep", "raw_prompt": "a red square",
                }, {
                    "uid": "a1", "family": "s", "split": "train", "approved": True, "tier": "A",
                    "source_image": "src.png", "target_image": "edit.png",
                    "raw_prompt": "scene", "instruction": "make it green",
                }]
            }
            path.write_text(json.dumps(records))
            corpus = Corpus(path, rt, cfg)
            tier_b = corpus.row(0, 3)
            self.assertEqual(tier_b["condition"], "ONLINE")
            self.assertNotIn("OFFLINE", tier_b["condition"])
            self.assertEqual(tuple(tier_b["target"].shape), (1, 3, 8, 8))
            zero = corpus.row(1, 4)
            self.assertTrue(torch.equal(zero["target"], trace["final"]))
            self.assertTrue(zero["zero"])
            tier_a = corpus.row(2, 2)
            self.assertIn("[Semantic Corrections]:\nmake it green", tier_a["condition"])
            self.assertEqual(tier_a["tau"], cfg["deep_time"])
            odd = corpus.row(2, 3)
            self.assertEqual(odd["tau"], cfg["shallow_time"])


if __name__ == "__main__":
    unittest.main()
