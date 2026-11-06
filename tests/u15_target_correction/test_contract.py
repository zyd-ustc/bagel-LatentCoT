"""Contracts that do not load SenseNova or touch an accelerator."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from qwen_latent_cot.u15_target_correction import CACHE_KIND, KIND, PROTOCOL_VERSION
from qwen_latent_cot.u15_target_correction.conditions import derived_tail_condition, edit_prompt
from qwen_latent_cot.u15_target_correction.config import (
    OLD_KEYS, load_config, stage_updates, validate_config,
)
from qwen_latent_cot.u15_target_correction.data import (
    LinearManifest, StatePairManifest, TrainingSampler,
)
from qwen_latent_cot.u15_target_correction.diagnosis import feedback_from_questions
from qwen_latent_cot.u15_target_correction.field import (
    edit_state, euler, native_shift_time, native_unshift_time, preview_image,
    protocol, reverse_heun_states, sample_state_time, time_key,
)
from qwen_latent_cot.u15_target_correction.images import load_pair
from qwen_latent_cot.u15_target_correction.loss import masked_l1, render, state_flow_matching
from qwen_latent_cot.u15_target_correction.model import GENLoRA
from qwen_latent_cot.u15_target_correction.prepare_state_pairs import _accepted

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/u15_isochronic_state"


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
    def test_stage_files_merge_the_shared_core(self):
        first = load_config(CONFIGS / "stage1.json")
        second = load_config(CONFIGS / "stage2.json")
        sample = load_config(CONFIGS / "sample.json")
        self.assertEqual(first["kind"], KIND)
        self.assertEqual(first["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(first["stage"], 1)
        self.assertEqual(second["stage"], 2)
        self.assertEqual(first["state_times"], [0.15, 0.25, 0.36, 0.50])
        self.assertEqual(second["source_weights"]["inverted_pair"], 0.4)
        self.assertNotIn("stage", sample)

    def test_old_settings_are_rejected(self):
        cfg = load_config(CONFIGS / "stage1.json")
        for key in sorted(OLD_KEYS):
            broken = dict(cfg)
            broken[key] = 1
            with self.assertRaisesRegex(ValueError, "Old editor settings"):
                validate_config(broken)

    def test_budgets_follow_the_state_flow_recipe(self):
        first = load_config(CONFIGS / "stage1.json")
        second = load_config(CONFIGS / "stage2.json")
        self.assertEqual(stage_updates(8000, first), 1000)
        self.assertEqual(stage_updates(100000, first), 6000)
        self.assertEqual(stage_updates(20000, second), 2500)
        self.assertEqual(stage_updates(100000, second), 4000)

    def test_time_probabilities_are_sampled_as_declared(self):
        import random

        cfg = load_config(CONFIGS / "stage1.json")
        rng = random.Random(7)
        seen = {}
        for _ in range(20000):
            value = sample_state_time(cfg, rng)
            seen[value] = seen.get(value, 0) + 1
        self.assertEqual(set(seen), set(cfg["state_times"]))
        self.assertAlmostEqual(seen[0.36] / 20000, 0.35, delta=0.02)

    def test_native_shift_round_trip(self):
        self.assertAlmostEqual(native_unshift_time(0.36, 3), 0.6279069767, places=7)
        self.assertAlmostEqual(native_shift_time(0.6279069767, 3), 0.36, places=7)


class Conditions(unittest.TestCase):
    def test_edit_prompt_exposes_the_target_and_optional_preserve(self):
        text = edit_prompt("Make the cup red.", "A red cup on a table.", "Keep the spoon.")
        self.assertEqual(
            text,
            "Edit request: Make the cup red.\n"
            "Generation target after editing: A red cup on a table.\n"
            "Preserve: Keep the spoon.",
        )
        self.assertNotIn("Preserve:", edit_prompt("Make it red.", "A red object."))

    def test_legacy_tail_is_derived_without_the_target_image(self):
        text = derived_tail_condition("A blue cup.", "Make the cup red.")
        self.assertIn("A blue cup.", text)
        self.assertIn("Make the cup red.", text)

    def test_feedback_splits_changes_preserve_and_rejections(self):
        questions = [
            "Are there exactly seven green croissants visible in the scene?",
            "Is each croissant green in color?",
            "Is the kangaroo an elephant?",
        ]
        changes, preserve, rejected = feedback_from_questions(
            questions, [questions[0], questions[2]]
        )
        self.assertIn("Change the number of green croissants to exactly seven.", changes)
        self.assertEqual(preserve, "Keep the croissant green.")
        self.assertEqual(rejected, [questions[2]])


class EditorMath(unittest.TestCase):
    def test_zero_lora_matches_the_native_linear(self):
        base = nn.Linear(3, 2, bias=False)
        adapter = GENLoRA(base, rank=2)
        with torch.no_grad():
            adapter.b.zero_()
        data = torch.randn(4, 3)
        self.assertTrue(torch.allclose(adapter(data), base(data)))

    def test_euler_keeps_state_gradients(self):
        scale = torch.tensor(1.0, requires_grad=True)
        z0 = torch.ones(2, 2)

        def dependent(z, t):
            return scale * z

        result = euler(z0, [0.0, 0.5, 1.0], dependent)
        result.sum().backward()
        self.assertAlmostEqual(float(scale.grad), 1.5 * 4, places=5)

    def test_preview_does_not_mutate_and_uses_three_evaluations(self):
        state = torch.zeros(1, 1, 2, 2)
        before = state.clone()
        calls = []

        def velocity(z, t):
            calls.append(float(t))
            return torch.ones_like(z)

        image = preview_image(state, 0.4, [0.0, 0.4, 0.7, 1.0], velocity)
        self.assertTrue(torch.equal(state, before))
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(float(image.mean()), 0.6, places=5)

    def test_state_edit_is_euler_in_progress(self):
        class Fake:
            def image_clean(self, anchor, t, prefix):
                return torch.zeros_like(anchor)

            def state_velocity(self, state, t, s, prefixes, base):
                return torch.ones_like(state)

        anchor = torch.zeros(1, 3, 2, 2)
        edited, base = edit_state(
            Fake(), anchor, 0.36, {"image": None, "joint": None}, 8, base_prediction=None
        )
        self.assertTrue(torch.equal(base, anchor))
        self.assertAlmostEqual(float(edited.mean()), 1.0, places=5)

    def test_reverse_heun_recovers_a_constant_flow(self):
        class Constant:
            def __call__(self, z, t):
                return torch.ones_like(z)

        grid = [0.0, 0.15, 0.25, 0.36, 0.5, 0.7, 1.0]
        end = euler(torch.zeros(1, 1, 2, 2), grid, Constant())
        states = reverse_heun_states(end, 1.0, [0.15, 0.25, 0.36, 0.5], grid, Constant(), 2)
        for target in (0.15, 0.25, 0.36, 0.5):
            self.assertAlmostEqual(float(states[time_key(target)].mean()), target, places=5)

    def test_state_loss_scales_by_time_and_ignores_padding(self):
        delta = torch.ones(1, 1, 2, 2)
        self.assertEqual(float(state_flow_matching(delta, delta, 0.25)), 0)
        error = state_flow_matching(delta + 0.5, delta, 0.25)
        self.assertAlmostEqual(float(error), 4.0, places=5)
        valid = torch.zeros(1, 1, 2, 2)
        valid[:, :, :, :1] = 1
        wrong = torch.full_like(delta, 100)
        wrong[:, :, :, :1] = 1
        self.assertEqual(float(state_flow_matching(wrong, delta, 0.25, valid)), 0)
        self.assertEqual(float(masked_l1(wrong, delta, valid)), 0)
        self.assertEqual(tuple(render(torch.full((1, 3, 4, 4), 5.0), valid).shape[-2:]), (2, 1))


class Records(unittest.TestCase):
    def _linear(self, root, split="train"):
        Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "source.png")
        Image.new("RGB", (8, 8), (0, 255, 0)).save(root / "target.png")
        return {
            "uid": "a",
            "role": "native_edit",
            "split": split,
            "prompt_family": "p",
            "source_family": "s",
            "source_image": "source.png",
            "target_image": "target.png",
            "edit_condition": "Make the square green.",
            "raw_prompt": "A red square.",
            "semantic_approved": True,
        }

    def test_linear_pairs_build_both_ends_and_keep(self):
        cfg = load_config(CONFIGS / "stage1.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._linear(root)
            (root / "manifest.json").write_text(json.dumps({"records": [row]}))
            corpus = LinearManifest(root / "manifest.json", Runtime(), cfg)
            sample = corpus.sample(0, 0.36)
            self.assertEqual(sample["source"].shape, (1, 3, 8, 8))
            self.assertEqual(sample["edit_request"], "Make the square green.")
            self.assertIn("A red square.", sample["tail_condition"])
            keep = corpus.sample(0, 0.36, keep=True)
            self.assertTrue(torch.equal(keep["source"], keep["target"]))
            self.assertTrue(keep["keep"])

    def test_families_cannot_cross_splits(self):
        cfg = load_config(CONFIGS / "stage1.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._linear(root)
            other = dict(row, uid="b", split="test")
            (root / "manifest.json").write_text(json.dumps({"records": [row, other]}))
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                LinearManifest(root / "manifest.json", Runtime(), cfg)

    def _state_cache(self, root, cfg, rt, source_kind="inverted_pair", accepted=True):
        (root / "states").mkdir()
        states = {}
        for time in cfg["state_times"]:
            path = root / "states" / f"{time_key(time)}.pt"
            torch.save(
                dict(
                    time=time,
                    state_a=torch.zeros(1, 3, 8, 8),
                    state_b=torch.ones(1, 3, 8, 8),
                    preview=torch.zeros(1, 3, 8, 8),
                    base_prediction=None,
                    valid=torch.ones(1, 1, 8, 8),
                ),
                path,
            )
            states[time_key(time)] = dict(path=str(path.relative_to(root)), accepted=accepted)
        row = dict(
            uid="x",
            source_kind=source_kind,
            split="train",
            source_family="s",
            prompt_family="p",
            edit_instruction="Make it green.",
            parent_condition="A red square.",
            tail_condition="A green square.",
            preserve="",
            condition_protocol=cfg["condition_protocol"],
            states=states,
        )
        return dict(kind=CACHE_KIND, protocol=protocol(rt, cfg), records=[row])

    def test_state_pair_cache_uses_fp32_states_and_keep(self):
        cfg = load_config(CONFIGS / "stage2.json")
        rt = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._state_cache(root, cfg, rt)
            (root / "manifest.json").write_text(json.dumps(manifest))
            corpus = StatePairManifest(root, rt, cfg, "inverted_pair")
            sample = corpus.sample(0, 0.36)
            self.assertEqual(sample["state_a"].dtype, torch.float32)
            keep = corpus.sample(0, 0.36, keep=True)
            self.assertTrue(torch.equal(keep["state_a"], keep["state_b"]))
            self.assertEqual(keep["tail_condition"], "A red square.")

    def test_protocol_mismatch_and_split_families_are_rejected(self):
        cfg = load_config(CONFIGS / "stage2.json")
        rt = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._state_cache(root, cfg, rt)
            manifest["protocol"] = dict(manifest["protocol"], read_time=0.5)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "protocol"):
                StatePairManifest(root, rt, cfg, "inverted_pair")

    def test_stage_two_without_native_history_uses_twenty_eighty(self):
        cfg = load_config(CONFIGS / "stage2.json")
        rt = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = root / "inverted"
            state_root.mkdir()
            (state_root / "manifest.json").write_text(
                json.dumps(self._state_cache(state_root, cfg, rt))
            )
            Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "source.png")
            Image.new("RGB", (8, 8), (0, 255, 0)).save(root / "target.png")
            row = self._linear(root)
            (root / "linear.json").write_text(json.dumps({"records": [row]}))
            linear = LinearManifest(root / "linear.json", rt, cfg)
            inverted = StatePairManifest(state_root, rt, cfg, "inverted_pair")
            sampler = TrainingSampler(linear, inverted, None, cfg)
            self.assertTrue(torch.allclose(torch.tensor(list(sampler.actual_weights.values())), torch.tensor([0.2, 0.8])))
            sample = sampler.draw()
            self.assertIn(sample["source_kind"], ("linear_pair", "inverted_pair"))

    def test_cache_acceptance_rejects_nonfinite_and_bad_lpips(self):
        cfg = load_config(CONFIGS / "stage2.json")
        self.assertTrue(_accepted({"l1": 0.01}, cfg, True))
        self.assertFalse(_accepted({"l1": float("nan")}, cfg, True))
        self.assertFalse(_accepted({"l1": 0.01, "lpips": float("inf")}, cfg, True))
        self.assertFalse(_accepted({"l1": 0.01, "lpips": 0.5}, cfg, True))
        self.assertFalse(_accepted({"l1": 0.01}, cfg, False))
        self.assertFalse(_accepted({"l1": 0.5}, cfg, True))

    def test_common_pair_canvas_rejects_independent_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "a.png")
            Image.new("RGB", (4, 8)).save(root / "b.png")
            with self.assertRaisesRegex(ValueError, "share"):
                load_pair(
                    {"source_image": str(root / "a.png"), "target_image": str(root / "b.png")},
                    8, 8, "cpu",
                )
            Image.new("RGB", (4, 4)).save(root / "c.png")
            source, target, valid, rect = load_pair(
                {"source_image": str(root / "a.png"), "target_image": str(root / "c.png")},
                8, 8, "cpu",
            )
            self.assertEqual(source.shape, target.shape)
            self.assertEqual(tuple(valid.shape[-2:]), (8, 8))
