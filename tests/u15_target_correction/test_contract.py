"""Contracts that do not load SenseNova or touch an accelerator."""

import collections
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from qwen_latent_cot.u15_target_correction import (
    CACHE_KIND, GEOMETRY_PROTOCOL, INVERTED_PAIR, KIND, PROTOCOL_VERSION,
)
from qwen_latent_cot.u15_target_correction.conditions import derived_tail_condition, edit_prompt
from qwen_latent_cot.u15_target_correction.config import (
    OLD_KEYS, load_config, stage_updates, validate_config,
)
from qwen_latent_cot.u15_target_correction.data import (
    LinearManifest, StatePairManifest, TrainingSampler,
)
from qwen_latent_cot.u15_target_correction.diagnosis import (
    feedback_from_questions, question_grounded,
)
from qwen_latent_cot.u15_target_correction.field import (
    edit_state, euler, native_shift_time, native_unshift_time, preview_image,
    protocol, reverse_heun_states, sample_state_time, time_key,
)
from qwen_latent_cot.u15_target_correction.images import (
    load_canvas, load_pair, pair_geometry_error,
)
from qwen_latent_cot.u15_target_correction.eval_linear import category, select_cases
from qwen_latent_cot.u15_target_correction.loss import (
    masked_l1, masked_mse, render, state_flow_matching,
)
from qwen_latent_cot.u15_target_correction.model import GENLoRA
from qwen_latent_cot.u15_target_correction.prepare_state_pairs import (
    _accepted, _keep_designation, _refresh_candidate, _rejection, _target_binding_error,
)
from qwen_latent_cot.u15_target_correction.field import discrete_inverse_states, forward_native
import qwen_latent_cot.u15_target_correction.labels as labels_module
import qwen_latent_cot.u15_target_correction.prepare_state_pairs as pairs_module
from qwen_latent_cot.u15_target_correction.explicit import (
    build_request, parse_proposal, proposal_error, question_sets_error,
)


def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
from qwen_latent_cot.u15_target_correction.train import _update

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

    def test_point36_stage_matches_its_sampling_protocol(self):
        train = load_config(CONFIGS / "stage2_36.json")
        sample = load_config(CONFIGS / "sample36.json")
        self.assertEqual(train["state_times"], [0.36])
        self.assertEqual(train["state_time_probs"], [1.0])
        self.assertEqual(train["read_time"], 0.36)
        self.assertEqual(train["keep_fraction"], 0.0)
        self.assertAlmostEqual(sum(train["source_weights"].values()), 1.0)
        self.assertEqual(train["source_weights"]["native_parent"], 0.6)
        rt = Runtime()
        self.assertEqual(protocol(rt, train), protocol(rt, sample))
        self.assertEqual(protocol(rt, train)["geometry"], GEOMETRY_PROTOCOL)

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
        self.assertEqual(stage_updates(8000, first, global_batch=1), 6000)
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

    def test_relation_direction_and_count_are_bound_to_the_prompt(self):
        self.assertFalse(question_grounded("A dog behind a cat", "Is the cat behind the dog?"))
        self.assertTrue(question_grounded("A dog behind a cat", "Is the dog behind the cat?"))
        self.assertTrue(question_grounded("A dog in front of a cat", "Is the cat behind the dog?"))
        self.assertFalse(question_grounded("A dog behind a cat", "Is the dog in front of the cat?"))
        self.assertFalse(question_grounded("A dog behind a cat", "Is there a cat behind a dog?"))
        self.assertTrue(question_grounded("A dog behind a cat", "Is there a dog behind a cat?"))
        self.assertFalse(
            question_grounded(
                "A green backpack and a pig",
                "Are there exactly two pigs visible in the scene?",
            )
        )
        self.assertTrue(
            question_grounded(
                "seven green croissants",
                "Are there exactly seven green croissants visible in the scene?",
            )
        )

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

    def test_other_subject_failure_is_not_an_edit_request(self):
        question = "Is there any other subject or object besides the four monkeys?"
        changes, preserve, rejected = feedback_from_questions([question], [question])
        self.assertEqual(changes, "")
        self.assertEqual(preserve, "")
        self.assertEqual(rejected, [question])


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
            seen = None

            def image_clean(self, anchor, t, prefix):
                return torch.zeros_like(anchor)

            def state_velocity(self, state, t, s, joint_prefix, base):
                self.seen = joint_prefix
                return torch.ones_like(state)

        anchor = torch.zeros(1, 3, 2, 2)
        editor = Fake()
        joint = ("joint-cache",)
        edited, base = edit_state(
            editor, anchor, 0.36, {"image": None, "joint": joint}, 8, base_prediction=None
        )
        self.assertEqual(editor.seen, joint)
        self.assertTrue(torch.equal(base, anchor))
        self.assertAlmostEqual(float(edited.mean()), 1.0, places=5)

    def test_training_update_passes_the_joint_prefix_directly(self):
        class Fake:
            seen = None

            def prefixes(self, preview, edit_request, tail_condition, preserve=""):
                return {"image": ("image-cache",), "joint": ("joint-cache",)}

            def image_clean(self, anchor, t, prefix):
                return torch.zeros_like(anchor)

            def state_velocity(self, state, t, s, joint_prefix, base):
                self.seen = joint_prefix
                return torch.zeros_like(state)

        editor = Fake()
        sample = dict(
            source_kind=INVERTED_PAIR,
            state_a=torch.zeros(1, 3, 2, 2),
            state_b=torch.ones(1, 3, 2, 2),
            preview=torch.zeros(1, 3, 2, 2),
            base_prediction=None,
            valid=torch.ones(1, 1, 2, 2),
            time=0.36,
            edit_request="Make it green.",
            tail_condition="A green square.",
            preserve="",
            keep=False,
            uid="unit",
        )
        loss, _ = _update(editor, sample, load_config(CONFIGS / "stage1.json"), [0.0, 0.36, 1.0])
        self.assertEqual(editor.seen, ("joint-cache",))
        self.assertGreaterEqual(float(loss), 0)

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


class LinearEvaluation(unittest.TestCase):
    def test_validation_selection_is_category_balanced(self):
        class Manifest:
            rows = [
                dict(split="train", edit_instruction="Change the color to red."),
                dict(split="val", edit_instruction="Change the color to red."),
                dict(split="val", edit_instruction="Add a cup."),
                dict(split="val", edit_instruction="Remove the cup."),
                dict(split="val", edit_instruction="Move the cup left."),
                dict(split="val", edit_instruction="Replace the text 'STOP'."),
            ]

        self.assertEqual(category("Change the color to red."), "attribute")
        self.assertEqual(category("Add a cup."), "add")
        self.assertEqual(category("Remove the cup."), "remove")
        self.assertEqual(category("Move the cup left."), "spatial")
        self.assertEqual(category("Replace the text 'STOP'."), "text")
        selected = select_cases(Manifest(), 4, 17)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(Manifest.rows[index]["split"] == "val" for index, _ in selected))
        self.assertEqual(len({name for _, name in selected}), 4)

    def test_masked_mse_uses_the_valid_region(self):
        prediction = torch.zeros(1, 1, 2, 2)
        target = torch.ones_like(prediction)
        valid = torch.zeros_like(prediction)
        valid[:, :, :, :1] = 1
        self.assertEqual(float(masked_mse(prediction, target, valid)), 1.0)
        prediction[:, :, :, :1] = 1
        self.assertEqual(float(masked_mse(prediction, target, valid)), 0.0)


class StageTwoConfig(unittest.TestCase):
    def test_stage_two_accepts_the_no_inverted_recipe(self):
        from qwen_latent_cot.u15_target_correction.config import load_config

        cfg = load_config(CONFIGS / "stage2_v5.json")
        self.assertEqual(cfg["role_weights"]["I_edit"], 0.0)
        self.assertGreater(cfg["role_weights"]["R_edit"], 0.0)
        self.assertEqual(cfg["source_weights"]["inverted_pair"], 0.0)
        self.assertGreater(cfg["source_weights"]["native_parent"], 0.0)

    def test_stage_two_still_needs_a_real_state_edit_role(self):
        from qwen_latent_cot.u15_target_correction.config import load_config, validate_config

        base = load_config(CONFIGS / "stage2_v5.json")
        broken = dict(base, role_weights={"L_edit": 0.5, "I_edit": 0.0, "R_edit": 0.0, "R_keep": 0.5})
        with self.assertRaises(ValueError):
            validate_config(broken)


class CoupledPairs(unittest.TestCase):
    """Two trajectories from one noise draw, no inversion in the label chain."""

    class _Runtime:
        identity = "unit"
        width = 8
        height = 8
        device = "cpu"

        def noise_scale(self):
            return 1.0

        def times(self):
            return [i / 10 for i in range(11)]

    def test_trajectory_snapshots_and_continuation_share_the_state(self):
        import qwen_latent_cot.u15_target_correction.coupled as coupled
        original = coupled.NativeT2I
        coupled.NativeT2I = lambda rt, prompt, cfg: (lambda z, t: torch.zeros_like(z))
        try:
            cfg = dict(state_times=[0.36], read_time=0.36)
            noise = torch.full((1, 1, 2, 2), 0.5)
            out = coupled.run_trajectory(self._Runtime(), cfg, "prompt", noise,
                                        [0.0, 0.36, 0.7, 1.0], [0.36], 0.36)
            self.assertIn(time_key(0.36), out["states"])
            self.assertIn(time_key(0.36), out["previews"])
            self.assertAlmostEqual(float(out["states"][time_key(0.36)].mean()), 0.5)
            tail = coupled.continue_from(self._Runtime(), cfg, out["states"][time_key(0.36)],
                                        "prompt", [0.0, 0.36, 0.7, 1.0], 0.36)
            self.assertAlmostEqual(float(tail.mean()), 0.5)
        finally:
            coupled.NativeT2I = original


class EffectiveAcceptance(unittest.TestCase):
    """The label is judged on the image the frozen tail renders, not the candidate."""

    class _Scorer:
        def __init__(self, table):
            self.table = table

        def __call__(self, rt, image, questions, yes_ids=None, no_ids=None):
            pattern = self.table[image]
            scores = [
                dict(margin=1.0 if question in pattern else -1.0,
                     p_yes=0.9 if question in pattern else 0.1,
                     log_p_yes=-0.1 if question in pattern else -2.3)
                for question in questions
            ]
            return dict(questions=scores)

    def _run(self, table, request, parent="parent", effective="effective"):
        original = labels_module.und_score_image
        labels_module.und_score_image = self._Scorer(table)
        try:
            return labels_module.effective_acceptance(
                None, parent, effective, request, {"pass_threshold": 0.65}
            )
        finally:
            labels_module.und_score_image = original

    def test_explicit_edit_is_judged_on_the_continued_image(self):
        request = dict(task_type="explicit_edit",
                       edit_questions=["Is the mug blue?"],
                       preserve_questions=["Is there a desk?", "Is a lamp present?"],
                       passed_before_source=["Is there a desk?"])
        # The source already satisfies the desk requirement but not the colour one;
        # the lamp is protected but was never satisfied, so it cannot be damaged.
        approved = self._run({"effective": {"Is the mug blue?", "Is there a desk?"}}, request)
        self.assertTrue(approved["approved"])
        self.assertTrue(approved["edit_complete"])
        self.assertTrue(approved["edit_needed"])
        self.assertEqual(approved["protected"], 1)
        self.assertEqual(approved["broken_questions"], [])

    def test_explicit_edit_rejects_incomplete_or_unneeded_or_damaging(self):
        request = dict(task_type="explicit_edit",
                       edit_questions=["Is the mug blue?"],
                       preserve_questions=["Is there a desk?"],
                       passed_before_source=["Is there a desk?"])
        incomplete = self._run({"effective": {"Is there a desk?"}}, request)
        self.assertFalse(incomplete["approved"])
        self.assertEqual(incomplete["reason"], "edit_incomplete")
        unneeded = self._run({"effective": {"Is the mug blue?", "Is there a desk?"},
                              "parent": {"Is the mug blue?", "Is there a desk?"}},
                             dict(request, passed_before_source=[]))
        self.assertFalse(unneeded["edit_needed"])
        self.assertEqual(unneeded["reason"], "edit_not_needed")
        damaged = self._run({"effective": {"Is the mug blue?"}}, request)
        self.assertEqual(damaged["broken_questions"], ["Is there a desk?"])
        self.assertEqual(damaged["reason"], "preserve_broken")

    def test_self_correction_needs_one_repair_and_no_damage(self):
        request = dict(task_type="self_correct",
                       questions=["Is the mug blue?", "Is there a desk?"],
                       passed_before_source=["Is there a desk?"])
        good = self._run({"effective": {"Is the mug blue?", "Is there a desk?"}}, request)
        self.assertTrue(good["approved"])
        self.assertEqual(good["repaired_questions"], ["Is the mug blue?"])
        damaged = self._run({"effective": {"Is the mug blue?"}}, request)
        self.assertFalse(damaged["approved"])
        self.assertEqual(damaged["broken_questions"], ["Is there a desk?"])
        nochange = self._run({"effective": {"Is there a desk?"}}, request)
        self.assertEqual(nochange["reason"], "no_repair")


class ExplicitEdits(unittest.TestCase):
    ANSWER = (
        "EDIT: Change the largest mug to blue.\n"
        "TARGET: A blue ceramic mug sits on a wooden desk beside a lamp.\n"
        "EDIT_QUESTIONS:\n- Is the largest mug blue?\n- Is the mug colour changed?\n"
        "PRESERVE_QUESTIONS:\n- Is there a wooden desk?\n- Is a lamp present?\n"
    )

    def test_proposal_parses_the_four_blocks(self):
        parsed = parse_proposal(self.ANSWER)
        self.assertEqual(parsed["instruction"], "Change the largest mug to blue.")
        self.assertEqual(parsed["target_condition"],
                         "A blue ceramic mug sits on a wooden desk beside a lamp.")
        self.assertEqual(len(parsed["edit_questions"]), 2)
        self.assertEqual(len(parsed["preserve_questions"]), 2)
        self.assertTrue(all(q.endswith("?") for q in parsed["edit_questions"]))
        self.assertIsNone(parse_proposal("no blocks here")["instruction"])

    def test_edit_and_preserve_questions_must_be_separate(self):
        self.assertIsNone(question_sets_error(["Is the mug blue?"], ["Is there a desk?"]))
        self.assertEqual(question_sets_error([], ["Is there a desk?"]), "no_edit_questions")
        self.assertEqual(question_sets_error(["Is the mug blue?"], []), "no_preserve_questions")
        self.assertEqual(
            question_sets_error(["Is the mug blue?"], ["Is the mug blue?"]), "question_overlap"
        )
        self.assertEqual(
            question_sets_error(["Is it blue?"], ["Is there a wooden desk?"]), "question_length"
        )
        self.assertEqual(
            question_sets_error(["Is the mug blue?", "Are there two mugs?", "Is it red?"],
                                ["Is there a desk?"]),
            "too_many_edit_questions",
        )

    def test_target_must_be_a_minimal_update_of_the_source(self):
        from qwen_latent_cot.u15_target_correction.explicit import target_rewrite_error

        source = "A red mug sits on a wooden desk beside a lamp."
        self.assertIsNone(
            target_rewrite_error(source, "A blue mug sits on a wooden desk beside a lamp.")
        )
        self.assertEqual(
            target_rewrite_error(source, "A blue mug on a table in a bright room with plants."),
            "target_rewrite",
        )

    def test_proposal_validation_rejects_unusable_edits(self):
        raw = "A red mug on a wooden desk with a lamp."
        self.assertIsNone(proposal_error(raw, "Change the mug to blue.", "A blue mug on a wooden desk."))
        self.assertEqual(proposal_error(raw, "", "A blue mug."), "unparsed")
        long_instruction = ("Make it much better and nicer and brighter and warmer and softer "
                            "and cleaner and sharper and lovelier")
        self.assertEqual(
            proposal_error(raw, long_instruction, "A blue mug on a wooden desk."), "instruction_length"
        )
        self.assertEqual(proposal_error(raw, "Keep the mug red.", "A blue mug on a desk."), "instruction_form")
        self.assertEqual(proposal_error(raw, "Change the mug to blue.", "A blue mug."), "target_length")
        self.assertEqual(
            proposal_error(raw, "Change the mug to blue.", raw), "target_equals_request"
        )

    def test_explicit_request_carries_both_question_sets(self):
        row = build_request(
            uid="u1", seed=17, parent_branch_id="17:t2i", source_image="/x/final.png",
            raw="A red mug on a desk.", instruction="Change the mug to blue.",
            target_condition="A blue mug on a desk.",
            edit_questions=["Is the mug blue?"], preserve_questions=["Is there a desk?"],
        )
        self.assertEqual(row["task_type"], "explicit_edit")
        self.assertEqual(row["tail_condition"], "A blue mug on a desk.")
        self.assertEqual(row["edit_questions"], ["Is the mug blue?"])
        self.assertEqual(row["preserve_questions"], ["Is there a desk?"])

    def test_explicit_approval_requires_needed_complete_and_undamaged(self):
        targets = _load_script("produce_r_targets")
        edit = ["Is the mug blue?", "Are there two mugs?"]
        keep = ["Is there a desk?", "Is a lamp present?"]
        clean = targets._explicit_approval(
            edit, keep,
            {"passed": keep + ["Is the mug blue?"]},
            {"passed": keep + edit},
        )
        self.assertTrue(clean["approved"])
        self.assertTrue(clean["edit_complete"])
        self.assertTrue(clean["edit_needed"])
        self.assertEqual(clean["broken"], [])
        noop = targets._explicit_approval(
            edit, keep, {"passed": keep + edit}, {"passed": keep + edit}
        )
        self.assertFalse(noop["edit_needed"])
        self.assertFalse(noop["approved"])
        partial = targets._explicit_approval(
            edit, keep, {"passed": keep}, {"passed": keep + ["Is the mug blue?"]}
        )
        self.assertFalse(partial["edit_complete"])
        self.assertFalse(partial["approved"])
        damaged = targets._explicit_approval(
            edit, keep, {"passed": keep + edit}, {"passed": edit + ["Is there a desk?"]}
        )
        self.assertEqual(damaged["broken"], ["Is a lamp present?"])
        self.assertFalse(damaged["approved"])
        unprotective = targets._explicit_approval(
            edit, keep, {"passed": ["Is there a desk?"]}, {"passed": edit}
        )
        self.assertEqual(unprotective["protected"], 1)
        self.assertFalse(unprotective["approved"])


class Labels(unittest.TestCase):
    """Acceptance semantics for cache and target labels."""

    def test_target_approval_separates_repair_from_damage(self):
        targets = _load_script("produce_r_targets")
        verdict = targets._approval(["a", "b"], ["a", "c"])
        self.assertEqual(verdict["repaired"], ["c"])
        self.assertEqual(verdict["broken"], ["b"])
        self.assertFalse(verdict["approved"])
        clean = targets._approval(["a"], ["a", "c"])
        self.assertEqual(clean["broken"], [])
        self.assertTrue(clean["approved"])
        self.assertFalse(targets._approval(["a"], ["a"])["approved"])
        damaged = targets._approval(["a", "b"], ["c"])
        self.assertEqual(damaged["broken"], ["a", "b"])
        self.assertFalse(damaged["approved"])

    def test_selection_never_promotes_a_rejected_record(self):
        selection = _load_script("select_accepted")
        def entry(uid, accepted, source_l1, target_l1, lpips=0.1):
            return dict(uid=uid, key="0.360000", score=source_l1 + target_l1 + 0.5 * lpips,
                        item=dict(accepted=accepted,
                                  source_reconstruction=dict(l1=source_l1),
                                  target_reconstruction=dict(l1=target_l1, lpips=lpips)))
        entries = [entry("bad", False, 0.001, 0.001), entry("ok1", True, 0.05, 0.06),
                   entry("ok2", True, 0.02, 0.03)]
        qualified, selected = selection._select(entries, 10, {})
        self.assertEqual({e["uid"] for e in qualified}, {"ok1", "ok2"})
        self.assertEqual({e["uid"] for e in selected}, {"ok1", "ok2"})
        self.assertNotIn("bad", {e["uid"] for e in selected})
        _, capped = selection._select(entries, 1, {})
        self.assertEqual([e["uid"] for e in capped], ["ok2"])
        self.assertEqual(selection._select(entries, 10, {"target_l1": 0.04})[0][0]["uid"], "ok2")
        narrowed = entry("ok2", False, 0.02, 0.03)
        narrowed["item"]["gate_accepted"] = True
        keep = [e for e in entries if e["uid"] == "ok1"] + [narrowed]
        self.assertEqual({e["uid"] for e in selection._select(keep, 10, {})[0]}, {"ok1", "ok2"})

    def test_inversion_residual_is_a_gate(self):
        cfg = load_config(CONFIGS / "stage2_36.json")
        limit = float(cfg["inversion"]["max_residual"])
        self.assertIsNone(_rejection({"l1": 0.01}, cfg, True, residual=limit * 0.5))
        self.assertIsNone(
            _rejection({"l1": 0.01}, cfg, True, residual=limit * 0.5, margin_ratio=0.5)
        )
        self.assertEqual(_rejection({"l1": 0.01}, cfg, True, residual=limit * 10), "target_inversion")
        self.assertEqual(_rejection({"l1": 0.01}, cfg, True, source=True, residual=limit * 10),
                         "source_inversion")

    def test_discrete_inverse_matches_the_deployment_map(self):
        import torch

        def velocity(z, t):
            return 0.3 + 0.5 * torch.sin(z) * (1.0 - float(t))

        grid = [0.0, 0.2, 0.36, 0.5, 0.7, 0.85, 1.0]
        image = torch.full((1, 1, 2, 2), 0.7)
        snapshots, diagnostics = discrete_inverse_states(
            image, grid, [0.36], velocity, iterations=8, damping=0.7
        )
        state = snapshots[time_key(0.36)]
        recovered = forward_native(state, 0.36, grid, velocity)
        self.assertLess(float((recovered - image).abs().max()), 1e-4)
        self.assertLess(diagnostics[time_key(0.36)]["residual"], 1e-3)


class Roles(unittest.TestCase):
    def _linear(self, root):
        Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "source.png")
        Image.new("RGB", (8, 8), (0, 255, 0)).save(root / "target.png")
        row = Records._linear(Records, root)
        (root / "linear.json").write_text(json.dumps({"records": [row]}))
        return root / "linear.json"

    def _pairs(self, root, cfg, rt, roles, source_kind="native_parent"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "states").mkdir(exist_ok=True)
        states = {}
        for time in cfg["state_times"]:
            path = root / "states" / f"{time_key(time)}.pt"
            torch.save(dict(
                time=time, state_a=torch.zeros(1, 3, 8, 8), state_b=torch.ones(1, 3, 8, 8),
                preview=torch.zeros(1, 3, 8, 8), base_prediction=None,
                valid=torch.ones(1, 1, 8, 8),
            ), path)
            states[time_key(time)] = dict(path=str(path.relative_to(root)), accepted=True)
        records = [
            dict(
                uid=f"r{index}", source_kind=source_kind, role=role, split="train",
                source_family=f"f{index}", prompt_family=f"p{index}",
                edit_instruction="Keep the current scene unchanged." if role == "R_keep" else "Make it green.",
                parent_condition="A red square.", tail_condition="A red square.", preserve="",
                condition_protocol=cfg["condition_protocol"], states=states,
            )
            for index, role in enumerate(roles)
        ]
        (root / "manifest.json").write_text(
            json.dumps(dict(kind=CACHE_KIND, protocol=protocol(rt, cfg), records=records))
        )
        return root

    def test_role_weights_draw_every_configured_role(self):
        cfg = load_config(CONFIGS / "stage2_36.json")
        rt = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            linear = LinearManifest(self._linear(root), rt, cfg)
            inverted = StatePairManifest(
                self._pairs(root / "inv", cfg, rt, ["I_edit", "I_edit"], "inverted_pair"),
                rt, cfg, "inverted_pair",
            )
            native = StatePairManifest(
                self._pairs(root / "nat", cfg, rt, ["R_edit", "R_keep", "R_edit"]),
                rt, cfg, "native_parent",
            )
            sampler = TrainingSampler(linear, inverted, native, cfg)
            counts = collections.Counter()
            for _ in range(4000):
                counts[sampler.draw()["role"]] += 1
            self.assertEqual(set(counts), {"L_edit", "I_edit", "R_edit", "R_keep"})
            for role, weight in cfg["role_weights"].items():
                self.assertAlmostEqual(counts[role] / 4000, weight, delta=0.04)
            self.assertEqual(sampler.keep_fraction, 0.0)
            self.assertEqual(
                set(sampler.input_domains()), {"linear", "inverted", "native_history"}
            )

    def test_missing_role_fails_instead_of_renormalizing(self):
        cfg = load_config(CONFIGS / "stage2_36.json")
        rt = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            linear = LinearManifest(self._linear(root), rt, cfg)
            native = StatePairManifest(
                self._pairs(root / "nat", cfg, rt, ["R_keep"]), rt, cfg, "native_parent"
            )
            with self.assertRaisesRegex(ValueError, "does not have"):
                TrainingSampler(linear, None, native, cfg)

    def test_role_and_recovery_settings_are_validated(self):
        cfg = load_config(CONFIGS / "stage2_36.json")
        for broken, message in (
            (dict(cfg, role_weights={"L_edit": 0.5}), "sum to one"),
            (dict(cfg, role_weights={"X_edit": 1.0}), "unknown role"),
            (dict(cfg, keep_fraction=0.1), "replaces keep_fraction"),
            (dict(cfg, recovery_step=0.3), "divide one"),
            (dict(cfg, recovery_prob=1.0), "recovery_prob"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                validate_config(broken)

    def test_keep_designation_is_stable_and_balanced(self):
        hits = sum(_keep_designation(str(index), 0.34) for index in range(4000))
        self.assertGreater(hits / 4000, 0.28)
        self.assertLess(hits / 4000, 0.40)
        self.assertEqual(_keep_designation("a", 0.34), _keep_designation("a", 0.34))
        self.assertFalse(_keep_designation("a", 0.0))
        self.assertTrue(_keep_designation("a", 1.0))


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
            "tail_condition": "A green square on a plain canvas.",
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
            self.assertEqual(sample["tail_condition"], "A green square on a plain canvas.")
            self.assertEqual(sample["tail_source"], "declared")
            keep = corpus.sample(0, 0.36, keep=True)
            self.assertTrue(torch.equal(keep["source"], keep["target"]))
            self.assertTrue(keep["keep"])

    def test_missing_full_tail_requires_the_explicit_legacy_flag(self):
        cfg = load_config(CONFIGS / "stage1.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._linear(root)
            row.pop("tail_condition")
            (root / "manifest.json").write_text(json.dumps({"records": [row]}))
            with self.assertRaisesRegex(ValueError, "full tail_condition"):
                LinearManifest(root / "manifest.json", Runtime(), cfg)
            row["allow_derived_tail"] = True
            (root / "manifest.json").write_text(json.dumps({"records": [row]}))
            corpus = LinearManifest(root / "manifest.json", Runtime(), cfg)
            self.assertEqual(corpus.sample(0, 0.36)["tail_source"], "derived_parent_plus_edit")

    def test_target_binding_requires_approval_parent_and_instruction(self):
        target = dict(
            semantic_approved=True,
            edit_instruction="Put the cat behind the dog.",
            parent_branch_id="17:t2i",
            seed=17,
            target_image="target.png",
        )
        self.assertIsNone(
            _target_binding_error(target, 17, "17:t2i", "Put the cat behind the dog.")
        )
        self.assertEqual(
            _target_binding_error(dict(target, semantic_approved=None), 17, "17:t2i", "Put the cat behind the dog."),
            "target_unapproved",
        )
        self.assertEqual(
            _target_binding_error(dict(target, edit_instruction="Other."), 17, "17:t2i", "Put the cat behind the dog."),
            "target_instruction_mismatch",
        )
        self.assertEqual(
            _target_binding_error(dict(target, parent_branch_id="18:t2i"), 17, "17:t2i", "Put the cat behind the dog."),
            "target_parent_mismatch",
        )
        self.assertEqual(
            _target_binding_error(dict(target, seed=18), 17, "17:t2i", "Put the cat behind the dog."),
            "target_seed_mismatch",
        )

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

    def test_cache_acceptance_uses_absolute_caps_without_a_margin(self):
        cfg = load_config(CONFIGS / "stage2_36.json")
        self.assertIsNone(_rejection({"l1": 0.01}, cfg, True, source=True))
        self.assertIsNone(_rejection({"l1": 0.01}, cfg, True))
        self.assertEqual(_rejection({"l1": 0.2}, cfg, True, source=True), "source_l1")
        self.assertEqual(_rejection({"l1": 0.2}, cfg, True), "target_l1")
        self.assertEqual(_rejection({"l1": 0.01, "lpips": 0.5}, cfg, True), "target_lpips")
        self.assertEqual(_rejection({"l1": 0.01}, cfg, False), "target_semantics")
        self.assertEqual(_rejection({"l1": float("inf")}, cfg, True, source=True), "source_l1")

    def test_pair_margin_replaces_the_absolute_caps_when_enabled(self):
        base = load_config(CONFIGS / "stage2_36.json")
        cfg = dict(base, inversion=dict(base["inversion"], margin_fraction=0.25))
        self.assertIsNone(_rejection({"l1": 0.4}, cfg, True, margin_ratio=0.35))
        self.assertEqual(_rejection({"l1": 0.01}, cfg, True, margin_ratio=0.15), "target_margin")
        self.assertEqual(
            _rejection({"l1": 0.01}, cfg, False, margin_ratio=0.35), "target_semantics"
        )
        self.assertEqual(
            _rejection({"l1": 0.01}, cfg, True, margin_ratio=None), "target_margin_missing"
        )
        # With the criterion switched off, the absolute caps decide again.
        self.assertEqual(_rejection({"l1": 0.4}, base, True, margin_ratio=0.35), "target_l1")

    def test_equal_aspect_pairs_share_one_scene_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "source.png")
            Image.new("RGB", (16, 16), (0, 255, 0)).save(root / "target.png")
            source, target, valid, rect = load_pair(
                {"source_image": str(root / "source.png"), "target_image": str(root / "target.png")},
                8, 8, "cpu",
            )
            self.assertEqual(rect, (0, 0, 8, 8))
            self.assertEqual(float(valid.sum()), 64.0)
            # Both files fill the canvas: neither subject is shrunk into the centre.
            self.assertGreater(float(source[:, 0].mean()), 0.9)
            self.assertLess(float(target[:, 0].mean()), -0.9)  # green: no red channel

    def test_pairs_with_different_extents_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "a.png")
            Image.new("RGB", (4, 8)).save(root / "b.png")
            row = {"source_image": str(root / "a.png"), "target_image": str(root / "b.png")}
            with self.assertRaisesRegex(ValueError, "scene extents"):
                load_pair(row, 8, 8, "cpu")
            self.assertIsNotNone(pair_geometry_error(row, (8, 8), (4, 8)))
            self.assertIsNone(pair_geometry_error(row, (8, 8), (16, 16)))

    def test_declared_geometry_crops_share_one_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), (255, 0, 0)).save(root / "source.png")
            Image.new("RGB", (16, 8), (0, 255, 0)).save(root / "target.png")
            row = {
                "source_image": str(root / "source.png"),
                "target_image": str(root / "target.png"),
                "geometry": {"source": "full", "target": [0, 0, 8, 8]},
            }
            source, target, valid, rect = load_pair(row, 8, 8, "cpu")
            self.assertEqual(rect, (0, 0, 8, 8))
            self.assertEqual(source.shape, target.shape)
            self.assertEqual(float(valid.sum()), 64.0)
            self.assertLess(float(target[:, 0].mean()), -0.9)  # green: no red channel
            broken = dict(row, geometry={"source": "full", "target": [0, 0, 16, 4]})
            with self.assertRaisesRegex(ValueError, "aspect ratio"):
                load_pair(broken, 8, 8, "cpu")

    def test_keep_canvas_honours_a_declared_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (16, 8), (0, 255, 0))
            image.paste(Image.new("RGB", (8, 8), (255, 0, 0)), (8, 0))
            image.save(root / "source.png")
            canvas, valid, rect = load_canvas(
                str(root / "source.png"), 8, 8, "cpu", crop=[0, 0, 8, 8]
            )
            self.assertEqual(rect, (0, 0, 8, 8))
            self.assertEqual(float(valid.sum()), 64.0)
            self.assertLess(float(canvas[:, 0].mean()), -0.9)

    def test_target_refresh_needs_a_bound_target(self):
        marker = dict(
            uid="a",
            diagnosis_status="needs_edit",
            parent_branch_id="17:t2i",
            edit_instruction="Put the cat behind the dog.",
        )
        target = dict(
            semantic_approved=True,
            edit_instruction="Put the cat behind the dog.",
            parent_branch_id="17:t2i",
            seed=17,
            target_image="target.png",
        )
        self.assertTrue(_refresh_candidate(marker, {"a": target}))
        self.assertFalse(_refresh_candidate(marker, {}))
        self.assertFalse(_refresh_candidate(dict(marker, diagnosis_status="passed"), {"a": target}))
        self.assertFalse(
            _refresh_candidate(marker, {"a": dict(target, edit_instruction="Something else.")})
        )
        self.assertFalse(_refresh_candidate(marker, {"a": dict(target, seed=18)}))


def test_reverse_instruction_never_uses_a_keep_wording():
    """A flipped pair changes the picture, so a keep instruction would teach
    that subjects may vanish unasked."""
    from qwen_latent_cot.u15_target_correction.explicit import reverse_instruction

    forward_count = "Add 3 laptops so that it contains exactly 5 instead of the current 2"
    assert reverse_instruction("T8", forward_count) == (
        "Remove 3, keeping the same scene otherwise unchanged."
    )
    assert reverse_instruction("T8", "Remove 2 chairs so that exactly 3 remain") == (
        "Add 2 back, keeping the rest of the scene unchanged."
    )
    for task in ("T9", "T3", "T1"):
        text = reverse_instruction(task, "anything")
        assert text
        assert "keep the current scene unchanged" not in text.lower()
    # Tasks whose reverse is near-identity or composite must be dropped, not guessed.
    assert reverse_instruction("T6", "Restyle the whole image as pop art") is None
    assert reverse_instruction("T11", "Perform these edits in order") is None
    assert reverse_instruction("T8", "make it nicer") is None


def test_repair_role_is_terminal_but_linear_pairs_are_not():
    from qwen_latent_cot.u15_target_correction import (
        ROLE_L_EDIT, ROLE_L_REVERSE, ROLE_R_REPAIR, TERMINAL_ROLES,
    )

    assert ROLE_R_REPAIR in TERMINAL_ROLES
    assert ROLE_L_EDIT not in TERMINAL_ROLES
    assert ROLE_L_REVERSE not in TERMINAL_ROLES
