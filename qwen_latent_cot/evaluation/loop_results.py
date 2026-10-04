"""Paired result verification; semantic scores never come from velocity logs."""

import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return statistics.fmean(values) if values else None


def prompt_gm(atoms):
    return (
        0.0 if min(atoms) == 0 else math.exp(mean([math.log(value) for value in atoms]))
    )


def paired_mean_ci(values, *, seed=0, resamples=1000):
    """Bootstrap over paired prompts, not over questions within one image."""
    if not values:
        return {"mean": None, "ci95": None, "pairs": 0}
    import numpy as np

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [
        float(values[rng.integers(len(values), size=len(values))].mean())
        for _ in range(resamples)
    ]
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "pairs": len(values),
    }


def pair_key(row):
    return (row["index"], row["seed"])


def merge_manifests(specifications):
    """Select arm families from separate checkpoints without mixing baselines."""
    merged, identities, comparison = {"images": [], "arms": {}}, {}, None
    if not specifications or any(item is None for item in specifications):
        raise ValueError("at least one generation manifest is required")
    for specification in specifications:
        if isinstance(specification, str):
            specification = {"path": specification}
        path = Path(specification["path"])
        manifest = json.loads(path.read_text())
        args = manifest.get("arguments", {})
        current = {
            key: args.get(key)
            for key in [
                "model_path",
                "seed",
                "num_timesteps",
                "timestep_shift",
                "cfg_text_scale",
                "start_layer",
                "end_layer",
                "batch_size",
                "image_size",
            ]
        }
        actual_loop = manifest.get("allocated_loop_config", {})
        current["start_layer"] = actual_loop.get(
            "loop_start_layer", current["start_layer"]
        )
        current["end_layer"] = actual_loop.get("loop_end_layer", current["end_layer"])
        if comparison is not None and current != comparison:
            raise ValueError("generation settings differ across checkpoint manifests")
        comparison = current
        selected = specification.get("modes")
        for row in manifest["images"]:
            if selected is not None and not any(
                row["arm"].startswith(mode + "_R") for mode in selected
            ):
                continue
            identity = (row["arm"], *pair_key(row))
            if identity in identities:
                old = identities[identity]
                if old["prompt"] != row["prompt"] or file_sha256(
                    old["path"]
                ) != file_sha256(row["path"]):
                    raise ValueError(
                        "duplicate selected arms differ; explicitly select one checkpoint per mode"
                    )
                continue
            identities[identity] = row
            merged["images"].append(row)
        merged["arms"].update(manifest.get("arms", {}))
    return merged


def validate_required_arms(manifest, depths):
    arms = {row["arm"] for row in manifest["images"]}
    if not any(arm.startswith("base_R0_") for arm in arms):
        raise ValueError("result validation requires Base R0")
    for mode in ["legacy_memory_only", "gen_only", "gen_memory_anchored"]:
        for depth in depths:
            if depth and not any(arm.startswith(f"{mode}_R{depth}_") for arm in arms):
                raise ValueError(f"result matrix missing {mode} at R={depth}")


def score_manifest(
    manifest, benchmark_rows, *, dataset, semantic_scorer=None, quality_judge=None
):
    """Scorers receive actual generated files and original benchmark questions.

    semantic_scorer accepts a complete arm batch and returns atom lists. Invalid
    files fail semantics explicitly. Missing scorers remain null, never zero.
    """
    images = manifest["images"]
    if not images:
        raise ValueError("empty generation manifest")
    grouped = defaultdict(list)
    for row in images:
        grouped[row["arm"]].append(row)
    output = []
    for arm, rows in grouped.items():
        rows.sort(key=lambda row: (row["index"], row["seed"]))
        seen = set()
        valid_rows, valid_benchmarks, prepared = [], [], []
        for row in rows:
            if pair_key(row) in seen:
                raise ValueError("duplicate index/seed within an evaluation arm")
            seen.add(pair_key(row))
            if type(row["index"]) is not int or not 0 <= row["index"] < len(
                benchmark_rows
            ):
                raise ValueError("manifest benchmark index is out of range")
            benchmark = benchmark_rows[row["index"]]
            if benchmark["prompt"] != row["prompt"]:
                raise ValueError("manifest prompt does not match benchmark index")
            decoded_shape = None
            try:
                with Image.open(row["path"]) as image:
                    decoded_shape = [image.height, image.width]
                    image.verify()
                valid, error = True, None
            except (OSError, ValueError) as exc:
                valid, error = False, str(exc)
            value = {
                "dataset": dataset,
                **row,
                "valid_file": valid,
                "decode_error": error,
                "decoded_shape": decoded_shape,
                "image_sha256": file_sha256(row["path"])
                if Path(row["path"]).exists()
                else None,
                "semantic_atoms": None,
                "quality_proxy": None,
                "judge_invalid": None,
            }
            if not valid and semantic_scorer is not None:
                questions = benchmark.get(
                    "vqa_list", benchmark.get("yn_question_list", [])
                )
                if not questions:
                    raise ValueError("semantic benchmark requires questions")
                value["semantic_atoms"] = [0.0] * len(questions)
            prepared.append(value)
            if valid:
                valid_rows.append(row)
                valid_benchmarks.append(benchmark)
        if {row["index"] for row in rows} != set(range(len(benchmark_rows))):
            raise ValueError("each arm must cover the full selected benchmark")
        if semantic_scorer is not None and valid_rows:
            scores = semantic_scorer(valid_rows, valid_benchmarks)
            if len(scores) != len(valid_rows):
                raise ValueError("semantic scorer returned an incomplete arm")
            score_iter = iter(scores)
            for value, benchmark in zip(
                prepared, [benchmark_rows[row["index"]] for row in rows]
            ):
                if value["valid_file"]:
                    atoms = next(score_iter)
                    question_count = len(
                        benchmark.get("vqa_list", benchmark.get("yn_question_list", []))
                    )
                    if (
                        len(atoms) != question_count
                        or not atoms
                        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in atoms)
                    ):
                        raise ValueError("semantic scorer returned invalid atom scores")
                    value["semantic_atoms"] = [float(v) for v in atoms]
        if quality_judge is not None:
            for value in prepared:
                quality = (
                    quality_judge(value["path"])
                    if value["valid_file"]
                    else {"quality_proxy": 0.0, "invalid": True}
                )
                if (
                    not math.isfinite(quality["quality_proxy"])
                    or not 0 <= quality["quality_proxy"] <= 1
                    or type(quality["invalid"]) is not bool
                ):
                    raise ValueError("quality judge returned invalid scores")
                value["quality_proxy"], value["judge_invalid"] = (
                    quality["quality_proxy"],
                    quality["invalid"],
                )
        output.extend(prepared)
    return output


def constraint_repair_damage(pairs, threshold):
    if any(
        a["semantic_atoms"] is None or b["semantic_atoms"] is None for a, b in pairs
    ):
        return None
    decisions = []
    for a, b in pairs:
        if len(a["semantic_atoms"]) != len(b["semantic_atoms"]):
            raise ValueError("paired semantic constraints differ")
        decisions.extend(
            (x >= threshold, y >= threshold)
            for x, y in zip(a["semantic_atoms"], b["semantic_atoms"])
        )
    repair = sum(not a and b for a, b in decisions)
    damage = sum(a and not b for a, b in decisions)
    failed, passed = sum(not a for a, _ in decisions), sum(a for a, _ in decisions)
    return {
        "repair_count": repair,
        "damage_count": damage,
        "reference_fail_count": failed,
        "reference_pass_count": passed,
        "repair_rate_given_reference_fail": repair / failed if failed else None,
        "damage_rate_given_reference_pass": damage / passed if passed else None,
        "repair_fraction_all_constraints": repair / len(decisions),
        "damage_fraction_all_constraints": damage / len(decisions),
        "paired_constraints": len(decisions),
        "paired_images": len(pairs),
        "pass_rule": "individual_atom_at_least_threshold",
        "threshold": threshold,
    }


def summarize_results(scored_rows, *, threshold=0.5):
    if not scored_rows:
        raise ValueError("no scored rows to summarize")
    if not 0 <= threshold <= 1:
        raise ValueError("semantic pass threshold must lie in [0,1]")
    groups = defaultdict(dict)
    for row in scored_rows:
        key = pair_key(row)
        group = groups[(row["dataset"], row["arm"])]
        if key in group:
            raise ValueError("duplicate result identity")
        group[key] = row
    summaries = []
    for (dataset, arm), rows in groups.items():
        base_arms = [
            key for key in groups if key[0] == dataset and key[1].startswith("base_R0_")
        ]
        if len(base_arms) != 1:
            raise ValueError("each dataset needs exactly one Base R0 arm")
        base = groups[base_arms[0]]
        if set(rows) != set(base):
            raise ValueError("all arms must have the same paired index/seed coverage")
        pairs = [(base[key], rows[key]) for key in sorted(rows)]
        if any(a["prompt"] != b["prompt"] for a, b in pairs):
            raise ValueError("paired prompts differ")
        if any(
            a["valid_file"]
            and b["valid_file"]
            and a.get("decoded_shape") != b.get("decoded_shape")
            for a, b in pairs
        ):
            raise ValueError("paired image dimensions differ")
        if any(
            a.get("initial_noise_sha256") != b.get("initial_noise_sha256")
            for a, b in pairs
        ):
            raise ValueError("paired initial noise differs")
        # Track constraint repair/damage both from the initial Base and from
        # the preceding depth in the same mode, K and memory intervention.
        match = re.fullmatch(r"(.+)_R(\d+)_(.+)", arm)
        previous_arm = None
        if match and int(match[2]) > 0:
            previous_arm = (
                base_arms[0][1]
                if int(match[2]) == 1
                else f"{match[1]}_R{int(match[2]) - 1}_{match[3]}"
            )
            if (dataset, previous_arm) not in groups:
                previous_arm = None
        if previous_arm and set(groups[(dataset, previous_arm)]) != set(rows):
            raise ValueError("previous depth has different paired coverage")
        previous_pairs = (
            [(groups[(dataset, previous_arm)][key], rows[key]) for key in sorted(rows)]
            if previous_arm
            else []
        )
        values = list(rows.values())
        semantic_ready = all(row["semantic_atoms"] is not None for row in values)
        semantic_paired = semantic_ready and all(
            a["semantic_atoms"] is not None for a, b in pairs
        )
        qualities = [row["quality_proxy"] for row in values]
        quality_ready = all(value is not None for value in qualities)
        quality_deltas = [
            b["quality_proxy"] - a["quality_proxy"]
            for a, b in pairs
            if a["quality_proxy"] is not None and b["quality_proxy"] is not None
        ]
        repair_damage = None
        semantic_delta = {"mean": None, "ci95": None, "pairs": 0}
        if semantic_paired:
            decisions = [
                (
                    all(v >= threshold for v in a["semantic_atoms"]),
                    all(v >= threshold for v in b["semantic_atoms"]),
                )
                for a, b in pairs
            ]
            repair = sum(not a and b for a, b in decisions)
            damage = sum(a and not b for a, b in decisions)
            base_fail = sum(not a for a, b in decisions)
            base_pass = sum(a for a, b in decisions)
            repair_damage = {
                "repair_count": repair,
                "damage_count": damage,
                "base_fail_count": base_fail,
                "base_pass_count": base_pass,
                "repair_rate_given_base_fail": repair / base_fail
                if base_fail
                else None,
                "damage_rate_given_base_pass": damage / base_pass
                if base_pass
                else None,
                "repair_fraction_all_pairs": repair / len(pairs),
                "damage_fraction_all_pairs": damage / len(pairs),
                "paired_count": len(pairs),
                "pass_rule": "every_atom_at_least_threshold",
                "threshold": threshold,
            }
            semantic_delta = paired_mean_ci(
                [
                    prompt_gm(b["semantic_atoms"]) - prompt_gm(a["semantic_atoms"])
                    for a, b in pairs
                ]
            )
        invalid_ready = all(row["judge_invalid"] is not None for row in values)
        summaries.append(
            {
                "dataset": dataset,
                "arm": arm,
                "n": len(values),
                "depth_status": sorted(
                    set(row.get("depth_status", "unknown") for row in values)
                ),
                "semantic_status": "scored" if semantic_ready else "unscored",
                "semantic_AM": mean([mean(row["semantic_atoms"]) for row in values])
                if semantic_ready
                else None,
                "semantic_GM": mean(
                    [prompt_gm(row["semantic_atoms"]) for row in values]
                )
                if semantic_ready
                else None,
                "semantic_GM_delta_vs_base": semantic_delta,
                "quality_status": "proxy_scored" if quality_ready else "unscored",
                "quality_proxy_mean": mean(qualities) if quality_ready else None,
                "quality_delta_vs_base": paired_mean_ci(quality_deltas)
                if len(quality_deltas) == len(pairs)
                else {"mean": None, "ci95": None, "pairs": len(quality_deltas)},
                "decode_invalid_rate": sum(not row["valid_file"] for row in values)
                / len(values),
                "invalid_rate": sum(
                    not row["valid_file"] or row["judge_invalid"] for row in values
                )
                / len(values)
                if invalid_ready
                else None,
                "repair_damage": repair_damage,
                "constraint_repair_damage_vs_base": constraint_repair_damage(
                    pairs, threshold
                ),
                "previous_depth_arm": previous_arm,
                "constraint_repair_damage_vs_previous": constraint_repair_damage(
                    previous_pairs, threshold
                )
                if previous_pairs
                else None,
            }
        )
    return summaries


class OfficialGenEval2:
    def __init__(self, root, output_dir, *, python=sys.executable):
        self.script = Path(root).resolve() / "evaluation.py"
        if not self.script.is_file():
            raise FileNotFoundError(
                f"official GenEval2 evaluator missing: {self.script}"
            )
        self.output_dir, self.python = Path(output_dir).resolve(), python
        self.provenance = {
            "kind": "official_geneval2_soft_tifa",
            "method": "soft_tifa_gm",
            "script": str(self.script),
            "script_sha256": file_sha256(self.script),
        }

    def __call__(self, rows, benchmark_rows):
        # Official prompt->image maps hold one image per prompt. Repeated seeds
        # need separate invocations, while unique prompts share one model load.
        if len(set(row["prompt"] for row in rows)) != len(rows):
            occurrences, batches, result = {}, {}, [None] * len(rows)
            for i, (row, benchmark) in enumerate(zip(rows, benchmark_rows)):
                repeat = occurrences.get(row["prompt"], 0)
                occurrences[row["prompt"]] = repeat + 1
                batches.setdefault(repeat, []).append((i, row, benchmark))
            for repeat, batch in batches.items():
                adjusted = [
                    {**row, "arm": row["arm"] + f"/repeat_{repeat:03d}"}
                    for i, row, benchmark in batch
                ]
                scores = self(adjusted, [benchmark for i, row, benchmark in batch])
                for (i, row, benchmark), atoms in zip(batch, scores):
                    result[i] = atoms
            return result
        directory = self.output_dir / rows[0]["arm"]
        directory.mkdir(parents=True, exist_ok=True)
        if len(set(row["prompt"] for row in rows)) != len(rows):
            raise ValueError(
                "official GenEval2 image map requires unique prompts per arm"
            )
        benchmark, images, scores = (
            directory / "benchmark.jsonl",
            directory / "images.json",
            directory / "score_lists.json",
        )
        benchmark.write_text(
            "\n".join(json.dumps(row) for row in benchmark_rows) + "\n"
        )
        images.write_text(json.dumps({row["prompt"]: row["path"] for row in rows}))
        command = [
            self.python,
            str(self.script),
            "--benchmark_data",
            str(benchmark),
            "--image_filepath_data",
            str(images),
            "--method",
            "soft_tifa_gm",
            "--output_file",
            str(scores),
        ]
        (directory / "command.json").write_text(json.dumps(command))
        # Stream scorer progress; failures never masquerade as zero scores.
        subprocess.run(command, cwd=self.script.parent, check=True)
        return json.loads(scores.read_text())


class LocalVLMJudge:
    """Offline Transformers VLM judge for TIIF yes/no questions and quality.

    TIIF uses a deterministic question protocol with original yes/no ground
    truth. This is a local judge variant, not the official randomized API run.
    Quality is a VLM proxy; it never replaces human pairwise review.
    """

    def __init__(self, model_path, *, device="cpu"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"local judge weights missing: {path}")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                path,
                local_files_only=True,
                dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            )
            .to(device)
            .eval()
        )
        self.provenance = {
            "kind": "deterministic_local_vlm",
            "model_path": str(path),
            "device": device,
            "config_sha256": file_sha256(path / "config.json"),
            "quality_scale": "1_to_5_normalized_to_0_to_1",
        }

    def answer(self, image_path, question):
        import torch

        with Image.open(image_path) as image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False
                )
            return self.processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0].strip()

    def tiif(self, rows, benchmarks):
        output = []
        for row, benchmark in zip(rows, benchmarks):
            questions, answers = (
                benchmark["yn_question_list"],
                benchmark["yn_answer_list"],
            )
            if not questions or len(questions) != len(answers):
                raise ValueError(
                    "TIIF spatial records require matched questions/answers"
                )
            scores = []
            for question, expected in zip(questions, answers):
                predicted = (
                    self.answer(
                        row["path"],
                        f"Inspect this image. {question}\nAnswer only yes or no.",
                    )
                    .lower()
                    .rstrip(". ")
                )
                expected = str(expected).lower()
                if predicted not in {"yes", "no"} or expected not in {"yes", "no"}:
                    raise ValueError(
                        "TIIF judge/ground truth must be yes or no; no silent fallback"
                    )
                scores.append(float(predicted == expected))
            output.append(scores)
        return output

    def quality(self, image_path):
        question = (
            "Assess visual coherence and visible artifacts, independently of instruction following. "
            "Quality: 1=severely broken, 2=poor, 3=usable, 4=good, 5=excellent. "
            "Invalid means an unusable or severely corrupted image. Return only JSON with integer quality and boolean invalid."
        )
        answer = self.answer(image_path, question)
        # Some models wrap otherwise valid JSON in a fenced block.
        payload = json.loads(
            answer.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        if (
            type(payload["quality"]) is not int
            or not 1 <= payload["quality"] <= 5
            or type(payload["invalid"]) is not bool
        ):
            raise ValueError("quality judge must emit quality 1..5 and boolean invalid")
        return {
            "quality_proxy": (payload["quality"] - 1) / 4,
            "invalid": payload["invalid"],
        }


def summarize_human_pairwise(path, scored_rows):
    import csv

    indexed = {(row["dataset"], row["arm"], *pair_key(row)): row for row in scored_rows}
    baselines = {
        (row["dataset"], *pair_key(row)): row
        for row in scored_rows
        if row["arm"].startswith("base_R0_")
    }
    collected, seen = defaultdict(list), set()
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if not row["preference"]:
                continue
            key = (row["dataset"], row["arm"], int(row["index"]), int(row["seed"]))
            if key not in indexed or key in seen:
                raise ValueError("human review has unknown or duplicated pair")
            seen.add(key)
            base = baselines[(key[0], key[2], key[3])]
            if (
                row["loop_sha256"] != indexed[key]["image_sha256"]
                or row["base_sha256"] != base["image_sha256"]
            ):
                raise ValueError("human pairwise review refers to changed images")
            if row["preference"] not in {"base", "loop", "tie"}:
                raise ValueError("human preference must be base, loop, or tie")
            collected[(key[0], key[1])].append(row["preference"])
    return {
        key: {
            "reviewed_pairs": len(v),
            "total_pairs": sum(
                row["dataset"] == key[0] and row["arm"] == key[1] for row in scored_rows
            ),
            "loop_wins": v.count("loop"),
            "base_wins": v.count("base"),
            "ties": v.count("tie"),
            "loop_win_fraction": v.count("loop") / len(v),
            "paired_preference": paired_mean_ci(
                [1 if item == "loop" else -1 if item == "base" else 0 for item in v]
            ),
        }
        for key, v in collected.items()
    }


def write_reports(output_dir, rows, summaries, provenance):
    import csv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"provenance": provenance, "summaries": summaries}, indent=2)
    )
    lines = [
        "# Paired T2I result validation",
        "",
        "Quality is a judge proxy. Unscored metrics remain null. Unseen depths are explicitly labeled.",
        "",
        "| Dataset | Arm | Depth status | Semantic GM | Quality Δ vs Base | Invalid | Constraint Repair | Constraint Damage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "unscored" if value is None else f"{value:.4f}"

    for item in summaries:
        change = item["constraint_repair_damage_vs_base"] or {}
        lines.append(
            f"| {item['dataset']} | {item['arm']} | {','.join(item['depth_status'])} | {number(item['semantic_GM'])} | {number(item['quality_delta_vs_base']['mean'])} | {number(item['invalid_rate'])} | {number(change.get('repair_rate_given_reference_fail'))} | {number(change.get('damage_rate_given_reference_pass'))} |"
        )
    identity = [
        (row["dataset"], row["arm"], *pair_key(row), row["image_sha256"])
        for row in rows
    ]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    template = out / f"human_pairwise_{digest}.csv"
    lines.extend(
        [
            "",
            "Human pairwise review: "
            + (
                "imported; see summary.json counts"
                if any(item.get("human_pairwise") for item in summaries)
                else f"pending; use {template.name}"
            ),
        ]
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    # Preserve annotations on reruns; changed images receive a new template.
    if template.exists():
        return
    with template.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "arm",
                "index",
                "seed",
                "base_image",
                "loop_image",
                "base_sha256",
                "loop_sha256",
                "preference",
            ],
        )
        writer.writeheader()
        base = {
            (row["dataset"], pair_key(row)): row
            for row in rows
            if row["arm"].startswith("base_R0_")
        }
        for row in rows:
            if not row["arm"].startswith("base_R0_"):
                writer.writerow(
                    {
                        "dataset": row["dataset"],
                        "arm": row["arm"],
                        "index": row["index"],
                        "seed": row["seed"],
                        "base_image": base[(row["dataset"], pair_key(row))]["path"],
                        "loop_image": row["path"],
                        "base_sha256": base[(row["dataset"], pair_key(row))][
                            "image_sha256"
                        ],
                        "loop_sha256": row["image_sha256"],
                        "preference": "",
                    }
                )
