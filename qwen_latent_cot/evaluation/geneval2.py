"""GenEval2 score aggregation and image-map utilities.

The official GenEval2 evaluator writes one list of per-atom scores per prompt.
This module keeps the heavy VLM scoring separate from lightweight aggregation:

1. build ``{prompt: image_path}`` maps for ``refs/GenEval2/evaluation.py``;
2. aggregate the resulting score lists into overall AM/GM, per-skill AM, and
   per-atomicity GM tables.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SKILL_ORDER = ("object", "count", "attribute", "position", "verb")


@dataclass(frozen=True)
class GenEval2Prompt:
    prompt: str
    atom_count: int
    skills: tuple[str, ...]


@dataclass(frozen=True)
class GenEval2Benchmark:
    prompts: tuple[GenEval2Prompt, ...]

    @property
    def prompt_count(self) -> int:
        return len(self.prompts)

    @property
    def atom_count(self) -> int:
        return sum(len(item.skills) for item in self.prompts)

    @property
    def skill_counts(self) -> Dict[str, int]:
        counts = {skill: 0 for skill in SKILL_ORDER}
        for item in self.prompts:
            for skill in item.skills:
                counts[skill] = counts.get(skill, 0) + 1
        return counts


def load_benchmark(path: str | Path) -> GenEval2Benchmark:
    prompts: list[GenEval2Prompt] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = str(record["prompt"])
            atom_count = int(record["atom_count"])
            skills = tuple(str(skill) for skill in record["skills"])
            if not skills:
                raise ValueError(f"line {line_no}: empty skills list")
            prompts.append(GenEval2Prompt(prompt=prompt, atom_count=atom_count, skills=skills))
    if not prompts:
        raise ValueError(f"empty GenEval2 benchmark: {path}")
    return GenEval2Benchmark(tuple(prompts))


def _as_score_lists(payload: Any, benchmark: GenEval2Benchmark) -> list[list[float]]:
    """Accept official score-list JSON and a few report-friendly wrappers."""
    if isinstance(payload, dict):
        if "score_lists" in payload:
            payload = payload["score_lists"]
        elif "scores" in payload:
            payload = payload["scores"]
        else:
            prompt_map = payload
            if all(item.prompt in prompt_map for item in benchmark.prompts):
                return [
                    [float(value) for value in prompt_map[item.prompt]]
                    for item in benchmark.prompts
                ]
            raise ValueError(
                "score JSON dict must contain 'score_lists', 'scores', or prompt keys"
            )
    if not isinstance(payload, list):
        raise ValueError("score JSON must be a list or a supported dict wrapper")
    return [[float(value) for value in row] for row in payload]


def load_score_lists(path: str | Path, benchmark: GenEval2Benchmark) -> list[list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    score_lists = _as_score_lists(payload, benchmark)
    validate_score_lists(score_lists, benchmark, source=str(path))
    return score_lists


def validate_score_lists(
    score_lists: Sequence[Sequence[float]],
    benchmark: GenEval2Benchmark,
    *,
    source: str = "score_lists",
) -> None:
    if len(score_lists) != benchmark.prompt_count:
        raise ValueError(
            f"{source}: expected {benchmark.prompt_count} prompts, got {len(score_lists)}"
        )
    for idx, (scores, item) in enumerate(zip(score_lists, benchmark.prompts)):
        if len(scores) != len(item.skills):
            raise ValueError(
                f"{source}: prompt {idx} has {len(scores)} scores but "
                f"{len(item.skills)} skills"
            )
        for score in scores:
            if not math.isfinite(float(score)):
                raise ValueError(f"{source}: non-finite score at prompt {idx}: {score}")
            if float(score) < 0.0 or float(score) > 1.0:
                raise ValueError(
                    f"{source}: score values must be probabilities in [0, 1], got {score}"
                )


def arithmetic_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot gmean an empty sequence")
    vals = [float(value) for value in values]
    if any(value < 0.0 for value in vals):
        raise ValueError("geometric mean is undefined for negative values")
    if any(value == 0.0 for value in vals):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in vals) / len(vals)))


def summarize_score_lists(
    benchmark: GenEval2Benchmark,
    score_lists: Sequence[Sequence[float]],
    *,
    name: str,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    validate_score_lists(score_lists, benchmark, source=source or name)

    per_prompt_am = [arithmetic_mean(row) for row in score_lists]
    per_prompt_gm = [geometric_mean(row) for row in score_lists]
    all_atom_scores = [score for row in score_lists for score in row]

    skill_sum = {skill: 0.0 for skill in SKILL_ORDER}
    skill_count = {skill: 0 for skill in SKILL_ORDER}
    for scores, item in zip(score_lists, benchmark.prompts):
        for score, skill in zip(scores, item.skills):
            if skill not in skill_sum:
                skill_sum[skill] = 0.0
                skill_count[skill] = 0
            skill_sum[skill] += float(score)
            skill_count[skill] += 1
    skills = {
        skill: {
            "count": skill_count.get(skill, 0),
            "soft_tifa_am": _percent(skill_sum[skill] / skill_count[skill])
            if skill_count.get(skill, 0)
            else float("nan"),
        }
        for skill in sorted(skill_sum, key=lambda s: SKILL_ORDER.index(s) if s in SKILL_ORDER else 999)
    }

    atomicity: dict[str, dict[str, float | int]] = {}
    for scores, item, gm in zip(score_lists, benchmark.prompts, per_prompt_gm):
        key = str(item.atom_count)
        if key not in atomicity:
            atomicity[key] = {"count": 0, "sum_gm": 0.0}
        atomicity[key]["count"] = int(atomicity[key]["count"]) + 1
        atomicity[key]["sum_gm"] = float(atomicity[key]["sum_gm"]) + gm
    for key, value in atomicity.items():
        count = int(value["count"])
        value["soft_tifa_gm"] = _percent(float(value.pop("sum_gm")) / count)

    return {
        "name": name,
        "source": source,
        "num_prompts": benchmark.prompt_count,
        "num_atoms": benchmark.atom_count,
        "overall": {
            "soft_tifa_am": _percent(arithmetic_mean(per_prompt_am)),
            "soft_tifa_gm": _percent(arithmetic_mean(per_prompt_gm)),
            "atom_weighted_am": _percent(arithmetic_mean(all_atom_scores)),
        },
        "skills": skills,
        "atomicity": dict(sorted(atomicity.items(), key=lambda kv: int(kv[0]))),
    }


def _percent(value: float) -> float:
    return float(value * 100.0)


def compare_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    baseline_name: Optional[str] = None,
) -> Dict[str, Any]:
    result = {"runs": list(summaries), "baseline": baseline_name, "deltas": {}}
    if not baseline_name:
        return result
    baseline = next((item for item in summaries if item["name"] == baseline_name), None)
    if baseline is None:
        raise ValueError(f"baseline run not found: {baseline_name}")
    base_overall = baseline["overall"]
    base_skills = baseline["skills"]
    for item in summaries:
        name = item["name"]
        result["deltas"][name] = {
            "overall": {
                key: float(item["overall"][key]) - float(base_overall[key])
                for key in ("soft_tifa_am", "soft_tifa_gm", "atom_weighted_am")
            },
            "skills": {
                skill: float(item["skills"][skill]["soft_tifa_am"])
                - float(base_skills[skill]["soft_tifa_am"])
                for skill in item["skills"]
                if skill in base_skills
            },
        }
    return result


def write_summary_csv(summary: Mapping[str, Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in summary["runs"]:
        row = {
            "run": item["name"],
            "soft_tifa_am": item["overall"]["soft_tifa_am"],
            "soft_tifa_gm": item["overall"]["soft_tifa_gm"],
            "atom_weighted_am": item["overall"]["atom_weighted_am"],
        }
        delta = summary.get("deltas", {}).get(item["name"], {}).get("overall", {})
        row.update({f"delta_{key}": value for key, value in delta.items()})
        for skill in SKILL_ORDER:
            if skill in item["skills"]:
                row[f"{skill}_am"] = item["skills"][skill]["soft_tifa_am"]
                skill_delta = summary.get("deltas", {}).get(item["name"], {}).get("skills", {})
                if skill in skill_delta:
                    row[f"delta_{skill}_am"] = skill_delta[skill]
        rows.append(row)
    if not rows:
        raise ValueError("no runs to write")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_skill_csv(summary: Mapping[str, Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in summary["runs"]:
        skill_delta = summary.get("deltas", {}).get(item["name"], {}).get("skills", {})
        for skill, payload in item["skills"].items():
            rows.append(
                {
                    "run": item["name"],
                    "skill": skill,
                    "count": payload["count"],
                    "soft_tifa_am": payload["soft_tifa_am"],
                    "delta_soft_tifa_am": skill_delta.get(skill),
                }
            )
    _write_dict_rows(rows, path)


def write_atomicity_csv(summary: Mapping[str, Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in summary["runs"]:
        for atomicity, payload in item["atomicity"].items():
            rows.append(
                {
                    "run": item["name"],
                    "atomicity": atomicity,
                    "count": payload["count"],
                    "soft_tifa_gm": payload["soft_tifa_gm"],
                }
            )
    _write_dict_rows(rows, path)


def _write_dict_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# GenEval2 Report", ""]
    baseline = summary.get("baseline")
    if baseline:
        lines.append(f"Baseline: `{baseline}`")
        lines.append("")

    headers = ["Run", "AM", "GM", "Atom AM", "ΔAM", "ΔGM"]
    lines.extend(_markdown_table(headers, [
        [
            item["name"],
            _fmt(item["overall"]["soft_tifa_am"]),
            _fmt(item["overall"]["soft_tifa_gm"]),
            _fmt(item["overall"]["atom_weighted_am"]),
            _fmt(summary.get("deltas", {}).get(item["name"], {}).get("overall", {}).get("soft_tifa_am")),
            _fmt(summary.get("deltas", {}).get(item["name"], {}).get("overall", {}).get("soft_tifa_gm")),
        ]
        for item in summary["runs"]
    ]))
    lines.append("")

    headers = ["Run", *SKILL_ORDER]
    lines.extend(_markdown_table(headers, [
        [item["name"], *[_fmt(item["skills"].get(skill, {}).get("soft_tifa_am")) for skill in SKILL_ORDER]]
        for item in summary["runs"]
    ]))
    lines.append("")

    all_atomicities = sorted(
        {int(key) for item in summary["runs"] for key in item.get("atomicity", {}).keys()}
    )
    headers = ["Run", *[str(key) for key in all_atomicities]]
    lines.extend(_markdown_table(headers, [
        [
            item["name"],
            *[_fmt(item.get("atomicity", {}).get(str(key), {}).get("soft_tifa_gm")) for key in all_atomicities],
        ]
        for item in summary["runs"]
    ]))
    lines.append("")
    return "\n".join(lines)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.2f}"
    return str(value)


def resolve_ordered_image_path(root: Path, index: int, *, sample_index: int = 0) -> Optional[Path]:
    stems = [
        root / f"{index:05d}" / "samples" / f"{sample_index:05d}.png",
        root / f"{index:05d}" / "samples" / f"{sample_index}.png",
        root / f"{index:05d}" / f"{sample_index:05d}.png",
        root / f"{index:05d}.png",
        root / f"{index:06d}.png",
    ]
    for path in stems:
        if path.is_file():
            return path
    return None


def build_image_filepath_map(
    benchmark: GenEval2Benchmark,
    image_root: str | Path,
    *,
    sample_index: int = 0,
    strict: bool = True,
) -> Dict[str, str]:
    root = Path(image_root)
    result: dict[str, str] = {}
    missing: list[int] = []
    for index, item in enumerate(benchmark.prompts):
        image_path = resolve_ordered_image_path(root, index, sample_index=sample_index)
        if image_path is None:
            missing.append(index)
            continue
        result[item.prompt] = str(image_path)
    if strict and missing:
        preview = ", ".join(str(i) for i in missing[:10])
        raise FileNotFoundError(
            f"missing {len(missing)} images under {root}; first missing indices: {preview}"
        )
    return result


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--run must use NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty run name in --run {value}")
    return name, Path(path)
