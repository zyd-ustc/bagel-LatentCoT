#!/usr/bin/env python3
"""Summarize full-800 loop ablations and Z0 seed variability."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


METRICS = ("soft_tifa_am", "soft_tifa_gm", "atom_weighted_am")


def _load_summary(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing GenEval2 summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    match = next((run for run in summary["runs"] if run["name"] == name), None)
    if match is None:
        raise ValueError(f"run {name!r} is missing from score summary")
    return match


def summarize_multiseed(
    main_summary: Mapping[str, Any],
    extra_z0_summaries: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    main_seed: int = 42,
) -> dict[str, Any]:
    main_z0 = _find_run(main_summary, "Z0")
    z0_runs = [(int(main_seed), main_z0)] + [
        (int(seed), _find_run(summary, "Z0"))
        for seed, summary in extra_z0_summaries
    ]
    seed_rows = [
        {"seed": seed, **{metric: float(run["overall"][metric]) for metric in METRICS}}
        for seed, run in z0_runs
    ]

    variability = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in seed_rows]
        variability[metric] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }

    loop_rows = []
    for run in main_summary["runs"]:
        if run["name"] == "Z0":
            continue
        row = {"name": run["name"], "seed": int(main_seed)}
        for metric in METRICS:
            score = float(run["overall"][metric])
            stats = variability[metric]
            row[metric] = {
                "score": score,
                "delta_vs_z0_same_seed": score - float(main_z0["overall"][metric]),
                "delta_vs_z0_seed_mean": score - float(stats["mean"]),
                "outside_z0_seed_range": score < stats["min"] or score > stats["max"],
            }
        loop_rows.append(row)

    return {
        "schema": "bagel_loop_t2i_full800_multiseed_v1",
        "main_seed": int(main_seed),
        "z0_seed_count": len(seed_rows),
        "z0_seeds": seed_rows,
        "z0_variability": variability,
        "loop_arms": loop_rows,
        "interpretation_limit": (
            "Loop arms were evaluated only at the main seed; extra seeds quantify "
            "Z0 variability and do not constitute multi-seed loop evaluation."
        ),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# BAGEL loop T2I full-800 multi-seed summary",
        "",
        "## Z0 seed variability",
        "",
        "| Seed | Soft-TIFA AM | Soft-TIFA GM | Atom-weighted AM |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["z0_seeds"]:
        lines.append(
            f"| {row['seed']} | {row['soft_tifa_am']:.3f} | "
            f"{row['soft_tifa_gm']:.3f} | {row['atom_weighted_am']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Metric | Mean | Sample std | Min | Max | Range |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, stats in summary["z0_variability"].items():
        lines.append(
            f"| {metric} | {stats['mean']:.3f} | {stats['sample_std']:.3f} | "
            f"{stats['min']:.3f} | {stats['max']:.3f} | {stats['range']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"## Loop arms at seed {summary['main_seed']}",
            "",
            "| Arm | AM | ΔAM vs same-seed Z0 | GM | ΔGM vs same-seed Z0 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["loop_arms"]:
        am = row["soft_tifa_am"]
        gm = row["soft_tifa_gm"]
        lines.append(
            f"| {row['name']} | {am['score']:.3f} | "
            f"{am['delta_vs_z0_same_seed']:+.3f} | {gm['score']:.3f} | "
            f"{gm['delta_vs_z0_same_seed']:+.3f} |"
        )
    lines.extend(["", f"> {summary['interpretation_limit']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root_dir.expanduser().resolve()
    score_name = Path("geneval2/geneval2_summary.json")
    main_summary = _load_summary(root / "seed_42_main" / score_name)
    extras = [
        (seed, _load_summary(root / f"seed_{seed}_z0" / score_name))
        for seed in (43, 44)
    ]
    summary = summarize_multiseed(main_summary, extras, main_seed=42)
    json_path = root / "multiseed_summary.json"
    markdown_path = root / "multiseed_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(markdown_path, flush=True)


if __name__ == "__main__":
    main()
