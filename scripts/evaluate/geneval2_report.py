#!/usr/bin/env python3
"""Aggregate official GenEval2 score lists into report-ready tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_latent_cot.evaluation.geneval2 import (  # noqa: E402
    compare_summaries,
    load_benchmark,
    load_score_lists,
    parse_run_arg,
    render_markdown,
    write_atomicity_csv,
    write_skill_csv,
    summarize_score_lists,
    write_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run score file in NAME=score_lists.json format. Repeat for multiple runs.",
    )
    parser.add_argument("--baseline-run", type=str, default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_data)
    summaries = []
    for run_arg in args.run:
        name, score_path = parse_run_arg(run_arg)
        score_lists = load_score_lists(score_path, benchmark)
        summaries.append(
            summarize_score_lists(
                benchmark,
                score_lists,
                name=name,
                source=str(score_path),
            )
        )

    result = compare_summaries(
        summaries,
        baseline_name=args.baseline_run or None,
    )
    result["benchmark_data"] = str(args.benchmark_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "geneval2_summary.json"
    summary_csv = args.output_dir / "geneval2_summary.csv"
    skill_csv = args.output_dir / "geneval2_skills.csv"
    atomicity_csv = args.output_dir / "geneval2_atomicity.csv"
    summary_md = args.output_dir / "geneval2_summary.md"

    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(result, summary_csv)
    write_skill_csv(result, skill_csv)
    write_atomicity_csv(result, atomicity_csv)
    summary_md.write_text(render_markdown(result), encoding="utf-8")

    print(summary_json)
    print(summary_csv)
    print(skill_csv)
    print(atomicity_csv)
    print(summary_md)


if __name__ == "__main__":
    main()
