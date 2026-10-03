#!/usr/bin/env python3
"""Report external semantic scores separately from velocity diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json

from qwen_latent_cot.evaluation.geneval2 import (
    load_benchmark,
    load_score_lists,
    render_markdown,
    summarize_score_lists,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    benchmark = load_benchmark(args.benchmark)
    summary = summarize_score_lists(
        benchmark, load_score_lists(args.scores, benchmark), name=args.name
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(render_markdown(summary))


if __name__ == "__main__":
    main()
