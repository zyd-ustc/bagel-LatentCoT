#!/usr/bin/env python3
"""Normalize official TIIF spatial records without rewriting their questions."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--description",
        choices=["short_description", "long_description"],
        default="short_description",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.source_dir).glob("*.jsonl")):
        for index, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if "spatial" not in row["type"].lower():
                continue
            questions, answers = row["yn_question_list"], row["yn_answer_list"]
            if (
                not questions
                or len(questions) != len(answers)
                or any(str(value).lower() not in {"yes", "no"} for value in answers)
            ):
                raise ValueError("TIIF source requires matched yes/no ground truth")
            rows.append(
                {
                    **row,
                    "prompt": row[args.description],
                    "source_file": str(path.resolve()),
                    "source_index": index,
                    "description_variant": args.description,
                }
            )
    if not rows:
        raise ValueError("no TIIF spatial records found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


if __name__ == "__main__":
    main()
