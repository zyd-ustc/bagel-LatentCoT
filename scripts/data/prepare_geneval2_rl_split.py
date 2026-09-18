#!/usr/bin/env python3
"""Create a deterministic hard-composition RL/held-out split for GenEval2."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--min-atom-count", type=int, default=7)
    parser.add_argument("--eval-per-atomicity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.source).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hard = [row for row in rows if int(row["atom_count"]) >= args.min_atom_count]
    rng = random.Random(args.seed)
    train, held_out = [], []
    for atom_count in sorted({int(row["atom_count"]) for row in hard}):
        bucket = [row for row in hard if int(row["atom_count"]) == atom_count]
        rng.shuffle(bucket)
        if len(bucket) <= args.eval_per_atomicity:
            raise ValueError(f"atomicity {atom_count} has no room for an RL split")
        held_out.extend(bucket[: args.eval_per_atomicity])
        train.extend(bucket[args.eval_per_atomicity :])
    rng.shuffle(train)

    for split, output in ((train, args.train_output), (held_out, args.eval_output)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "hard_rows": len(hard),
                "train_rows": len(train),
                "held_out_rows": len(held_out),
                "train_atomicity": Counter(int(row["atom_count"]) for row in train),
                "held_out_atomicity": Counter(
                    int(row["atom_count"]) for row in held_out
                ),
            },
            default=dict,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
