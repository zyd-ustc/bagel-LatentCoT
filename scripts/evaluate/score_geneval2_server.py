#!/usr/bin/env python3
"""Score a GenEval2 image map with the local Soft-TIFA reward server."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", required=True)
    parser.add_argument("--image-paths", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18086")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.benchmark_data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_paths = json.loads(Path(args.image_paths).read_text(encoding="utf-8"))
    score_lists, log_gm_scores = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        payload = {
            "images": [Path(image_paths[row["prompt"]]).read_bytes() for row in batch],
            "meta_datas": batch,
            "only_strict": True,
        }
        response = requests.post(
            args.server_url,
            data=pickle.dumps(payload),
            timeout=args.timeout_seconds,
        )
        response.raise_for_status()
        result = pickle.loads(response.content)
        if "error" in result:
            raise RuntimeError(result["error"])
        score_lists.extend(result["atom_scores"])
        log_gm_scores.extend(result["scores"])
        print(f"[{min(start + args.batch_size, len(rows))}/{len(rows)}]", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"score_lists": score_lists, "log_gm_scores": log_gm_scores},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
