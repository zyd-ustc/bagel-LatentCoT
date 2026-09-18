#!/usr/bin/env python3
"""Score every candidate in a BAGEL loop group with GenEval2 Soft-TIFA."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import requests


def _load_metadata(path: Path, prompt: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("prompt") == prompt:
            return row
    raise ValueError(f"prompt is not present in benchmark data: {prompt!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--benchmark-data", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18086")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prompt = str(report["prompt"])
    metadata = _load_metadata(Path(args.benchmark_data), prompt)
    candidates = []
    for run in report["runs"]:
        for depth in ("depth1", "depth2"):
            image_path = Path(run["images"][depth])
            if not image_path.is_absolute():
                image_path = report_path.parent / image_path
            candidates.append(
                {
                    "seed": int(run["seed"]),
                    "depth": depth,
                    "image": str(image_path),
                }
            )

    payload = {
        "images": [Path(row["image"]).read_bytes() for row in candidates],
        "meta_datas": [metadata] * len(candidates),
        "only_strict": True,
    }
    response = requests.post(
        args.server_url,
        data=pickle.dumps(payload),
        timeout=float(args.timeout_seconds),
    )
    response.raise_for_status()
    result = pickle.loads(response.content)
    if "error" in result:
        raise RuntimeError(result["error"])
    if len(result["scores"]) != len(candidates):
        raise RuntimeError("GenEval2 result length does not match candidate count")

    rows = []
    for candidate, log_gm, atom_scores in zip(
        candidates, result["scores"], result["atom_scores"]
    ):
        rows.append(
            {
                **candidate,
                "semantic_log_gm": float(log_gm),
                "semantic_gm_percent": 100.0 * math.exp(float(log_gm)),
                "atom_scores": [float(value) for value in atom_scores],
            }
        )
    output = {
        "schema": "bagel_rl_zero_shot_geneval2_group_v1",
        "prompt": prompt,
        "metadata": metadata,
        "candidates": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
