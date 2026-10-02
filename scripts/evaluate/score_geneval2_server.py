#!/usr/bin/env python3
"""Score a GenEval2 image map with the local Soft-TIFA reward server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_latent_cot.evaluation.image_scoring import score_image_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", required=True)
    parser.add_argument("--image-paths", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18086")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scores: {output}")
    result = score_image_map(args.benchmark_data, args.image_paths,
        server_url=args.server_url, batch_size=args.batch_size, timeout_seconds=args.timeout_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
