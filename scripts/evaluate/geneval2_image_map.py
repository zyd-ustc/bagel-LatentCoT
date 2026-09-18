#!/usr/bin/env python3
"""Create the prompt-to-image JSON required by refs/GenEval2/evaluation.py."""

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
    build_image_filepath_map,
    load_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write available prompt mappings instead of failing on missing images.",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_data)
    image_map = build_image_filepath_map(
        benchmark,
        args.image_root,
        sample_index=args.sample_index,
        strict=not args.allow_partial,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(image_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)
    print(f"mapped={len(image_map)} total={benchmark.prompt_count}")


if __name__ == "__main__":
    main()

