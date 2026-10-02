"""Check generated Phase 1A images and build prompt-aligned GenEval2 image maps."""

import argparse
import json
from pathlib import Path

from qwen_latent_cot.evaluation.image_scoring import prepare_phase1a_image_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    report = prepare_phase1a_image_maps(args.benchmark_data, args.image_dir, args.arm)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for arm, image_map in report["image_maps"].items():
        (args.output_dir / f"{arm}_image_map.json").write_text(json.dumps(image_map, indent=2) + "\n")
    (args.output_dir / "score_inputs.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output_dir), prompts=report["num_prompts"], arms=args.arm)))


if __name__ == "__main__":
    main()
