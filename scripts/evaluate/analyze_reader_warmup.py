"""Generate a fresh CPU-only report from existing Phase 1A.0 logs."""

import argparse
import json
from pathlib import Path

from qwen_latent_cot.evaluation.offline_reader import analyze_reader, render_reader_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heldout-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=4000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    report = analyze_reader(args.run_dir, args.heldout_data, seed=args.seed, resamples=args.resamples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "reader_diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "reader_diagnosis.md").write_text(render_reader_report(report))
    print(json.dumps(dict(output=str(args.output_dir), verdict=report["verdict"],
                          last_update=report["latest_step"], last_eval=report["checkpoints"][-1]["step"])))


if __name__ == "__main__":
    main()
