"""Matched native / untrained reader / step-5000 fixed-injection images."""

import argparse
import json
from pathlib import Path

from qwen_latent_cot.bagel.reader_injection_eval import build_plan, merge_shards, run_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-data", default="experiments/data/phase1a_semantic_hard64.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--gate-scale", type=float, default=1.)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--seed", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.checkpoint, args.benchmark_data, args.output_dir,
        gate_scale=args.gate_scale, max_prompts=args.max_prompts, num_shards=args.num_shards,
        seed=args.seed, model_path=args.model_path)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        saved = Path(plan["output"]) / "launch_plan.json"
        if saved.is_file() and json.loads(saved.read_text()) != plan:
            raise ValueError("launch plan changed between preflight and execution")
        if args.merge_only:
            merge_shards(plan)
        else:
            run_shard(plan, args.shard_id)


if __name__ == "__main__":
    main()
