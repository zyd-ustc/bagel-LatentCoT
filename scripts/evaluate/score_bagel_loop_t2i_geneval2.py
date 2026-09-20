#!/usr/bin/env python3
"""Score every T2I loop arm with the GenEval2 Soft-TIFA server."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_latent_cot.evaluation.geneval2 import (  # noqa: E402
    compare_summaries,
    load_benchmark,
    render_markdown,
    summarize_score_lists,
    write_atomicity_csv,
    write_skill_csv,
    write_summary_csv,
)


def load_generation_inputs(
    output_dir: Path,
    benchmark_path: Path,
) -> tuple[list[dict], list[dict], Dict[str, Dict[str, str]]]:
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    prompts = [str(prompt) for prompt in manifest["prompts"]]
    prompt_set = set(prompts)
    rows = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_prompt = {str(row["prompt"]): row for row in rows}
    missing = [prompt for prompt in prompts if prompt not in by_prompt]
    if missing:
        raise ValueError(f"benchmark is missing generated prompts: {missing[:3]}")
    rows = [by_prompt[prompt] for prompt in prompts]

    arm_maps = {}
    for arm in manifest["arms"]:
        arm_id = str(arm["id"])
        relative = manifest["geneval2_image_maps"][arm_id]
        image_map = json.loads((output_dir / relative).read_text(encoding="utf-8"))
        absent = [prompt for prompt in prompts if not Path(image_map.get(prompt, "")).is_file()]
        if absent:
            raise FileNotFoundError(
                f"{arm_id}: missing {len(absent)} generated images; first={absent[0]!r}"
            )
        arm_maps[arm_id] = {prompt: str(image_map[prompt]) for prompt in prompt_set}
    return rows, list(manifest["arms"]), arm_maps


def score_image_map(
    rows: Sequence[Dict[str, Any]],
    image_map: Dict[str, str],
    *,
    server_url: str,
    timeout_seconds: float,
    batch_size: int,
) -> tuple[list[list[float]], list[float]]:
    atom_scores, log_gm_scores = [], []
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        payload = {
            "images": [Path(image_map[str(row["prompt"])]).read_bytes() for row in batch],
            "meta_datas": list(batch),
            "only_strict": True,
        }
        response = requests.post(
            server_url,
            data=pickle.dumps(payload),
            timeout=float(timeout_seconds),
        )
        response.raise_for_status()
        result = pickle.loads(response.content)
        if "error" in result:
            raise RuntimeError(result["error"])
        atom_scores.extend(result["atom_scores"])
        log_gm_scores.extend(result["scores"])
        print(f"    [{min(start + batch_size, len(rows))}/{len(rows)}]", flush=True)
    return atom_scores, log_gm_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-data",
        type=Path,
        default=Path("experiments/data/geneval2_hard_128.jsonl"),
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:18086")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    rows, arms, arm_maps = load_generation_inputs(
        output_dir, args.benchmark_data.expanduser().resolve()
    )
    score_dir = output_dir / "geneval2"
    filtered_benchmark = score_dir / "benchmark.jsonl"
    filtered_benchmark.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    benchmark = load_benchmark(filtered_benchmark)

    summaries = []
    for arm in arms:
        arm_id = str(arm["id"])
        print(f"[score] {arm_id}", flush=True)
        atom_scores, log_gm_scores = score_image_map(
            rows,
            arm_maps[arm_id],
            server_url=str(args.server_url),
            timeout_seconds=float(args.timeout_seconds),
            batch_size=int(args.batch_size),
        )
        score_path = score_dir / f"{arm['slug']}_scores.json"
        score_path.write_text(
            json.dumps(
                {
                    "name": arm_id,
                    "score_lists": atom_scores,
                    "log_gm_scores": log_gm_scores,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(
            summarize_score_lists(
                benchmark,
                atom_scores,
                name=arm_id,
                source=str(score_path),
            )
        )

    result = compare_summaries(summaries, baseline_name="Z0")
    result["benchmark_data"] = str(filtered_benchmark)
    summary_json = score_dir / "geneval2_summary.json"
    summary_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(result, score_dir / "geneval2_summary.csv")
    write_skill_csv(result, score_dir / "geneval2_skills.csv")
    write_atomicity_csv(result, score_dir / "geneval2_atomicity.csv")
    (score_dir / "geneval2_summary.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(summary_json, flush=True)


if __name__ == "__main__":
    main()
