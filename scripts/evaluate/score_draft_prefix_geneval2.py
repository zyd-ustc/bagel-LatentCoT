#!/usr/bin/env python3
"""Score a draft_prefix_loop run with the GenEval2 Soft-TIFA server.

Writes one image-map + score JSON per snapshot (baseline, r0, each mode r1/r3/final)
and a comparison table.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_latent_cot.evaluation.geneval2 import (  # noqa: E402
    arithmetic_mean,
    geometric_mean,
    load_benchmark,
    summarize_score_lists,
)


def _maps(output_dir: Path, prompts: list[str], t_tag: str, modes: list[str]) -> dict[str, dict[str, Path]]:
    snapshots: dict[str, dict[str, Path]] = {
        "baseline": {},
        "r0": {},
    }
    for mode in modes:
        for r in (1, 3):
            snapshots[f"{mode}_r{r}"] = {}
        snapshots[f"{mode}_final"] = {}
    for index, prompt in enumerate(prompts):
        pdir = output_dir / f"p{index:03d}"
        snapshots["baseline"][prompt] = pdir / "baseline.png"
        snapshots["r0"][prompt] = pdir / f"draft_{t_tag}_r0.png"
        for mode in modes:
            snapshots[f"{mode}_r1"][prompt] = pdir / f"draft_{t_tag}_{mode}_r1.png"
            snapshots[f"{mode}_r3"][prompt] = pdir / f"draft_{t_tag}_{mode}_r3.png"
            snapshots[f"{mode}_final"][prompt] = pdir / f"final_{t_tag}_{mode}.png"
    return snapshots


def _score_map(
    rows: list[dict],
    image_map: dict[str, Path],
    server_url: str,
    timeout: float,
    batch_size: int,
):
    import requests

    atom_scores, log_gm = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images, meta = [], []
        for row in batch:
            path = image_map[row["prompt"]]
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(path.read_bytes())
            meta.append(row)
        response = requests.post(
            server_url,
            data=pickle.dumps({"images": images, "meta_datas": meta, "only_strict": True}),
            timeout=timeout,
        )
        response.raise_for_status()
        result = pickle.loads(response.content)
        if "error" in result:
            raise RuntimeError(result["error"])
        atom_scores.extend(result["atom_scores"])
        log_gm.extend(result["scores"])
        print(f"    [{min(start + batch_size, len(rows))}/{len(rows)}]", flush=True)
    return atom_scores, log_gm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark-data",
        default="experiments/data/geneval2_hard_16.jsonl",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:18086")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    prompts = [str(p) for p in manifest["prompts"]]
    modes = [str(m) for m in manifest["gen_text_modes"]]
    t_tag = f"t{float(manifest['truncations'][0]):g}"
    rows = [
        json.loads(line)
        for line in Path(args.benchmark_data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["prompt"] in set(prompts)]
    rows.sort(key=lambda row: prompts.index(row["prompt"]))
    score_dir = output_dir / "geneval2"
    score_dir.mkdir(parents=True, exist_ok=True)
    filtered = score_dir / "benchmark.jsonl"
    filtered.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark(filtered)

    snapshots = _maps(output_dir, prompts, t_tag, modes)
    summaries = []
    for name, image_map in snapshots.items():
        print(f"[score] {name}", flush=True)
        atom_scores, log_gm = _score_map(
            rows,
            image_map,
            str(args.server_url),
            float(args.timeout_seconds),
            int(args.batch_size),
        )
        per_prompt = [
            {
                "prompt": row["prompt"],
                "atom_am": arithmetic_mean(scores),
                "atom_gm": geometric_mean(scores),
                "atom_scores": scores,
            }
            for row, scores in zip(rows, atom_scores)
        ]
        payload = {
            "name": name,
            "score_lists": atom_scores,
            "log_gm_scores": log_gm,
            "per_prompt": per_prompt,
        }
        (score_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        summary = summarize_score_lists(benchmark, atom_scores, name=name, source=name)
        summaries.append(
            {
                "name": name,
                "soft_tifa_am": summary["overall"]["soft_tifa_am"],
                "soft_tifa_gm": summary["overall"]["soft_tifa_gm"],
                "atom_weighted_am": summary["overall"]["atom_weighted_am"],
                "skills": summary["skills"],
                "per_prompt_am": [item["atom_am"] for item in per_prompt],
            }
        )
        print(
            f"    AM={summaries[-1]['soft_tifa_am']:.2f} GM={summaries[-1]['soft_tifa_gm']:.2f}",
            flush=True,
        )

    table_md = [
        "# GenEval2 Soft-TIFA — draft_prefix_loop",
        "",
        "| snapshot | Soft-TIFA AM | Soft-TIFA GM | atom-weighted AM |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        table_md.append(
            f"| {row['name']} | {row['soft_tifa_am']:.2f} | {row['soft_tifa_gm']:.2f} | {row['atom_weighted_am']:.2f} |"
        )
    csv_header = "prompt," + ",".join(row["name"] for row in summaries)
    csv_lines = [csv_header]
    for index, prompt in enumerate(prompts):
        cells = ",".join(f"{row['per_prompt_am'][index]:.4f}" for row in summaries)
        csv_lines.append(f"{json.dumps(prompt, ensure_ascii=False)},{cells}")
    (score_dir / "per_prompt_am.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )
    (score_dir / "summary.md").write_text("\n".join(table_md) + "\n", encoding="utf-8")
    (score_dir / "summary.json").write_text(
        json.dumps({"snapshots": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {score_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
