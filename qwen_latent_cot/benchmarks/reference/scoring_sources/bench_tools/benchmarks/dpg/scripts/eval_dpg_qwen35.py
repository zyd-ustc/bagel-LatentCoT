#!/usr/bin/env python3
"""Score DPG-Bench images with the local Qwen3.5 VLM judge.

The official DPG evaluator uses mPLUG-large VQA with answer=='yes' scoring and
a Davidsonian Scene Graph dependency zero-out.  We replace mPLUG with Qwen3.5
(Ascend NPU, no CUDA) but keep the dependency graph logic, per-prompt score
formula and aggregation identical.  Result is labelled
``DPG-Bench-Qwen3.5-custom-VLM``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import judge_client as jc  # noqa: E402


SYSTEM_PROMPT = (
    "You are a strict visual question answering auditor. Look at the image and "
    "answer the question with a single word: yes or no. Answer yes only when the "
    "image clearly and unambiguously shows what the question asks. If it is "
    "missing, unclear, or only partially present, answer no."
)


def parse_dependencies(raw: str) -> list[int]:
    """Parse the official DPG dependency list.

    The official ``dpg_bench.csv`` stores zero or more parent proposition ids
    as a comma-separated string (for example ``0``, ``1,10`` or
    ``1,2,3,4``).  A dependency is satisfied only when all listed parents are
    satisfied; this is the same dependency zero-out rule used by the official
    evaluator.
    """
    value = raw.strip()
    if not value or value == "0":
        return []
    dependencies: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        dependency = int(token)
        if dependency <= 0:
            raise ValueError(f"invalid DPG dependency token: {token!r} in {raw!r}")
        dependencies.append(dependency)
    return dependencies


def load_questions(csv_path: Path) -> dict[str, list[dict]]:
    """Group all proposition rows by item_id, preserving proposition order."""
    grouped: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item_id = row["item_id"]
            grouped.setdefault(item_id, []).append(
                {
                    "proposition_id": int(row["proposition_id"]),
                    "dependencies": parse_dependencies(row["dependency"]),
                    "category_broad": row["category_broad"],
                    "category_detailed": row["category_detailed"],
                    "tuple": row["tuple"],
                    "question": row["question_natural_language"],
                }
            )
    return grouped


def evaluate_one(
    row: dict[str, Any],
    questions: list[dict],
    image_root: Path,
    api_base: str,
    model: str,
    timeout: float,
    rounds: int,
) -> dict[str, Any]:
    image_path = image_root / row["output_rel"]
    base = {**row, "image_path": str(image_path), "question_count": len(questions)}
    if not image_path.is_file():
        return {**base, "status": "missing_image", "prompt_score": 0.0}
    if not questions:
        return {**base, "status": "scored", "prompt_score": 0.0, "answers": {}}

    image_url = jc.image_data_url(image_path)
    raw_answers: dict[int, str] = {}
    for q in questions:
        last_err = ""
        for attempt in range(1, rounds + 1):
            try:
                raw, verdict = jc.call_judge_yesno(
                    api_base,
                    model,
                    SYSTEM_PROMPT,
                    q["question"],
                    [image_url],
                    timeout=timeout,
                    max_tokens=16,
                )
                if verdict is not None:
                    raw_answers[q["proposition_id"]] = "yes" if verdict == 1 else "no"
                    last_err = ""
                    break
                last_err = f"unparseable: {raw[:120]!r}"
            except Exception as exc:
                last_err = repr(exc)
            time.sleep(min(1.0 * attempt, 3.0))
        if q["proposition_id"] not in raw_answers:
            return {
                **base,
                "status": "judge_pending",
                "prompt_score": None,
                "answers": {str(k): v for k, v in raw_answers.items()},
                "error": last_err,
            }

    # Dependency zero-out in proposition order.
    s: dict[int, int] = {}
    for q in questions:
        pid = q["proposition_id"]
        dependencies = q["dependencies"]
        a_i = 1 if raw_answers[pid] == "yes" else 0
        if any(s.get(dep, 1) == 0 for dep in dependencies):
            s[pid] = 0
        else:
            s[pid] = a_i
    score = sum(s.values()) / len(questions)
    return {
        **base,
        "status": "scored",
        "prompt_score": round(score, 6),
        "answers": {str(k): v for k, v in raw_answers.items()},
        "graph_scores": {str(k): v for k, v in s.items()},
    }


def aggregate(manifest, rows, args, system_prompt, csv_path) -> None:
    by_id = {str(r["record_id"]): r for r in rows}
    ordered = [by_id[str(m["record_id"])] for m in manifest if str(m["record_id"]) in by_id]
    jc.write_jsonl_atomic(args.output_root / "results.jsonl", ordered)
    scored = [r for r in ordered if r.get("status") == "scored"]

    scores = [float(r["prompt_score"]) for r in scored]
    overall = (sum(scores) / len(scores) * 100.0) if scores else 0.0
    summary = {
        "benchmark": args.benchmark_label,
        "protocol": "DSG dependency zero-out + mean over prompts (Qwen3.5 VLM yes/no)",
        "judge_model": args.model,
        "api_base": args.api_base,
        "question_source": str(csv_path),
        "manifest_count": len(manifest),
        "scored_count": len(scored),
        "missing_image_count": sum(1 for r in ordered if r.get("status") == "missing_image"),
        "judge_pending_count": sum(1 for r in ordered if r.get("status") == "judge_pending"),
        "overall": round(overall, 4),
        "overall_formula": "mean(prompt_score) * 100; prompt_score=sum(graph_s)/N",
        "scale": "0-100",
        "system_prompt_sha256": jc.prompt_sha256(system_prompt),
    }
    jc.write_json_atomic(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-judge")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--benchmark-label", default="DPG-Bench-Qwen3.5-custom-VLM")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = jc.read_jsonl(args.manifest)
    questions_by_item = load_questions(args.csv)
    store = jc.ProgressStore(
        args.output_root / "progress.jsonl",
        terminal_statuses={"scored", "missing_image"},
    )
    pending = [m for m in manifest if not store.is_terminal(str(m["record_id"]))]
    print(
        f"[dpg] manifest={len(manifest)} pending={len(pending)} workers={args.max_workers}",
        flush=True,
    )

    def worker(row):
        existing = store.get(str(row["record_id"]))
        if existing and existing.get("status") in {"scored", "missing_image"}:
            return existing
        questions = questions_by_item.get(row["item_id"], [])
        result = evaluate_one(
            row, questions, args.image_root, args.api_base, args.model,
            args.timeout, args.rounds,
        )
        store.update(result)
        return result

    if pending:
        jc.run_threaded(pending, worker, args.max_workers, label="dpg")
    aggregate(manifest, store.all_rows(), args, SYSTEM_PROMPT, args.csv)


if __name__ == "__main__":
    main()
