#!/usr/bin/env python3
"""Score GenEval images with the local Qwen3.5 VLM judge.

The official GenEval metric uses Mask2Former detection + CLIP zero-shot color
classification.  We have only Ascend NPU (no CUDA/mmdet), so this evaluator
replaces the detector+CLIP with a Qwen3.5 VLM that answers per-clause yes/no
visual questions.  The clause-evaluation LOGIC (include AND, exclude OR-fail,
counting exactly-N semantics) and the aggregation (per-category mean, then
unweighted mean of 6 categories) are kept identical to the official
``evaluate_images.py`` / ``summary_scores.py``.

Result must be labelled ``GenEval-Qwen3.5-custom-VLM``; it is NOT comparable
to official Mask2Former+CLIP numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3] / "common"
    ),
)
import judge_client as jc  # noqa: E402


CATALOG = [
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
]


def build_questions(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build a list of yes/no questions, one per clause, from include/exclude.

    Each question dict has: ``id``, ``question``, ``must_be`` ("yes"|"no").
    A clause passes when the VLM answer equals ``must_be``.
    """
    questions: list[dict[str, str]] = []
    include = row.get("geneval_include", [])
    exclude = row.get("geneval_exclude", [])

    for ci, clause in enumerate(include):
        cls = clause["class"]
        count = int(clause.get("count", 1))
        color = clause.get("color")
        position = clause.get("position")
        noun = cls if count == 1 else cls + "s"
        if color:
            q = (
                f"Does the image contain at least {count} {color} {noun}?"
            )
        else:
            q = f"Does the image contain at least {count} {noun}?"
        questions.append({"id": f"include_{ci}", "question": q, "must_be": "yes"})

        if position:
            rel, target_idx = position[0], int(position[1])
            target_cls = include[target_idx]["class"]
            target_noun = target_cls
            q = (
                f"Is there a {cls} that is positioned {rel} {target_noun} in the image? "
                f"Answer based only on clear visual evidence."
            )
            questions.append(
                {"id": f"include_{ci}_position", "question": q, "must_be": "yes"}
            )

    for ci, clause in enumerate(exclude):
        cls = clause["class"]
        count = int(clause.get("count", 1))
        noun = cls if count == 1 else cls + "s"
        # exclude fails when >= count objects exist; pass requires NO (not too many)
        q = f"Does the image contain {count} or more {noun}?"
        questions.append({"id": f"exclude_{ci}", "question": q, "must_be": "no"})

    return questions


SYSTEM_PROMPT = (
    "You are a strict visual auditor for text-to-image generation. "
    "You answer yes/no questions about whether specific objects, colors, counts, "
    "or spatial relations are clearly and unambiguously visible in an image. "
    "Answer with a single word: yes or no. Ambiguity means no."
)


def evaluate_one(
    row: dict[str, Any],
    image_root: Path,
    api_base: str,
    model: str,
    timeout: float,
    rounds: int,
) -> dict[str, Any]:
    image_path = image_root / row["output_rel"]
    base = {**row, "image_path": str(image_path)}
    if not image_path.is_file():
        return {**base, "status": "missing_image", "correct": 0}
    questions = build_questions(row)
    if not questions:
        return {**base, "status": "scored", "correct": 1, "answers": {}}

    image_url = jc.image_data_url(image_path)
    answers: dict[str, str] = {}
    raw_log: list[str] = []
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
                raw_log.append(raw.strip())
                if verdict is not None:
                    answers[q["id"]] = "yes" if verdict == 1 else "no"
                    last_err = ""
                    break
                last_err = f"unparseable: {raw[:120]!r}"
            except Exception as exc:
                last_err = repr(exc)
            time.sleep(min(1.0 * attempt, 3.0))
        if q["id"] not in answers:
            # A judge failure is NOT a model failure; leave pending.
            return {
                **base,
                "status": "judge_pending",
                "correct": None,
                "answers": answers,
                "error": last_err,
                "raw": raw_log,
            }

    correct = 1
    for q in questions:
        if answers[q["id"]] != q["must_be"]:
            correct = 0
            break
    return {
        **base,
        "status": "scored",
        "correct": correct,
        "answers": answers,
        "raw": raw_log,
    }


def aggregate(manifest: list[dict], rows: list[dict], args, system_prompt: str) -> None:
    by_id = {str(r["record_id"]): r for r in rows}
    ordered = [by_id[str(m["record_id"])] for m in manifest if str(m["record_id"]) in by_id]
    jc.write_jsonl_atomic(args.output_root / "results.jsonl", ordered)
    scored = [r for r in ordered if r.get("status") == "scored"]

    cat_scores: dict[str, list[int]] = {}
    for row in scored:
        cat_scores.setdefault(row["category"], []).append(int(row["correct"]))
    category_mean = {
        c: (sum(v) / len(v) if v else 0.0) for c, v in cat_scores.items()
    }
    present_cats = [c for c in CATALOG if c in cat_scores and cat_scores[c]]
    overall = (
        sum(category_mean[c] for c in present_cats) / len(present_cats)
        if present_cats
        else 0.0
    )
    summary = {
        "benchmark": args.benchmark_label,
        "protocol": "GenEval clause logic (include AND / exclude OR-fail) via Qwen3.5 VLM yes/no",
        "judge_model": args.model,
        "api_base": args.api_base,
        "manifest_count": len(manifest),
        "scored_count": len(scored),
        "missing_image_count": sum(1 for r in ordered if r.get("status") == "missing_image"),
        "judge_pending_count": sum(1 for r in ordered if r.get("status") == "judge_pending"),
        "category_scores": {c: round(category_mean.get(c, 0.0), 4) for c in CATALOG},
        "overall": round(overall, 4),
        "overall_formula": "unweighted mean of 6 category means",
        "scale": "0-1",
        "system_prompt_sha256": jc.prompt_sha256(system_prompt),
    }
    jc.write_json_atomic(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-judge")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--benchmark-label", default="GenEval-Qwen3.5-custom-VLM")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = jc.read_jsonl(args.manifest)
    store = jc.ProgressStore(
        args.output_root / "progress.jsonl",
        terminal_statuses={"scored", "missing_image"},
    )
    pending = [m for m in manifest if not store.is_terminal(str(m["record_id"]))]
    print(
        f"[geneval] manifest={len(manifest)} pending={len(pending)} workers={args.max_workers}",
        flush=True,
    )

    def worker(row):
        existing = store.get(str(row["record_id"]))
        if existing and existing.get("status") in {"scored", "missing_image"}:
            return existing
        result = evaluate_one(
            row, args.image_root, args.api_base, args.model, args.timeout, args.rounds
        )
        store.update(result)
        return result

    if pending:
        jc.run_threaded(pending, worker, args.max_workers, label="geneval")
    aggregate(manifest, store.all_rows(), args, SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
