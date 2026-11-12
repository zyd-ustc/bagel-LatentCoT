#!/usr/bin/env python3
"""Score T2I-CoReBench images with the local Qwen3.5 VLM judge.

Qwen3.5-35B-A3B is an officially supported evaluator in the T2I-CoReBench
``evaluate.py``.  We replicate the exact system prompt (TEMPLATE), the binary
yes/no parser, the per-image ratio score and the unweighted per-dimension /
overall aggregation.  Judge failures remain pending (never mapped to model 0).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import judge_client as jc  # noqa: E402


COMPOSITION = ["C-MI", "C-MA", "C-MR", "C-TR"]
REASONING = [
    "R-LR", "R-BR", "R-HR", "R-PR", "R-GR", "R-AR", "R-CR", "R-RR",
]
ALL_DIMS = COMPOSITION + REASONING


# Verbatim TEMPLATE from the official evaluate.py (used as the system message).
SYSTEM_PROMPT = """You are an AI quality auditor for text-to-image generation.

Your task is to analyze the given image and answer a yes/no question based solely on its visual content. The question may relate to the presence of a specific object, its attributes, or relationships between multiple elements in the image.

You will also be given the original prompt used to generate the image. The prompt may provide additional context to help interpret the question, but it must never be used to supply or assume visual details.
Your judgment must rely entirely on the image itself. The image must contain clear, unmistakable visual evidence to justify a "yes" answer — the prompt cannot compensate for missing or ambiguous content.

Respond with:
- "yes" only if the answer is **clearly and unambiguously** yes based solely on the visual content. The visual evidence must be **strong, definitive, and require no assumptions or guesses**.
- "no" in **all other cases** — including if the relevant visual detail is missing, unclear, ambiguous, partially shown, obscured, or only suggested.

Even if the image closely matches what is described in the prompt, you must rely on **visible evidence** alone. If the relevant detail cannot be confirmed visually with certainty, answer "no".
**Ambiguity equals no.**

For conditional questions, answer "yes" only if **both** the condition and the main clause are **clearly and unambiguously true** in the image. If **either part** is false or uncertain, respond "no".

Do **not** provide any explanation, justification, or extra text.
Only return a single word: either "yes" or "no".

Example input:
Prompt: "a golden retriever running in a grassy field under the sun"\nQuestion: "Is there a sun in the image?"
Example output:
"yes"

Example input:
Prompt: "a white cat sitting on a red couch in a modern living room"
Question: "Is the couch is present, is it red in color?"
Example output:
"no"
"""


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
    checklist = row.get("checklist", [])
    if not image_path.is_file():
        return {**base, "status": "missing_image", "image_score": 0.0}
    if not checklist:
        return {**base, "status": "scored", "image_score": 0.0, "answers": []}

    image_url = jc.image_data_url(image_path)
    answers: list[dict[str, Any]] = []
    for qi, q in enumerate(checklist):
        question = q.get("question", "")
        user_text = f'Prompt: "{row["instruction"]}"\nQuestion: "{question}"'
        last_err = ""
        for attempt in range(1, rounds + 1):
            try:
                raw, verdict = jc.call_judge_yesno(
                    api_base,
                    model,
                    SYSTEM_PROMPT,
                    user_text,
                    [image_url],
                    timeout=timeout,
                    max_tokens=16,
                )
                if verdict is not None:
                    answers.append(
                        {"question": question, "answer": "yes" if verdict == 1 else "no"}
                    )
                    last_err = ""
                    break
                last_err = f"unparseable: {raw[:120]!r}"
            except Exception as exc:
                last_err = repr(exc)
            time.sleep(min(1.0 * attempt, 3.0))
        if len(answers) <= qi:
            return {
                **base,
                "status": "judge_pending",
                "image_score": None,
                "answers": answers,
                "error": last_err,
            }

    valid = [a for a in answers if a["answer"] in ("yes", "no")]
    score = sum(1 for a in valid if a["answer"] == "yes") / len(valid) if valid else 0.0
    return {
        **base,
        "status": "scored",
        "image_score": round(score, 6),
        "answers": answers,
    }


def aggregate(manifest, rows, args, system_prompt) -> None:
    by_id = {str(r["record_id"]): r for r in rows}
    ordered = [by_id[str(m["record_id"])] for m in manifest if str(m["record_id"]) in by_id]
    jc.write_jsonl_atomic(args.output_root / "results.jsonl", ordered)
    scored = [r for r in ordered if r.get("status") == "scored"]

    dim_scores: dict[str, list[float]] = {}
    for r in scored:
        dim_scores.setdefault(r["dimension"], []).append(float(r["image_score"]))
    dim_mean = {d: (sum(v) / len(v) if v else 0.0) for d, v in dim_scores.items()}

    comp = [dim_mean[d] for d in COMPOSITION if d in dim_scores and dim_scores[d] is not None]
    reas = [dim_mean[d] for d in REASONING if d in dim_scores and dim_scores[d] is not None]
    all_means = [dim_mean[d] for d in ALL_DIMS if d in dim_scores and dim_scores[d] is not None]
    summary = {
        "benchmark": args.benchmark_label,
        "protocol": "T2I-CoReBench checklist yes/no (official TEMPLATE) via Qwen3.5",
        "judge_model": args.model,
        "api_base": args.api_base,
        "manifest_count": len(manifest),
        "scored_count": len(scored),
        "missing_image_count": sum(1 for r in ordered if r.get("status") == "missing_image"),
        "judge_pending_count": sum(1 for r in ordered if r.get("status") == "judge_pending"),
        "dimension_scores": {d: round(dim_mean.get(d, 0.0), 4) for d in ALL_DIMS},
        "composition_mean": round(sum(comp) / len(comp), 4) if comp else 0.0,
        "reasoning_mean": round(sum(reas) / len(reas), 4) if reas else 0.0,
        "overall": round(sum(all_means) / len(all_means), 4) if all_means else 0.0,
        "overall_formula": "unweighted mean of 12 dimension means",
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
    parser.add_argument("--benchmark-label", default="T2I-CoReBench-Qwen3.5")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = jc.read_jsonl(args.manifest)
    store = jc.ProgressStore(
        args.output_root / "progress.jsonl",
        terminal_statuses={"scored", "missing_image"},
    )
    pending = [m for m in manifest if not store.is_terminal(str(m["record_id"]))]
    print(
        f"[t2i-corebench] manifest={len(manifest)} pending={len(pending)} workers={args.max_workers}",
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
        jc.run_threaded(pending, worker, args.max_workers, label="t2i-corebench")
    aggregate(manifest, store.all_rows(), args, SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
