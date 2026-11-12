#!/usr/bin/env python3
"""Score the official Step1X-Edit GEdit-Bench English images with local Qwen3.5.

The image source is the downloaded Shiyu95/gedit_results repository.  The
scoring prompts and parser are imported unchanged from the official
GEdit-Bench VIEScore implementation.  This evaluator only changes the image
root and adds resumable parsing/retries for the local vLLM endpoint.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageOps


DEFAULT_REPO = Path(
    "/mnt/sfs/zx_wang_add_vg_output/unimoe_joint_vg_milestone_formal500_v2_review"
    "/eval_formal_20260811/repos/Step1X-Edit/GEdit-Bench"
)
DEFAULT_MANIFEST = Path(
    "/mnt/sfs/zx_wang_add_vg_output/unimoe_joint_vg_milestone_formal500_v2_review"
    "/eval_formal_20260811/manifests/gedit_en.jsonl"
)
DEFAULT_IMAGE_ROOT = Path(
    "/mnt/sfs/zx_wang_add_vg_output/unimoe_joint_vg_milestone_formal500_v2_review"
    "/eval_formal_20260811/official_gedit_results_shiyu95_20260812"
    "/gedit_results/step1x-edit-v1p1/fullset"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def protocol(repo: Path):
    sys.path.insert(0, str(repo / "viescore"))
    from utils import mllm_output_to_dict  # type: ignore
    from vie_prompts import (  # type: ignore
        _context_no_delimit,
        _prompts_0shot_rule_PQ,
        _prompts_0shot_tie_rule_SC,
        _prompts_0shot_two_image_edit_rule,
    )

    sc = "\n".join(
        [_context_no_delimit, _prompts_0shot_two_image_edit_rule, _prompts_0shot_tie_rule_SC]
    )
    pq = "\n".join([_context_no_delimit, _prompts_0shot_rule_PQ])
    return sc, pq, mllm_output_to_dict


def image_data_url(path: Path) -> str:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    ratio = image.width / image.height
    width = max(1, int(math.sqrt(512 * 512 * ratio)))
    height = max(1, int(width / ratio))
    image = image.resize((width, height))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def clean_response(content: str) -> str:
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    return re.sub(r"</think>\s*", "", content).strip()


def ask_vllm(
    api_base: str,
    model: str,
    prompt: str,
    image_urls: list[str],
    timeout: int,
    max_tokens: int,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict professional digital artist and image editor evaluator.",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return clean_response(body["choices"][0]["message"]["content"])


def valid_scores(value: Any) -> list[int] | None:
    if isinstance(value, (int, float, str)):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return None
    scores: list[int] = []
    for value in values:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return None
        if not 0 <= score <= 10:
            return None
        scores.append(score)
    return scores or None


def parse_scores(raw: str, parser) -> tuple[list[int] | None, str]:
    """Parse official output, accepting a scalar score as a recovery case.

    The official parser is tried first.  The two regex fallbacks only recover
    an explicit JSON-like ``score`` field; they do not infer a score from
    rubric text.
    """
    try:
        parsed = parser(raw, give_up_parsing=False)
    except Exception:
        parsed = None
    if isinstance(parsed, dict) and "score" in parsed:
        scores = valid_scores(parsed["score"])
        if scores is not None:
            mode = "official_parser" if len(scores) >= 2 else "official_parser_scalar_recovered"
            return scores, mode

    field = re.search(r"['\"]score['\"]\s*:\s*(\[[^\]]*\]|-?\d+(?:\.\d+)?)", raw, re.I | re.S)
    if field:
        value = field.group(1)
        if value.startswith("["):
            values = re.findall(r"-?\d+(?:\.\d+)?", value)
            scores = valid_scores(values)
        else:
            scores = valid_scores(value)
        if scores is not None:
            mode = "explicit_score_field" if len(scores) >= 2 else "explicit_score_scalar_recovered"
            return scores, mode

    bracket = re.search(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*\]", raw)
    if bracket:
        scores = valid_scores(re.split(r"\s*,\s*", bracket.group(1)))
        if scores is not None:
            return scores, "explicit_score_list_recovered"
    return None, "unparseable"


def official_image_path(image_root: Path, item: dict[str, Any]) -> Path | None:
    relative = Path(item["output_rel"])
    if relative.parts and relative.parts[0] == "fullset":
        relative = relative.relative_to("fullset")
    for suffix in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = image_root / relative.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def evaluate_one(
    item: dict[str, Any],
    image_root: Path,
    api_base: str,
    model: str,
    sc_prompt: str,
    pq_prompt: str,
    score_parser,
    timeout: int,
    rounds: int,
    image_source_label: str,
    existing_row: dict[str, Any] | None = None,
    persist_partial: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    edited_path = official_image_path(image_root, item)
    source_path = Path(item["source_image"])
    base = {
        **item,
        "image_source": image_source_label,
        "edited_image": str(edited_path) if edited_path else str(image_root / item["output_rel"]),
    }
    if edited_path is None:
        return {**base, "status": "missing_official_image", "ok": False}
    if not source_path.is_file():
        return {**base, "status": "missing_source_image", "ok": False}

    source_url = image_data_url(source_path)
    edited_url = image_data_url(edited_path)
    carry = {
        key: value
        for key, value in (existing_row or {}).items()
        if key.startswith("sc_") or key.startswith("pq_")
    }
    row = {**base, **carry}

    def score_dimension(
        prefix: str,
        prompt: str,
        image_urls: list[str],
    ) -> bool:
        if valid_scores(row.get(f"{prefix}_scores")) is not None:
            return True
        prior_attempts = int(row.get(f"{prefix}_attempts", 0) or 0)
        last_error = ""
        for local_attempt in range(1, rounds + 1):
            # Keep the official prompt unchanged. On a later attempt only the
            # response budget grows, matching the historical evaluator.
            # Attempts are persisted across outer resume passes. Do not reset
            # the response budget to 500 merely because a new process starts.
            cumulative_attempt = prior_attempts + local_attempt
            if cumulative_attempt == 1:
                max_tokens = 500
            elif cumulative_attempt <= 3:
                max_tokens = 1000
            else:
                # A very small number of otherwise valid judge responses can
                # spend more than 1,000 tokens explaining the rubric before
                # emitting the required JSON.  Preserve the official prompt
                # and parser; only give late format-recovery attempts enough
                # room to reach their final answer.
                max_tokens = 2000
            started = time.monotonic()
            try:
                raw = ask_vllm(
                    api_base,
                    model,
                    prompt,
                    image_urls,
                    timeout,
                    max_tokens,
                )
                elapsed = time.monotonic() - started
                scores, parse_mode = parse_scores(raw, score_parser)
                row[f"{prefix}_seconds"] = elapsed
                row[f"{prefix}_attempts"] = prior_attempts + local_attempt
                row[f"{prefix}_raw"] = raw
                row[f"{prefix}_parse_mode"] = parse_mode
                if scores is not None:
                    row[f"{prefix}_scores"] = scores
                    row.pop(f"{prefix}_error", None)
                    return True
                last_error = f"unparseable: {prefix}={parse_mode} {raw!r}"
            except Exception as exc:
                row[f"{prefix}_seconds"] = time.monotonic() - started
                row[f"{prefix}_attempts"] = prior_attempts + local_attempt
                last_error = repr(exc)
            row[f"{prefix}_error"] = last_error
        return False

    if not score_dimension(
        "sc",
        sc_prompt.replace("<instruction>", item["instruction"]),
        [source_url, edited_url],
    ):
        return {
            **row,
            "status": "judge_failed",
            "ok": False,
            "failed_dimension": "sc",
            "error": row.get("sc_error", "SC scoring failed"),
            "attempts": row.get("sc_attempts", rounds),
        }

    # Persist SC before beginning PQ. Previously a PQ timeout discarded a
    # successful SC response and the next pass repeated both multimodal calls.
    row.update({"status": "partial_scored", "ok": False, "pending_dimension": "pq"})
    if persist_partial is not None:
        persist_partial(dict(row))

    if not score_dimension("pq", pq_prompt, [edited_url]):
        return {
            **row,
            "status": "judge_failed",
            "ok": False,
            "failed_dimension": "pq",
            "error": row.get("pq_error", "PQ scoring failed"),
            "attempts": max(
                int(row.get("sc_attempts", 0) or 0),
                int(row.get("pq_attempts", 0) or 0),
            ),
        }

    sc_scores = valid_scores(row["sc_scores"])
    pq_scores = valid_scores(row["pq_scores"])
    assert sc_scores is not None and pq_scores is not None
    semantic = min(sc_scores)
    quality = min(pq_scores)
    row.pop("error", None)
    row.pop("failed_dimension", None)
    row.pop("pending_dimension", None)
    return {
        **row,
        "status": "scored",
        "ok": True,
        "semantic_score": semantic,
        "quality_score": quality,
        "overall_score": math.sqrt(semantic * quality),
        "attempts": max(
            int(row.get("sc_attempts", 0) or 0),
            int(row.get("pq_attempts", 0) or 0),
        ),
    }


def mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def aggregate(
    manifest: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    sc_prompt: str,
    pq_prompt: str,
) -> None:
    by_id = {str(row["record_id"]): row for row in rows}
    ordered = [by_id[str(item["record_id"])] for item in manifest if str(item["record_id"]) in by_id]
    write_jsonl_atomic(args.output_root / "results.jsonl", ordered)
    scored = [row for row in ordered if row.get("status") == "scored"]
    task_summary: dict[str, dict[str, Any]] = {}
    for task in sorted({row["task_type"] for row in scored}):
        task_rows = [row for row in scored if row["task_type"] == task]
        task_summary[task] = {
            "count": len(task_rows),
            "semantic_score": mean(task_rows, "semantic_score"),
            "quality_score": mean(task_rows, "quality_score"),
            "overall_score": mean(task_rows, "overall_score"),
        }
    summary = {
        "benchmark": args.benchmark_label,
        "protocol": "official VIEScore tie: SC + PQ",
        "image_source": str(args.image_root),
        "judge_model": args.model,
        "api_base": args.api_base,
        "thinking": False,
        "manifest_count": len(manifest),
        "record_count": len(ordered),
        "scored_count": len(scored),
        "failed_or_unparsed_count": len(ordered) - len(scored),
        "not_processed_count": len(manifest) - len(ordered),
        "semantic_score": mean(scored, "semantic_score"),
        "quality_score": mean(scored, "quality_score"),
        "overall_score": mean(scored, "overall_score"),
        "single_score_recovery_count": sum(
            row.get("sc_parse_mode", "").endswith("scalar_recovered")
            or row.get("pq_parse_mode", "").endswith("scalar_recovered")
            for row in scored
        ),
        "task_summary": task_summary,
        "prompt_sha256": {
            "sc": hashlib.sha256(sc_prompt.encode()).hexdigest(),
            "pq": hashlib.sha256(pq_prompt.encode()).hexdigest(),
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "summary_by_task.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task_type", "count", "semantic_score", "quality_score", "overall_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task, row in task_summary.items():
            writer.writerow({"task_type": task, **row})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gedit-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-judge")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--image-source-label",
        default="Shiyu95/gedit_results:step1x-edit-v1p1/fullset/en",
    )
    parser.add_argument("--benchmark-label", default="GEdit-Bench-EN-official-images")
    args = parser.parse_args()
    for name in ("manifest", "image_root", "output_root", "gedit_repo"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.manifest)
    sc_prompt, pq_prompt, score_parser = protocol(args.gedit_repo)
    (args.output_root / "run_config.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "image_root": str(args.image_root),
                "gedit_repo": str(args.gedit_repo),
                "api_base": args.api_base,
                "model": args.model,
                "thinking": False,
                "max_workers": args.max_workers,
                "rounds": args.rounds,
                "dimension_checkpointing": True,
                "image_source_label": args.image_source_label,
                "benchmark_label": args.benchmark_label,
                "sc_prompt": sc_prompt,
                "pq_prompt": pq_prompt,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    progress_path = args.output_root / "progress.jsonl"
    old_rows = read_jsonl(progress_path)
    rows_by_id = {str(row["record_id"]): row for row in old_rows}
    terminal_statuses = {"scored", "missing_official_image", "missing_source_image"}
    terminal_ids = {
        record_id
        for record_id, row in rows_by_id.items()
        if row.get("status") in terminal_statuses
    }
    pending = [row for row in manifest if str(row["record_id"]) not in terminal_ids]
    progress_lock = threading.Lock()

    def persist(row: dict[str, Any]) -> None:
        with progress_lock:
            rows_by_id[str(row["record_id"])] = row
            ordered = [
                rows_by_id[str(item["record_id"])]
                for item in manifest
                if str(item["record_id"]) in rows_by_id
            ]
            write_jsonl_atomic(progress_path, ordered)

    print(
        f"[official-gedit-qwen35] manifest={len(manifest)} pending={len(pending)} "
        f"workers={args.max_workers} dimension_checkpointing=true source={args.image_root}",
        flush=True,
    )
    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    evaluate_one,
                    row,
                    args.image_root,
                    args.api_base,
                    args.model,
                    sc_prompt,
                    pq_prompt,
                    score_parser,
                    args.timeout,
                    args.rounds,
                    args.image_source_label,
                    rows_by_id.get(str(row["record_id"])),
                    persist,
                ): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                persist(row)
                print(
                    f"[official-gedit-qwen35] {index}/{len(pending)} "
                    f"{row['record_id']} {row['status']}",
                    flush=True,
                )
    aggregate(manifest, list(rows_by_id.values()), args, sc_prompt, pq_prompt)


if __name__ == "__main__":
    main()
