#!/usr/bin/env python3
"""Evaluate GenEval++ pipeline images with a local OpenAI-compatible Qwen judge."""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


SYSTEM_PROMPT = """You are an expert image evaluator. Determine whether the image fully satisfies the instruction and every item in the expectation checklist. Check object classes, exact counts, colors, spatial positions, and relative sizes. The result is correct only if every requirement is satisfied. The image must be one coherent photo-like scene; reject cartoons, sketches, anime, collages, and multi-panel images. Be strict and conservative. Return JSON only: {\"correct\": 0 or 1, \"reason\": \"concise evidence\"}. Do not output thinking or markdown."""

SYSTEM_PROMPTS = {
    "geneval_pp": SYSTEM_PROMPT,
    "t2i_compbench_pp": """You are an expert factual image evaluator. Determine whether the image satisfies every object, attribute, exact count, action, and spatial or non-spatial relationship stated in the instruction. Judge only visible factual compliance. Do not reject an image for aesthetics, realism, artistic style, composition, or quality unless the instruction explicitly requires it. Mark the result correct only when every stated constraint is satisfied. Return JSON only: {\"correct\": 0 or 1, \"reason\": \"concise visual evidence\"}. Do not output thinking or markdown.""",
}

NON_OFFICIAL_REASONS = {
    "geneval_pp": "The official GPT-4.1 judge is replaced by a local Qwen judge.",
    "t2i_compbench_pp": (
        "The official ten images per prompt and category-specific BLIP-VQA, UniDet, "
        "CLIPScore, and 3-in-1 metrics are replaced by one image per prompt and a "
        "local Qwen all-constraints judge."
    ),
}

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "round1", "round2", "final"), default="initial")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8297/v1")
    parser.add_argument("--model", default="qwen3.5-397b-genevalpp")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--benchmark-profile",
        choices=tuple(SYSTEM_PROMPTS),
        default="geneval_pp",
    )
    return parser.parse_args()


def metadata_to_checklist(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in metadata.get("include", []):
        count = int(item.get("count", 1))
        noun = str(item["class"])
        if count != 1:
            noun += "s"
        descriptions = [str(item[key]) for key in ("color", "size") if item.get(key)]
        phrase = " ".join(descriptions + [str(count), noun])
        if item.get("region"):
            phrase += f" located in the {item['region']} part of the image"
        else:
            phrase += " present in the image"
        parts.append(f"- {phrase}.")
    for item in metadata.get("exclude", []):
        count = int(item.get("count", 1))
        noun = str(item["class"])
        if count != 1:
            noun += "s"
        parts.append(f"- No more than {count - 1} {noun} should appear.")
    if not parts:
        return (
            "Treat every visible factual constraint in the instruction as required. "
            "Do not invent any requirement that is absent from the instruction."
        )
    return "This image should contain:\n" + "\n".join(parts)


def extract_result(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("no_json_object")
    value = json.loads(match.group(0))
    if value.get("correct") not in (0, 1, False, True):
        raise ValueError("invalid_correct")
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("invalid_reason")
    return {"correct": int(bool(value["correct"])), "reason": reason.strip()}


def resolve_image(pipeline_root: Path, dense_index: int, stage: str) -> Path:
    case_dir = pipeline_root / "cases" / f"case-{dense_index:03d}"
    initial = case_dir / "initial.png"
    if stage == "initial":
        return initial
    round1 = case_dir / "rectified_round_01.png"
    round2 = case_dir / "rectified_round_02.png"
    if stage == "round1":
        return round1 if round1.is_file() else initial
    if stage == "round2":
        if round2.is_file():
            return round2
        return round1 if round1.is_file() else initial
    report_path = case_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        final = Path(str(report.get("final_image", "")))
        if final.is_file():
            return final
    if round2.is_file():
        return round2
    return round1 if round1.is_file() else initial


def encode_jpeg(path: Path) -> str:
    import io

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_one(
    row: dict[str, Any], args: argparse.Namespace, api_url: str
) -> dict[str, Any]:
    dense_index = int(row["index"])
    image_path = resolve_image(args.pipeline_root, dense_index, args.stage)
    base = {
        "index": dense_index,
        "official_index": int(row["official_index"]),
        "tag": str(row["category"]),
        "prompt": str(row["required_prompt"]),
        "image": str(image_path.resolve()),
        "stage": args.stage,
    }
    if not image_path.is_file():
        return {**base, "status": "missing_image", "correct": 0, "reason": "Image not found"}
    metadata = dict(row["benchmark_metadata"])
    user_text = (
        f"Instruction:\n{row['required_prompt']}\n\n"
        f"Expectation checklist:\n{metadata_to_checklist(metadata)}"
    )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[args.benchmark_profile]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_jpeg(image_path)}"},
                    },
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"json": RESULT_SCHEMA},
    }
    last_error = ""
    for attempt in range(args.retries + 1):
        started = time.monotonic()
        try:
            response = post_json(api_url, payload, args.timeout)
            text = response["choices"][0]["message"]["content"]
            parsed = extract_result(text)
            usage = response.get("usage", {})
            return {
                **base,
                "status": "ok",
                **parsed,
                "raw_output": text,
                "attempt": attempt + 1,
                "seconds": round(time.monotonic() - started, 4),
                "usage": usage,
            }
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 4))
    return {**base, "status": "error", "correct": 0, "reason": last_error}


def completed(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[int(row["index"])] = row
    return result


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.pipeline_root = args.pipeline_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = list(manifest["samples"])
    if args.max_items:
        rows = rows[: args.max_items]
    results_path = args.output_dir / "results.jsonl"
    known = completed(results_path)
    pending = [row for row in rows if int(row["index"]) not in known]
    api_url = args.base_url.rstrip("/") + "/chat/completions"
    started = time.monotonic()
    lock = threading.Lock()
    with results_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_one, row, args, api_url): row for row in pending}
            for completed_count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "completed": completed_count,
                            "scheduled": len(pending),
                            "index": result["index"],
                            "status": result["status"],
                            "correct": result["correct"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    all_results = completed(results_path)
    selected = [all_results[int(row["index"])] for row in rows if int(row["index"]) in all_results]
    by_tag: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        tag = str(row["tag"])
        by_tag[tag]["total"] += 1
        by_tag[tag]["correct"] += int(row["correct"] == 1)
        by_tag[tag]["valid"] += int(row["status"] == "ok")
    tag_metrics = {
        tag: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            "valid": counts["valid"],
        }
        for tag, counts in sorted(by_tag.items())
    }
    elapsed = time.monotonic() - started
    total = len(selected)
    mean_tag_accuracy = (
        sum(item["accuracy"] for item in tag_metrics.values()) / len(tag_metrics)
        if tag_metrics
        else 0.0
    )
    summary = {
        "protocol": f"{args.benchmark_profile}_qwen_local_v1",
        "official_comparable": False,
        "reason_not_official": NON_OFFICIAL_REASONS[args.benchmark_profile],
        "manifest": str(args.manifest),
        "pipeline_root": str(args.pipeline_root),
        "stage": args.stage,
        "judge_model": args.model,
        "benchmark_profile": args.benchmark_profile,
        "base_url": args.base_url,
        "cases": total,
        "correct": sum(int(row["correct"] == 1) for row in selected),
        "accuracy": (sum(int(row["correct"] == 1) for row in selected) / total if total else 0.0),
        "mean_tag_accuracy": mean_tag_accuracy,
        "valid": sum(int(row["status"] == "ok") for row in selected),
        "by_tag": tag_metrics,
        "newly_scheduled": len(pending),
        "inference_wall_seconds": round(elapsed, 3),
        "samples_per_second": round(len(pending) / elapsed, 5) if elapsed else 0.0,
        "workers": args.workers,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
