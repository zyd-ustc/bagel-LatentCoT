#!/usr/bin/env python3
"""Shared Qwen3.5 OpenAI-compatible VLM judge client.

All new benchmark scorers call this single client so that retry policy, image
encoding, thinking-mode handling and resume semantics stay consistent with the
existing WISE/GEdit evaluators.

Conventions locked by BENCHMARK_EXTENSION_HANDOFF_20260825.md section 6:
  - temperature = 0.0
  - chat_template_kwargs.enable_thinking = false
  - images as base64 data URLs
  - preserve the judge raw output
  - HTTP/format failures are retried, never silently mapped to a model 0
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageOps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def image_data_url(path: Path | str, max_side: int = 768) -> str:
    """Encode an image as a PNG base64 data URL, EXIF-corrected and resized.

    The GEdit evaluator resizes to a ~512x512 area to keep token cost bounded;
    we keep the same idea but allow a configurable max side so large edits are
    not destroyed.  Aspect ratio is preserved.
    """
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    ratio = image.width / image.height
    width = max(1, int(math.sqrt(max_side * max_side * ratio)))
    height = max(1, int(width / ratio))
    if image.width > max_side or image.height > max_side:
        image = image.resize((width, height))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def clean_think(content: str) -> str:
    """Strip model thinking blocks and trailing artifacts."""
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    pattern = re.escape(open_tag) + r".*?" + re.escape(close_tag) + r"\s*"
    content = re.sub(pattern, "", content, flags=re.DOTALL)
    content = re.sub(re.escape(close_tag) + r"\s*", "", content)
    return content.strip()


def call_judge(
    api_base: str,
    model: str,
    system_prompt: str,
    user_content: list[dict[str, Any]] | str,
    timeout: float = 600.0,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send one multimodal chat request and return the cleaned text response.

    ``user_content`` may be a plain string (text-only) or the OpenAI multimodal
    content list (text + image_url parts).
    """
    if isinstance(user_content, str):
        user_content = [{"type": "text", "text": user_content}]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return clean_think(str(content))


def call_judge_yesno(
    api_base: str,
    model: str,
    system_prompt: str,
    question: str,
    image_urls: list[str],
    timeout: float = 600.0,
    max_tokens: int = 64,
) -> tuple[str, int | None]:
    """Ask a yes/no visual question. Returns (raw_text, 1|0|None).

    Following T2I-CoReBench's parser: yes => 1, no => 0, neither => None.
    ``yes`` takes precedence when both words appear.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    raw = call_judge(api_base, model, system_prompt, content, timeout=timeout, max_tokens=max_tokens)
    if re.search(r"\byes\b", raw, re.IGNORECASE):
        return raw, 1
    if re.search(r"\bno\b", raw, re.IGNORECASE):
        return raw, 0
    return raw, None


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_threaded(
    items: list[Any],
    worker: Callable[[Any], Any],
    max_workers: int,
    label: str = "judge",
    log_every: int = 20,
) -> list[Any]:
    """Run ``worker(item)`` over items with a thread pool, preserving no order.

    Returns the list of worker return values (order is completion order).
    """
    results: list[Any] = []
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % log_every == 0 or index == len(futures):
                print(f"[{label}] completed={index}/{len(futures)}", flush=True)
    return results


class ProgressStore:
    """Thread-safe resumable progress store keyed by record_id.

    Each scorer persists per-sample results to a progress.jsonl so that judge
    HTTP/format failures can be retried in a later pass without redoing
    successful work.  Terminal statuses are never overwritten.
    """

    def __init__(self, path: Path, terminal_statuses: set[str]):
        self.path = path
        self.lock = threading.Lock()
        self.terminal = terminal_statuses
        self.rows: dict[str, dict[str, Any]] = {
            str(row["record_id"]): row for row in read_jsonl(path)
        }

    def get(self, record_id: str) -> dict[str, Any] | None:
        return self.rows.get(record_id)

    def is_terminal(self, record_id: str) -> bool:
        row = self.rows.get(record_id)
        return row is not None and row.get("status") in self.terminal

    def update(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.rows[str(row["record_id"])] = row
            ordered = list(self.rows.values())
            write_jsonl_atomic(self.path, ordered)

    def all_rows(self) -> list[dict[str, Any]]:
        return list(self.rows.values())
