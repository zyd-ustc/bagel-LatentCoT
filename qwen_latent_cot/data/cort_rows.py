"""Canonical CoRT row loading for the v4 multi-turn consistency dataset.

A v4 source row (``cort_v4_{train,val,test}.jsonl``) looks like::

    {
      "sample_id": "...", "source": "...", "prompt": "...",
      "num_turns": K,
      "image_paths": [img0, img1, ..., imgK],        # len == K + 1
      "steps": [ {turn, analysis, fix, output_image, label_round}, ... ],
      "chain_status": "terminal" | "open_max_round",
      ...
    }

Semantics (verified against the v4 export):

* ``image_paths[i]`` is the i-th visual state (img0 is the initial generation,
  imgN are the edited results).
* ``steps[i]`` is the review of ``image_paths[i]``.
* terminal chains have ``len(steps) == num_turns + 1``; the last step is the
  terminal review (``fix`` is null) whose branch is ``<|cort_end|>``.
* open_max_round chains have ``len(steps) == num_turns``; every step carries a
  ``fix`` and continues. The final image is left unreviewed (truncated).

This module normalises rows into a canonical schema. Historical reflection
text uses the v3 token protocol natively, so no tag rewriting happens at
ingest::

    <|cort_boa|>{analysis}<|cort_eoa|>
    <|cort_bof|>{fix}<|cort_eof|>        # only for continuing steps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_latent_cot.constants import SPECIAL_TOKENS

_BOA = SPECIAL_TOKENS["boa"]
_EOA = SPECIAL_TOKENS["eoa"]
_BOF = SPECIAL_TOKENS["bof"]
_EOF = SPECIAL_TOKENS["eof"]

# Some v4 terminal analyses leaked the generator's stop marker (e.g. a literal
# ``<|cort_end|>``) into the prose. Structural branch tokens are emitted by the
# sample builder, so any control-token substring in free text must be stripped.
_CONTROL_TOKEN_STRINGS = tuple(SPECIAL_TOKENS.values())
def _strip_control_tokens(text: str) -> str:
    for tok in _CONTROL_TOKEN_STRINGS:
        if tok in text:
            text = text.replace(tok, " ")
    return " ".join(text.split())


def wrap_reflection(analysis: str, fix: str | None) -> str:
    """Render one review as ``boa/eoa`` analysis + optional ``bof/eof`` fix.

    No separator between segments: the special tokens are self-delimiting, so
    every collator can scan spans without handling a gap token.
    """
    analysis = str(analysis or "").strip()
    body = f"{_BOA}{analysis}{_EOA}"
    fix = str(fix or "").strip()
    if fix:
        body = f"{body}{_BOF}{fix}{_EOF}"
    return body


def _build_v4_row(
    raw: dict[str, Any],
    *,
    require_analysis: bool = True,
) -> dict | None:
    prompt = str(raw.get("prompt", "") or "").strip()
    source = str(raw.get("source", "") or "").strip()
    sample_id = str(raw.get("sample_id", "") or "").strip()
    num_turns = raw.get("num_turns")
    steps = raw.get("steps")
    image_paths = raw.get("image_paths")
    image_captions = raw.get("image_captions", raw.get("captions"))

    if not prompt or not source or not sample_id:
        return None
    if not isinstance(num_turns, int) or num_turns < 0:
        return None
    if not isinstance(steps, list) or not steps:
        return None
    if not isinstance(image_paths, list) or len(image_paths) != num_turns + 1:
        return None
    if any(not p for p in image_paths):
        return None
    if not isinstance(image_captions, list) or len(image_captions) != len(image_paths):
        image_captions = []

    is_terminal_chain = str(raw.get("chain_status", "")) == "terminal"
    # terminal: len(steps) == num_turns + 1 (last is the stop review)
    # open:     len(steps) == num_turns
    expected_steps = num_turns + 1 if is_terminal_chain else num_turns
    if len(steps) != expected_steps:
        return None

    reviews: list[dict] = []
    for idx, step in enumerate(steps):
        analysis = _strip_control_tokens(str(step.get("analysis", "") or ""))
        if not analysis and bool(require_analysis):
            return None
        is_terminal_review = is_terminal_chain and idx == len(steps) - 1
        fix = None if is_terminal_review else step.get("fix")
        fix = _strip_control_tokens(str(fix)) if fix else None
        reviews.append(
            {
                "analysis": analysis,
                "fix": (fix or None),
                "is_terminal": is_terminal_review,
            }
        )

    return {
        "sample_id": sample_id,
        "source": source,
        "prompt": prompt,
        "generator_model": raw.get("generator_model"),
        "num_turns": num_turns,
        "images": [str(p) for p in image_paths],
        "image_captions": [str(value or "").strip() for value in image_captions],
        "reviews": reviews,
        "is_terminal_chain": is_terminal_chain,
        "storage": "v4",
        "trajectory_id": sample_id,
    }


def _iter_jsonl_files(raw_path: str) -> list[Path]:
    path_obj = Path(raw_path)
    if path_obj.is_file() and path_obj.suffix == ".jsonl":
        return [path_obj]
    if path_obj.is_dir():
        return sorted(p for p in path_obj.iterdir() if p.is_file() and p.suffix == ".jsonl")
    raise ValueError(f"Unsupported data path (expected a .jsonl file or dir of them): {raw_path}")


def load_canonical_cort_rows(
    data_paths: list[str],
    *,
    require_analysis: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for raw_path in data_paths:
        for jsonl_file in _iter_jsonl_files(raw_path):
            with jsonl_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = _build_v4_row(
                        json.loads(line),
                        require_analysis=bool(require_analysis),
                    )
                    if row is None or row["sample_id"] in seen:
                        continue
                    rows.append(row)
                    seen.add(row["sample_id"])
    return rows


__all__ = ["load_canonical_cort_rows", "wrap_reflection"]
