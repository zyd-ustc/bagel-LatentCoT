"""Validation and loading for Phase-1 structured-reflection records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FIELDS = (
    "id",
    "source_image",
    "instruction",
    "reflection",
    "edit_type",
    "target_constraints",
    "preserve_constraints",
    "is_noop",
    "difficulty",
    "teacher_valid",
    "teacher_semantic_delta",
    "teacher_preserve_delta",
)


def _english_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _reflection_bullet_counts(reflection: str) -> tuple[int, int]:
    target = preserve = 0
    section = None
    for raw_line in reflection.splitlines():
        line = raw_line.strip()
        lowered = line.lower().rstrip(":")
        if lowered == "target changes":
            section = "target"
        elif lowered == "preserve":
            section = "preserve"
        elif line.startswith("-"):
            if section == "target":
                target += 1
            elif section == "preserve":
                preserve += 1
    return target, preserve


def validate_phase1_record(
    row: Mapping[str, Any],
    *,
    source_root: Path,
    location: str,
    teacher_min_semantic_delta: float = 0.0,
    teacher_preserve_epsilon: float = 0.02,
) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIELDS if key not in row]
    if missing:
        raise ValueError(f"{location}: missing fields {missing}")
    normalized = dict(row)
    for key in ("id", "source_image", "instruction", "reflection"):
        if not str(normalized[key]).strip():
            raise ValueError(f"{location}: {key} must be non-empty")
    reflection = str(normalized["reflection"])
    target_bullets, preserve_bullets = _reflection_bullet_counts(reflection)
    if "target changes:" not in reflection.lower() or "preserve:" not in reflection.lower():
        raise ValueError(
            f"{location}: reflection requires Target changes and Preserve sections"
        )
    first_line = next(
        (line.strip() for line in reflection.splitlines() if line.strip()), ""
    )
    if first_line.upper() != "EDIT PLAN":
        raise ValueError(f"{location}: reflection must start with EDIT PLAN")
    if not 1 <= target_bullets <= 4 or not 1 <= preserve_bullets <= 4:
        raise ValueError(
            f"{location}: reflection bullets must be 1-4 per section, got "
            f"target={target_bullets}, preserve={preserve_bullets}"
        )
    if _english_token_count(reflection) > 120:
        raise ValueError(f"{location}: reflection exceeds 120 English tokens")
    for key in ("edit_type", "target_constraints", "preserve_constraints"):
        if not isinstance(normalized[key], list):
            raise ValueError(f"{location}: {key} must be a JSON list")
    if not normalized["edit_type"]:
        raise ValueError(f"{location}: edit_type must be non-empty")
    if not normalized["preserve_constraints"]:
        raise ValueError(f"{location}: preserve_constraints must be non-empty")
    if not isinstance(normalized["is_noop"], bool):
        raise ValueError(f"{location}: is_noop must be a JSON boolean")
    if normalized["is_noop"]:
        noop_phrases = ("no structural change", "no change required")
        if not any(phrase in reflection.lower() for phrase in noop_phrases):
            raise ValueError(
                f"{location}: noop reflection must explicitly state "
                "No structural change or No change required"
            )
    elif not normalized["target_constraints"]:
        raise ValueError(
            f"{location}: non-noop target_constraints must be non-empty"
        )
    if not isinstance(normalized["difficulty"], (int, float)):
        raise ValueError(f"{location}: difficulty must be numeric")
    if not isinstance(normalized["teacher_valid"], bool):
        raise ValueError(f"{location}: teacher_valid must be a JSON boolean")
    for key in ("teacher_semantic_delta", "teacher_preserve_delta"):
        if not isinstance(normalized[key], (int, float)) or not math.isfinite(
            float(normalized[key])
        ):
            raise ValueError(f"{location}: {key} must be a finite number")
    semantic_delta = float(normalized["teacher_semantic_delta"])
    semantic_valid = (
        semantic_delta >= float(teacher_min_semantic_delta)
        if normalized["is_noop"]
        else semantic_delta > float(teacher_min_semantic_delta)
    )
    score_valid = semantic_valid and float(
        normalized["teacher_preserve_delta"]
    ) >= -float(teacher_preserve_epsilon)
    if normalized["teacher_valid"] and not score_valid:
        raise ValueError(
            f"{location}: teacher_valid conflicts with teacher score deltas"
        )
    source = Path(str(normalized["source_image"])).expanduser()
    if not source.is_absolute():
        source = source_root / source
    if not source.is_file():
        raise FileNotFoundError(f"{location}: source image not found: {source}")
    normalized["source_image"] = str(source.resolve())
    return normalized


def load_phase1_records(
    path: str,
    *,
    teacher_valid_only: bool = True,
    teacher_min_semantic_delta: float = 0.0,
    teacher_preserve_epsilon: float = 0.02,
) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser().resolve()
    rows = []
    for line_number, line in enumerate(
        data_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        normalized = validate_phase1_record(
            row,
            source_root=data_path.parent,
            location=f"{data_path}:{line_number}",
            teacher_min_semantic_delta=float(teacher_min_semantic_delta),
            teacher_preserve_epsilon=float(teacher_preserve_epsilon),
        )
        if not teacher_valid_only or normalized["teacher_valid"]:
            rows.append(normalized)
    if not rows:
        suffix = " teacher-valid" if teacher_valid_only else ""
        raise ValueError(f"no{suffix} Phase-1 records found in {data_path}")
    return rows


__all__ = ["REQUIRED_FIELDS", "load_phase1_records", "validate_phase1_record"]
