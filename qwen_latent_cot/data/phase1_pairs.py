"""Validation and loading for Phase-1 paired image-edit supervision."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


STAGE_A_EDIT_TYPES = frozenset(
    {
        "count",
        "relation",
        "spatial_relation",
        "move",
        "addition",
        "add",
        "deletion",
        "delete",
        "attribute",
        "attribute_binding",
        "simple_attribute_binding",
        "color",
        "noop",
        "no-op",
    }
)


def _resolve_image(value: Any, root: Path, *, location: str, field: str) -> str:
    if not str(value or "").strip():
        raise ValueError(f"{location}: {field} must be non-empty")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{location}: {field} not found: {path}")
    return str(path.resolve())


def _edit_types(value: Any, *, location: str) -> list[str]:
    values = value if isinstance(value, list) else [value]
    normalized = [str(item).strip().lower().replace(" ", "_") for item in values]
    normalized = [item for item in normalized if item]
    if not normalized:
        raise ValueError(f"{location}: edit_type must be non-empty")
    return normalized


def validate_phase1_pair(
    row: Mapping[str, Any],
    *,
    source_root: Path,
    location: str,
) -> dict[str, Any]:
    """Validate the new `(source, instruction, target)` contract.

    Reflection and teacher-score fields are deliberately ignored.  A no-op
    may omit ``target_image``; it is canonically mapped to its source image.
    """

    missing = [key for key in ("id", "source_image", "instruction", "is_noop") if key not in row]
    if missing:
        raise ValueError(f"{location}: missing fields {missing}")
    result = dict(row)
    if not str(result["id"]).strip():
        raise ValueError(f"{location}: id must be non-empty")
    if not isinstance(result["is_noop"], bool):
        raise ValueError(f"{location}: is_noop must be a JSON boolean")
    if not str(result["instruction"]).strip():
        raise ValueError(f"{location}: instruction must be non-empty")
    result["source_image"] = _resolve_image(
        result["source_image"], source_root, location=location, field="source_image"
    )
    target_value = result.get("target_image")
    if result["is_noop"] and not str(target_value or "").strip():
        result["target_image"] = result["source_image"]
    else:
        result["target_image"] = _resolve_image(
            target_value, source_root, location=location, field="target_image"
        )
    if "edit_type" not in result:
        if result["is_noop"]:
            result["edit_type"] = ["noop"]
        else:
            raise ValueError(f"{location}: edit_type must be present for edits")
    result["edit_type"] = _edit_types(result["edit_type"], location=location)
    return result


def load_phase1_pairs(
    path: str,
    *,
    allowed_edit_types: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser().resolve()
    allowed = (
        None
        if allowed_edit_types is None
        else {str(value).strip().lower().replace(" ", "_") for value in allowed_edit_types}
    )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        data_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = validate_phase1_pair(
            json.loads(line),
            source_root=data_path.parent,
            location=f"{data_path}:{line_number}",
        )
        if allowed is not None and not record["is_noop"]:
            # Stage A is exclusionary: a mixed relation+text or
            # addition+replacement pair belongs to a later stage.
            if not set(record["edit_type"]).issubset(allowed):
                continue
        records.append(record)
    if not records:
        raise ValueError(f"no matching Phase-1 pairs found in {data_path}")
    return records


def build_phase1_sampling_order(
    records: Iterable[Mapping[str, Any]],
    *,
    noop_fraction: float = 0.20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build one deterministic epoch with the requested no-op fraction."""

    fraction = float(noop_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("noop_fraction must lie in [0, 1)")
    rows = [dict(record) for record in records]
    edits = [record for record in rows if not bool(record["is_noop"])]
    noops = [record for record in rows if bool(record["is_noop"])]
    rng = random.Random(int(seed))
    rng.shuffle(edits)
    rng.shuffle(noops)
    if not edits:
        if fraction == 0.0:
            raise ValueError("no edit records available for noop_fraction=0")
        result = noops
    elif fraction == 0.0 or not noops:
        result = edits
    else:
        wanted = max(1, int(round(len(edits) * fraction / (1.0 - fraction))))
        selected_noops = [noops[index % len(noops)] for index in range(wanted)]
        result = edits + selected_noops
    rng.shuffle(result)
    if not result:
        raise ValueError("sampling order is empty")
    return result


__all__ = [
    "STAGE_A_EDIT_TYPES",
    "build_phase1_sampling_order",
    "load_phase1_pairs",
    "validate_phase1_pair",
]
