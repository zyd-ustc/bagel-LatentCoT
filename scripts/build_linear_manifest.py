"""Build the L manifest from qualified edit pairs plus source captions."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def _read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _full_tasks(path):
    """Only the join fields are kept; the full task file is large."""
    rows = {}
    for row in _read_jsonl(path):
        rows[row["uid"]] = {
            "source": row.get("source"),
            "src_uid": row.get("src_uid"),
        }
    return rows


def _captions(path):
    if not path:
        return {}
    out = {}
    for row in _read_jsonl(path):
        uid = row.get("uid")
        text = row.get("parent_condition") or row.get("caption") or row.get("caption_en")
        if uid and text and str(text).strip():
            out[str(uid)] = " ".join(str(text).split())
    return out


def _x2edit_caption(source_image):
    """X2Edit stores caption_en next to the source jpg."""
    path = Path(source_image)
    stem = path.name.split(".")[0]
    sidecar = path.parent / (stem + ".json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text())
    except Exception:
        return None
    text = data.get("caption_en") or data.get("caption_ori")
    return " ".join(str(text).split()) if text else None


def _tail(row, args):
    """Return (text, source) or (None, None) when no full tail was declared."""
    if args.tail_field == "prompt" and str(row.get("_tail_prompt") or "").strip():
        return " ".join(str(row["_tail_prompt"]).split()), "pe_rewritten_instruction"
    if args.tail_field == "tail_condition" and str(row.get("tail_condition") or "").strip():
        return " ".join(str(row["tail_condition"]).split()), "declared"
    if args.allow_derived_tail:
        return (
            row["parent_condition"] + "\nApply this edit to the scene: " + row["edit_instruction"],
            "derived_parent_plus_edit",
        )
    return None, None


def _split(family, val_fraction, test_fraction):
    digest = hashlib.sha1(str(family).encode()).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < val_fraction:
        return "val"
    if value < val_fraction + test_fraction:
        return "test"
    return "train"


def _geometry_error(row):
    """Why this pair cannot share one scene extent, or ``None`` when it can."""
    from PIL import Image

    from qwen_latent_cot.u15_target_correction.images import pair_geometry_error

    try:
        with Image.open(row["source_image"]) as source, Image.open(row["target_image"]) as target:
            return pair_geometry_error(row, source.size, target.size)
    except (OSError, ValueError) as error:
        return "unreadable: " + str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified", required=True)
    parser.add_argument("--full-tasks", required=True)
    parser.add_argument("--out")
    parser.add_argument("--selection-out", help="Write the selected caption candidates and stop")
    parser.add_argument("--captions", help="JSONL with uid and caption/parent_condition")
    parser.add_argument(
        "--tail-field", choices=("none", "prompt", "tail_condition"), default="none",
        help="Explicit source of the full target condition; none means no L records",
    )
    parser.add_argument("--allow-derived-tail", action="store_true")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--per-task", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--allow-missing-parent", action="store_true")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument(
        "--drop-unmappable-geometry",
        action="store_true",
        help="Drop pairs whose fields of view differ without declared geometry crops",
    )
    parser.add_argument("--max-scan", type=int, default=0)
    args = parser.parse_args()
    if not args.out and not args.selection_out:
        parser.error("--out or --selection-out is required")

    full = _full_tasks(args.full_tasks)
    captions = _captions(args.captions)
    source_filter = {str(value) for value in args.source}
    task_filter = {str(value) for value in args.task}
    rng = random.Random(args.seed)
    seen, selected = {}, {}

    for index, row in enumerate(_read_jsonl(args.qualified)):
        if args.max_scan and index >= args.max_scan:
            break
        uid = str(row.get("uid") or "")
        task = str(row.get("task") or "")
        if not uid or not task:
            continue
        if task_filter and task not in task_filter:
            continue
        joined = full.get(uid, {})
        source = joined.get("source")
        if source_filter and source not in source_filter:
            continue
        source_image = row.get("src") or row.get("source_image")
        target_image = row.get("target")
        edit = row.get("prompt_draft") or row.get("prompt")
        if not source_image or not target_image or not str(edit or "").strip():
            continue
        edit = " ".join(str(edit).split())
        family = joined.get("src_uid") or hashlib.sha1(str(source_image).encode()).hexdigest()[:20]
        record = dict(
            uid=uid,
            task=task,
            source=source or "unknown",
            source_family=str(family),
            prompt_family=str(family),
            source_image=str(source_image),
            target_image=str(target_image),
            original_source_image=row.get("source_image"),
            edit_instruction=edit,
            parent_condition="",
            tail_condition="",
            tail_source="",
            preserve="",
            semantic_approved=True,
            target_quality_source="qualified_nonzero",
            condition_protocol="isochronic_condition_v1",
            _tail_prompt=row.get("prompt"),
        )
        bucket = selected.setdefault(task, [])
        seen[task] = seen.get(task, 0) + 1
        if len(bucket) < args.per_task:
            bucket.append(record)
        else:
            replace = rng.randrange(seen[task])
            if replace < args.per_task:
                bucket[replace] = record

    rows = [row for bucket in selected.values() for row in bucket]
    if args.limit and len(rows) > args.limit:
        rows = rng.sample(rows, args.limit)

    if args.selection_out:
        selection = Path(args.selection_out)
        selection.parent.mkdir(parents=True, exist_ok=True)
        with selection.open("w") as handle:
            for row in sorted(rows, key=lambda item: item["uid"]):
                handle.write(
                    json.dumps(
                        dict(
                            uid=row["uid"],
                            task=row["task"],
                            source=row["source"],
                            source_family=row["source_family"],
                            source_image=row["source_image"],
                            target_image=row["target_image"],
                            edit_instruction=row["edit_instruction"],
                            prompt=row["_tail_prompt"],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(
            json.dumps(
                dict(records=len(rows), path=str(selection), mode="selection"),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    resolved = []
    geometry_dropped, geometry_examples = 0, []
    for row in rows:
        if args.skip_missing_images and (
            not os.path.exists(row["source_image"]) or not os.path.exists(row["target_image"])
        ):
            continue
        if args.drop_unmappable_geometry:
            error = _geometry_error(row)
            if error:
                geometry_dropped += 1
                if len(geometry_examples) < 5:
                    geometry_examples.append(dict(uid=row["uid"], error=error))
                continue
        parent = captions.get(row["uid"])
        if not parent and not args.captions and row["source"] == "X2Edit-Dataset":
            parent = _x2edit_caption(row.get("original_source_image") or row["source_image"])
        if not parent:
            if not args.allow_missing_parent:
                continue
            parent = row["edit_instruction"]
        parent = " ".join(str(parent).split())
        row["parent_condition"] = parent
        tail, tail_source = _tail(row, args)
        if tail is None:
            continue
        row["tail_condition"] = tail
        row["tail_source"] = tail_source
        row.pop("_tail_prompt", None)
        row["split"] = _split(row["source_family"], args.val_fraction, args.test_fraction)
        resolved.append(row)
    rows = resolved
    rows.sort(key=lambda row: row["uid"])

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": rows}, ensure_ascii=False) + "\n")
    summary = {
        "records": len(rows),
        "source": {key: sum(row["source"] == key for row in rows) for key in sorted({row["source"] for row in rows})},
        "task": {key: sum(row["task"] == key for row in rows) for key in sorted({row["task"] for row in rows})},
        "split": {key: sum(row["split"] == key for row in rows) for key in ("train", "val", "test")},
        "geometry_dropped": geometry_dropped,
        "geometry_examples": geometry_examples,
        "path": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
