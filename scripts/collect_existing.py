"""Collect existing coupled pairs into one review table.

Two caches hold every pair produced so far (cpl_a 778, cpl_all_b 534). Each
record already carries its instruction, target scene, edit box and preserve
boxes, so the table needs nothing from the plan files.

Usage::

    python3 scripts/collect_existing.py --out plan/exist_req.jsonl
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caches", nargs="+", default=["cpl_a", "cpl_all_b"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, seen = [], set()
    for root in args.caches:
        for path in glob.glob(os.path.join(root, "progress", "*.json")):
            try:
                record = json.load(open(path))
            except Exception:
                continue
            uid = record.get("uid")
            if not uid or uid in seen:
                continue
            if not record.get("source_image") or not record.get("target_image"):
                continue
            seen.add(uid)
            rows.append(dict(
                uid=uid,
                edit_instruction=record.get("edit_instruction") or "",
                tail_condition=record.get("tail_condition") or "",
                parent_condition=record.get("parent_condition") or "",
                edit_box=record.get("edit_box"),
                preserve_boxes=record.get("preserve_boxes") or [],
                family=record.get("diagnosis_status") or "existing",
                operation=None,
                teacher=record.get("teacher"),
                source_image=record.get("source_image"),
                target_image=record.get("target_image"),
            ))

    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(rows=len(rows),
                          with_instruction=sum(1 for r in rows if r["edit_instruction"].strip()),
                          with_boxes=sum(1 for r in rows if r["edit_box"]),
                          with_preserve=sum(1 for r in rows if r["preserve_boxes"]))))


if __name__ == "__main__":
    main()
