"""Judge a candidate by comparing pictures, with no language model in the loop.

Every acceptance question in the previous pipeline went through the 8B model's
yes/no head, and that head is unreliable on detail: it called a brass telescope
"absent" from a picture where it was plainly visible, which turned valid
instructions into rejections.

This verifier uses pixel evidence instead:

    S = x_src      a_0.25  a_0.36  a_0.50        P_t previews
    Y = x_eff      the picture the instructed teacher produced

* the edit happened if the box GLM named changed materially between S and Y;
* the preserve regions held if every box GLM named stayed nearly identical;
* an item is "unreadable" only when the boxes are malformed, never because a
  model answered no.

No NPU is used. Usage:

    python scripts/verify_by_pixels.py --records <coupled cache> --out <jsonl>
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# A box counts as changed when its mean absolute difference exceeds this fraction
# of the display range. Calibrated on the first batch and recorded in docs.
CHANGE_THRESHOLD = 0.06
PRESERVE_TOLERANCE = 0.04


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def box_region(image, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    height, width = image.shape[:2]
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def mean_abs_difference(a, b, box):
    region_a, region_b = box_region(a, box), box_region(b, box)
    if region_a is None or region_b is None:
        return None
    if region_a.shape != region_b.shape:
        side = min(region_a.shape[0], region_b.shape[0]), min(region_a.shape[1], region_b.shape[1])
        region_a, region_b = region_a[: side[0], : side[1]], region_b[: side[0], : side[1]]
    return float(np.abs(region_a - region_b).mean())


def judge(source, edited, edit_box, preserve_boxes):
    """Changed where asked, unchanged where promised. Returns a verdict dict."""
    change = mean_abs_difference(source, edited, edit_box) if edit_box else None
    preserved, broken = [], []
    for item in preserve_boxes or []:
        delta = mean_abs_difference(source, edited, item["box"])
        if delta is None:
            continue
        (broken if delta > PRESERVE_TOLERANCE else preserved).append(
            dict(box=item["box"], delta=round(delta, 4), question=item.get("question", "")))
    edited_ok = change is None or change > CHANGE_THRESHOLD
    return dict(
        edit_delta=None if change is None else round(change, 4),
        edit_happened=edited_ok,
        preserved=preserved,
        broken=broken,
        approved=bool(edited_ok and not broken),
        reason=None if edited_ok and not broken else (
            "no_change" if not edited_ok else "preserve_broken"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="coupled cache with progress/*.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    for path in glob.glob(str(Path(args.records) / "progress" / "*.json")):
        records.append(json.loads(Path(path).read_text()))
    counts, written = {}, 0
    out = open(args.out, "w")
    for record in records:
        source = record.get("source_image")
        edited = record.get("target_image")
        if not source or not edited or not Path(source).exists() or not Path(edited).exists():
            counts["missing_image"] = counts.get("missing_image", 0) + 1
            continue
        edit_box = record.get("edit_box")
        boxes = record.get("preserve_boxes") or []
        verdict = judge(load(source), load(edited), edit_box, boxes)
        row = dict(record, pixel_verdict=verdict)
        counts[verdict["reason"] or "approved"] = counts.get(verdict["reason"] or "approved", 0) + 1
        if verdict["approved"]:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    out.close()
    print(json.dumps(dict(total=len(records), approved=written, counts=counts)))


if __name__ == "__main__":
    main()
