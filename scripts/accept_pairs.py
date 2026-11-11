"""Accept a coupled pair only when pixels and GLM both agree it is usable.

The previous acceptance path could not reject anything. On the first 187 pairs
every one came back

    GLM  EDIT_DONE yes / KEEP_OK yes / QUALITY_OK yes
    pixel  edit_incomplete

because ``glm_verify.py`` wrote only approved rows and never compared GLM
against the picture. A judge that never says no is not a judge.

This verifier keeps both signals and gives each one the job it can actually do:

    pixels   the requested box must have changed, the preserve boxes must not
             have. This is cheap, deterministic and unfakeable.
    GLM      the target state named in ``tail_condition`` has to be reached, and
             the picture has to be free of fusion and tearing. GLM reads the
             instruction's completion state, which pixels cannot see.
    fold     whether the picture is malformed is the one question eyeballs beat
             every model at, so those cases are listed for review, not guessed.

``EDIT_DONE: partial`` counts as a failure. The instruction either happened or
it did not; a half-done edit teaches the editor a half-done target.

No NPU is used. Usage::

    METTO_GLM_KEY=... python scripts/accept_pairs.py \\
        --records <coupled cache> --requests <plan.jsonl> \\
        --out-approved <jsonl> --out-rejected <jsonl> --workers 48
"""

import argparse
import base64
import glob
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ENDPOINT = "/v1/chat/completions"
MODEL = "glm-5.3-flash"

# A box counts as changed above this fraction of the display range. Same
# thresholds as verify_by_pixels.py so the two stages cannot disagree.
CHANGE_THRESHOLD = 0.06
PRESERVE_TOLERANCE = 0.04

SYSTEM = (
    "You compare two pictures of the same scene. Picture A is before an edit; "
    "picture B is after it. You are given the edit instruction and the exact "
    "scene the edit was supposed to produce.\n"
    "Answer with five lines and nothing else:\n"
    "EDIT_DONE: yes|partial|no -- does B show the instructed change, completely? "
    "yes only if every part of the instruction happened; partial if some of it did.\n"
    "TARGET_MET: yes|partial|no -- does B match the REQUIRED SCENE? Judge the "
    "scene, not the wording.\n"
    "KEEP_OK: yes|no -- is everything the instruction did not ask to change still "
    "as in A?\n"
    "QUALITY_OK: yes|no -- are both pictures free of fusion, smearing, torn edges, "
    "duplicated or melted objects? A defect already present in A alone does not count.\n"
    "NOTE: <one short sentence on what is still wrong, or 'clean'>"
)

_FIELD = re.compile(r"^([A-Z_]+)\s*:\s*(.+)$", re.M)


def parse(text):
    fields = {}
    for match in _FIELD.finditer(text or ""):
        fields[match.group(1).strip().upper()] = match.group(2).strip()
    return fields


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
    ra, rb = box_region(a, box), box_region(b, box)
    if ra is None or rb is None:
        return None
    if ra.shape != rb.shape:
        h = min(ra.shape[0], rb.shape[0])
        w = min(ra.shape[1], rb.shape[1])
        ra, rb = ra[:h, :w], rb[:h, :w]
    return float(np.abs(ra - rb).mean())


def pixel_gate(source, edited, edit_box, preserve_boxes):
    """The hard gate. Runs first: a pair that fails here never costs a GLM call."""
    change = mean_abs_difference(source, edited, edit_box) if edit_box else None
    broken = []
    for item in preserve_boxes or []:
        delta = mean_abs_difference(source, edited, item["box"])
        if delta is not None and delta > PRESERVE_TOLERANCE:
            broken.append(dict(box=item["box"], delta=round(delta, 4)))
    edited_ok = change is not None and change > CHANGE_THRESHOLD
    return dict(
        edit_delta=None if change is None else round(change, 4),
        preserved=len(preserve_boxes or []) - len(broken),
        broken=broken,
        passed=bool(edited_ok and not broken),
        reason=None if edited_ok and not broken else (
            "no_change" if not edited_ok else "preserve_broken"),
    )


def encode(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def judge(session, before, after, instruction, required_scene, keep_text, timeout=150):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "PICTURE A (before):"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + encode(before)}},
                {"type": "text", "text": "PICTURE B (after):"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + encode(after)}},
                {"type": "text", "text":
                    "INSTRUCTION:\n" + (instruction or "") +
                    "\n\nREQUIRED SCENE:\n" + (required_scene or "") +
                    ("\n\nMUST STAY THE SAME:\n" + keep_text if keep_text else "")},
            ]},
        ],
        "stream": False,
        "temperature": 0,
    }
    for attempt in range(3):
        try:
            response = session.post(os.environ["METTO_BASE_URL"].rstrip("/") + ENDPOINT,
                                    json=body, timeout=(15, timeout))
            response.raise_for_status()
            content = (response.json().get("choices") or [{}])[0].get("message", {}).get("content")
            if content:
                return content
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def decide(fields):
    """Approval needs an explicit yes on every axis. partial is not yes."""
    done = (fields.get("EDIT_DONE") or "").strip().lower()
    target = (fields.get("TARGET_MET") or "").strip().lower()
    keep = (fields.get("KEEP_OK") or "").strip().lower()
    quality = (fields.get("QUALITY_OK") or "").strip().lower()
    if not done or not quality:
        return False, "glm_unparsed"
    if quality != "yes":
        return False, "quality"
    if done != "yes":
        return False, "edit_" + (done or "missing")
    if target != "yes":
        return False, "target_" + (target or "missing")
    if keep != "yes":
        return False, "keep_broken"
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, nargs="+",
                    help="coupled caches holding progress/*.json with source/target images")
    ap.add_argument("--requests", required=True, nargs="+",
                    help="plan jsonl carrying instruction, tail_condition and preserve_boxes")
    ap.add_argument("--out-approved", required=True)
    ap.add_argument("--out-rejected", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--no-pixel-gate", action="store_true",
                    help="run GLM on everything; use only to measure the gate")
    args = ap.parse_args()

    wanted = {}
    for path in args.requests:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                wanted[row["uid"]] = row

    records = []
    for root in args.records:
        for path in glob.glob(str(Path(root) / "progress" / "*.json")):
            try:
                record = json.loads(Path(path).read_text())
            except Exception:
                continue
            if record.get("uid") not in wanted:
                continue
            source, target = record.get("source_image"), record.get("target_image")
            if not source or not target:
                continue
            if not Path(source).exists() or not Path(target).exists():
                continue
            records.append(record)

    session = requests.Session()
    key = os.environ.get("METTO_GLM_KEY") or os.environ["METTO_API_KEY"]
    session.headers.update({"Authorization": "Bearer " + key,
                            "Content-Type": "application/json"})
    lock = threading.Lock()
    counts = {}
    approved, rejected = [], []
    started = time.time()

    def work(record):
        spec = wanted[record["uid"]]
        if not args.no_pixel_gate:
            gate = pixel_gate(load(record["source_image"]), load(record["target_image"]),
                              spec.get("edit_box") or record.get("edit_box"),
                              spec.get("preserve_boxes") or record.get("preserve_boxes") or [])
            if not gate["passed"]:
                with lock:
                    counts[gate["reason"]] = counts.get(gate["reason"], 0) + 1
                    rejected.append(dict(uid=record["uid"], stage="pixels",
                                         reason=gate["reason"], pixel=gate))
                return
        else:
            gate = None
        text = judge(session, record["source_image"], record["target_image"],
                     spec.get("edit_instruction"), spec.get("tail_condition"),
                     spec.get("preserve") or "")
        if text is None:
            with lock:
                counts["glm_unreachable"] = counts.get("glm_unreachable", 0) + 1
                rejected.append(dict(uid=record["uid"], stage="glm",
                                     reason="glm_unreachable"))
            return
        fields = parse(text)
        ok, reason = decide(fields)
        with lock:
            counts["approved" if ok else reason] = counts.get("approved" if ok else reason, 0) + 1
            payload = dict(record, family=spec.get("family"), operation=spec.get("operation"),
                           edit_instruction=spec.get("edit_instruction"),
                           tail_condition=spec.get("tail_condition"),
                           edit_box=spec.get("edit_box") or record.get("edit_box"),
                           preserve_boxes=spec.get("preserve_boxes") or record.get("preserve_boxes"),
                           glm_fields=fields, pixel=gate,
                           needs_eye_review=(fields.get("QUALITY_OK") or "").strip().lower() != "yes")
            (approved if ok else rejected).append(payload)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, records))

    for path, rows in ((args.out_approved, approved), (args.out_rejected, rejected)):
        with open(path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(pairs=len(records), approved=len(approved),
                          rejected=len(rejected), counts=counts,
                          seconds=round(time.time() - started, 1)), indent=2))


if __name__ == "__main__":
    main()
