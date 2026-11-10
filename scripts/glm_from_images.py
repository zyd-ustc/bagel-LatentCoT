"""Candidate edit instructions from GLM-5.3-flash, written against the real picture.

The first version sent only the prompt, and its candidates failed verification
three ways, all of them the same cause: the model did not know what the render
actually drew. Measured on 40 joined records, 15% survived:

    edit_true       37.5%   the edit it named was already true
    no_protected    30.0%   its preserve set was false on the picture
    subject_absent  17.5%   the object it named was not there

GLM accepts images, so it now reads the parent's own completion. The prompt is
sent as context only; every object and relation in the instruction must be one
the model can see.

Usage::

    METTO_GLM_KEY=... python scripts/glm_from_images.py \\
        --parents <multi-time cache> --out <jsonl> --limit 60000 --workers 32
"""

import argparse
import base64
import glob
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENDPOINT = "/v1/chat/completions"
MODEL = "glm-5.3-flash"
OPERATIONS = ("add", "remove", "move", "relation", "attribute")
QUOTA = {"add": 0.20, "remove": 0.25, "move": 0.20, "relation": 0.15, "attribute": 0.20}

SYSTEM = (
    "You look at a finished image and design ONE localised edit for it.\n"
    "Coordinates are pixels in the image, origin top-left, format [x1,y1,x2,y2].\n"
    "Name every object you mention by bounding box, so a reader can point at it.\n"
    "Answer with the blocks below and nothing else, headers copied exactly:\n"
    "OPERATION: <add|remove|move|relation|attribute>\n"
    "EDIT: <imperative, at most 12 words, naming one visible object>\n"
    "EDIT_BOX: [x1,y1,x2,y2] of the object this edit acts on (for add, the place it "
    "will occupy)\n"
    "TARGET: <the scene after the change, at most 60 words; keep the visible scene, "
    "describe only this one change>\n"
    "EDIT_QUESTION: <one yes/no question about that object; false for this image and "
    "true after the change>\n"
    "PRESERVE: then 2 lines, each starting with '- [x1,y1,x2,y2] ' followed by a "
    "yes/no question about that object or its relation; each must be clearly visible "
    "and untouched by this edit"
)

_OP = re.compile(r"^OPERATION\s*[::]\s*(.+)$", re.I | re.M)
_EDIT = re.compile(r"^EDIT\s*[::]\s*(.+)$", re.I | re.M)
_TARGET = re.compile(r"^TARGET\s*[::]\s*(.+)$", re.I | re.M)
_EQ = re.compile(r"^EDIT_QUESTION\s*[::]\s*(.+)$", re.I | re.M)
_BOX = re.compile(r"^EDIT_BOX\s*[::]\s*\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?", re.I | re.M)
_PBOX = re.compile(r"^-\s*\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?\s*(.+)$")
IMAGE_SIDE = 512.0


def _clip(values):
    x1, y1, x2, y2 = [max(0.0, min(IMAGE_SIDE, float(v))) for v in values]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse(text):
    def one(pattern):
        match = pattern.search(text or "")
        return match.group(1).strip() if match else None

    preserve, block = [], False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^PRESERVE\s*[::]?", stripped, re.I):
            block = True
            inline = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if inline.startswith("-"):
                stripped = inline
            else:
                continue
        if block and stripped.startswith("-"):
            match = _PBOX.match(stripped)
            if match:
                box = _clip(match.groups()[:4])
                if box:
                    preserve.append(dict(box=box, question=match.group(5).strip()))
    box_match = _BOX.search(text or "")
    edit_box = _clip(box_match.groups()) if box_match else None
    question = one(_EQ)
    return dict(operation=(one(_OP) or "").lower().strip() or None,
                instruction=one(_EDIT), tail_condition=one(_TARGET),
                edit_questions=[question] if question else [],
                edit_box=edit_box,
                preserve=preserve[:3],
                preserve_questions=[item["question"] for item in preserve[:3]])


def valid(row):
    if not row["instruction"] or not row["tail_condition"]:
        return "unparsed"
    if row["operation"] not in OPERATIONS:
        return "operation_missing"
    if not 2 <= len(row["instruction"].split()) <= 16:
        return "instruction_length"
    if not row["edit_questions"]:
        return "no_edit_question"
    if len(row["preserve_questions"]) < 1:
        return "no_preserve"
    if row["operation"] != "add" and not row["edit_box"]:
        return "no_edit_box"
    return None


def next_operation(counts):
    total = sum(counts.values()) + 1
    return max((op for op in QUOTA),
               key=lambda op: QUOTA[op] - counts.get(op, 0) / total)


def call(session, image_path, prompt, operation, timeout=120):
    try:
        encoded = base64.b64encode(open(image_path, "rb").read()).decode()
    except OSError:
        return None
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + encoded}},
                {"type": "text",
                 "text": f"REQUIRED OPERATION: {operation}\n\nORIGINAL PROMPT (context only, "
                         f"the image is what counts):\n{prompt[:800]}"},
            ]},
        ],
        "stream": False,
        "temperature": 0.9,
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
            time.sleep(1.5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", required=True, help="multi-time cache directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    records = []
    for path in glob.glob(str(Path(args.parents) / "progress" / "*.json")):
        record = json.loads(Path(path).read_text())
        if record.get("source_image") and Path(record["source_image"]).exists():
            records.append(record)
    records.sort(key=lambda r: hashlib.sha1(("glmimg:" + r["uid"]).encode()).hexdigest())
    records = records[: args.limit]

    done = set()
    if args.skip_existing and Path(args.out).exists():
        done = {json.loads(line)["uid"] for line in open(args.out) if line.strip()}

    key = os.environ.get("METTO_GLM_KEY") or os.environ["METTO_API_KEY"]
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + key,
                            "Content-Type": "application/json"})
    out = open(args.out, "a")
    lock = __import__("threading").Lock()
    counts, rejected, written = {}, {}, 0
    started = time.time()

    order = []
    for op in ("remove", "add", "move", "attribute", "relation"):
        order.extend([op] * int(round(QUOTA[op] * 100)))
    # Quota served by position: deterministic under any thread count.
    positions = {record["uid"]: index for index, record in enumerate(records)}

    def work(record):
        nonlocal written
        if record["uid"] in done:
            return
        operation = order[positions[record["uid"]] % len(order)]
        text = call(session, record["source_image"],
                    record.get("parent_condition") or "", operation)
        row = parse(text or "")
        reason = valid(row) if text else "empty_reply"
        if reason is None and row["operation"] != operation:
            reason = "operation_mismatch"
        with lock:
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                return
            out.write(json.dumps(dict(
                uid=record["uid"], operation=row["operation"],
                edit_instruction=row["instruction"],
                parent_condition=record.get("parent_condition") or "",
                tail_condition=row["tail_condition"], preserve="",
                edit_questions=row["edit_questions"],
                preserve_questions=row["preserve_questions"],
                edit_box=row["edit_box"],
                preserve_boxes=row["preserve"],
                task_type="active_edit", seen_image=True,
                source_image=record["source_image"],
                parent_marker=str(Path(args.parents) / "progress" / f"{record['uid']}.json"),
                split=record.get("split", "train"),
                source_family=record.get("source_family"),
            ), ensure_ascii=False) + "\n")
            out.flush()
            counts[row["operation"]] = counts.get(row["operation"], 0) + 1
            written += 1
            if written % 200 == 0:
                rate = written / max(time.time() - started, 1e-6)
                print(json.dumps(dict(written=written, per_min=round(rate * 60, 1),
                                      ops=counts, rejected=rejected)), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, records))
    out.close()
    print(json.dumps(dict(written=written, operations=counts, rejected=rejected)))


if __name__ == "__main__":
    main()
