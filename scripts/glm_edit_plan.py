"""Design one edit per rendered parent, mixing two sources of instruction.

The rendered parents carry no instruction of their own (all 42,760 of them have
an empty ``edit_instruction``), so the instruction has to be authored after the
fact against the picture that was actually drawn. Two families do that job, and
they answer different questions:

``invent``  70%  look at the completion and propose a change worth making.
                Deterministic to specify, so the target state is decidable.
``repair``  30%  compare the completion against its own prompt and name the one
                requirement the picture failed. This is the failure the editor
                meets in deployment.

Both families obey the same operation quota, and both are rejected unless the
picture supports the instruction: the object has box coordinates, the target
state is stated, and at least one preserve question is visibly true now.

Usage::

    METTO_GLM_KEY=... python scripts/glm_edit_plan.py \\
        --parents <multi-time cache> --out <jsonl> --limit 500 --workers 48
"""

import argparse
import base64
import glob
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENDPOINT = "/v1/chat/completions"
MODEL = "glm-5.3-flash"
IMAGE_SIDE = 512.0

# The four operations the first adaptation round covers. Relation and pose are
# held back until the closed loop is running: their target states are hard to
# state unambiguously, so they cost acceptance rate without teaching more.
OPERATIONS = ("remove", "attribute", "add", "move")
QUOTA = {"remove": 0.40, "attribute": 0.25, "add": 0.20, "move": 0.15}

FAMILIES = ("invent", "repair")
FAMILY_WEIGHTS = {"invent": 0.70, "repair": 0.30}

SYSTEM_INVENT = (
    "You look at one finished image and design ONE localised edit for it.\n"
    "Coordinates are pixels in a 512x512 image, origin top-left, [x1,y1,x2,y2].\n"
    "Every object you name must be one you can actually see, at a box you give.\n"
    "Answer with the blocks below and nothing else, headers copied exactly:\n"
    "OPERATION: <add|remove|move|attribute>\n"
    "EDIT: <imperative, at most 12 words, naming exactly one visible object>\n"
    "EDIT_BOX: [x1,y1,x2,y2] of the object this edit acts on "
    "(for add: the empty place it will occupy)\n"
    "TARGET: <the whole scene after the change, at most 60 words; keep what is "
    "visible, state the one change concretely enough that a different person "
    "could tell whether it happened>\n"
    "EDIT_QUESTION: <one yes/no question that is false now and true after>\n"
    "PRESERVE: then 2 lines, each '- [x1,y1,x2,y2] <yes/no question about that "
    "object or relation>'; each must be clearly visible now and untouched by the edit"
)

SYSTEM_REPAIR = (
    "You compare one finished image against the prompt that was meant to produce it.\n"
    "Find the single clearest requirement the image FAILED: a missing object, a "
    "wrong count, a wrong colour or size, a wrong position, a broken relation.\n"
    "Ignore style, lighting, background and anything the prompt did not state.\n"
    "If the image already satisfies the prompt in everything you can check, answer "
    "exactly: NO_FAILURE\n"
    "Otherwise answer with these blocks and nothing else, headers copied exactly:\n"
    "FAILURE: <one sentence: what the prompt required and what the image shows>\n"
    "OPERATION: <add|remove|move|attribute>\n"
    "EDIT: <imperative, at most 12 words, naming exactly one visible object, "
    "repairing that failure>\n"
    "EDIT_BOX: [x1,y1,x2,y2] pixels in a 512x512 image of the object to change "
    "(for add: the empty place it will occupy)\n"
    "TARGET: <the whole scene after the repair, at most 60 words, matching the prompt>\n"
    "EDIT_QUESTION: <one yes/no question that is false now and true after>\n"
    "PRESERVE: then 2 lines, each '- [x1,y1,x2,y2] <yes/no question about that "
    "object or relation>'; each must be correct now and untouched by the edit"
)

_OP = re.compile(r"^OPERATION\s*[::]\s*(.+)$", re.I | re.M)
_EDIT = re.compile(r"^EDIT\s*[::]\s*(.+)$", re.I | re.M)
_TARGET = re.compile(r"^TARGET\s*[::]\s*(.+)$", re.I | re.M)
_EQ = re.compile(r"^EDIT_QUESTION\s*[::]\s*(.+)$", re.I | re.M)
_FAILURE = re.compile(r"^FAILURE\s*[::]\s*(.+)$", re.I | re.M)
_BOX = re.compile(
    r"^EDIT_BOX\s*[::]\s*\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?",
    re.I | re.M)
_PBOX = re.compile(
    r"^-\s*\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?\s*(.+)$")


def _clip(values):
    x1, y1, x2, y2 = [max(0.0, min(IMAGE_SIDE, float(v))) for v in values]
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) < 64:  # a box under 8x8 pixels names nothing
        return None
    return [x1, y1, x2, y2]


def parse(text):
    def one(pattern):
        match = pattern.search(text or "")
        return match.group(1).strip() if match else None

    preserve = []
    in_block = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^PRESERVE\s*[::]?", stripped, re.I):
            in_block = True
            inline = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if inline.startswith("-"):
                stripped = inline
            else:
                continue
        if in_block and stripped.startswith("-"):
            match = _PBOX.match(stripped)
            if match:
                box = _clip(match.groups()[:4])
                if box:
                    preserve.append(dict(box=box, question=match.group(5).strip()))
    box_match = _BOX.search(text or "")
    question = one(_EQ)
    return dict(
        operation=(one(_OP) or "").lower().strip() or None,
        instruction=one(_EDIT),
        tail_condition=one(_TARGET),
        failure=one(_FAILURE),
        edit_questions=[question] if question else [],
        edit_box=_clip(box_match.groups()) if box_match else None,
        preserve=preserve[:3],
        preserve_questions=[item["question"] for item in preserve[:3]],
    )


def valid(row, family):
    if not row["instruction"] or not row["tail_condition"]:
        return "unparsed"
    if row["operation"] not in OPERATIONS:
        return "operation_missing"
    words = len(row["instruction"].split())
    if not 2 <= words <= 16:
        return "instruction_length"
    if not row["edit_questions"]:
        return "no_edit_question"
    if len(row["preserve_questions"]) < 1:
        return "no_preserve"
    if row["operation"] != "add" and not row["edit_box"]:
        return "no_edit_box"
    if family == "repair" and not row["failure"]:
        return "repair_without_failure"
    # A target state has to be usable as a terminal-image supervision target.
    if len(row["tail_condition"].split()) < 8:
        return "target_too_short"
    return None


def call(session, image_path, prompt, operation, family, timeout=150):
    try:
        encoded = base64.b64encode(open(image_path, "rb").read()).decode()
    except OSError:
        return None
    if family == "invent":
        system = SYSTEM_INVENT
        user = (f"REQUIRED OPERATION: {operation}\n\n"
                f"ORIGINAL PROMPT (context only; the picture is what counts):\n"
                f"{(prompt or '')[:800]}")
    else:
        system = SYSTEM_REPAIR
        user = (f"REQUIRED OPERATION: {operation}\n\n"
                f"PROMPT THIS IMAGE WAS MEANT TO FOLLOW:\n{(prompt or '')[:800]}")
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + encoded}},
                {"type": "text", "text": user},
            ]},
        ],
        "stream": False,
        # Repair is a judgement and wants a stable answer; invent benefits from
        # variety so the same parent does not always yield the same edit.
        "temperature": 0.2 if family == "repair" else 0.9,
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
    ap.add_argument("--parents", required=True, nargs="+", help="multi-time cache dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="write one raw reply per record to stderr for inspection")
    args = ap.parse_args()

    records, seen = [], set()
    for root in args.parents:
        for path in glob.glob(str(Path(root) / "progress" / "*.json")):
            try:
                record = json.loads(Path(path).read_text())
            except Exception:
                continue
            uid = record.get("uid")
            if not uid or uid in seen:
                continue
            if not record.get("source_image") or not Path(record["source_image"]).exists():
                continue
            # A parent that already carries an instruction is used as-is.
            if (record.get("edit_instruction") or "").strip():
                continue
            record["_root"] = root
            seen.add(uid)
            records.append(record)
    records.sort(key=lambda r: hashlib.sha1(("plan:" + r["uid"]).encode()).hexdigest())
    records = records[: args.limit] if args.limit else records

    done = set()
    if args.skip_existing and Path(args.out).exists():
        for line in open(args.out):
            if line.strip():
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass

    key = os.environ.get("METTO_GLM_KEY") or os.environ["METTO_API_KEY"]
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + key,
                            "Content-Type": "application/json"})
    out = open(args.out, "a")
    lock = threading.Lock()
    counts, rejected, written = {}, {}, 0
    started = time.time()

    # Quota and family are served by position, so the mix is identical however
    # many threads run and however the pool happens to schedule them.
    # Interleave rather than concatenate, so a small run already carries the
    # intended mix instead of the first N of one operation.
    def interleave(weights):
        # Stepped schedule: at step i pick the class furthest behind its own
        # share (i*weight). Every prefix therefore tracks the target mix, so a
        # 40-record run is not all one operation.
        names = list(weights)
        out = []
        for i in range(100):
            key = max(names, key=lambda k: i * weights[k] - out.count(k))
            out.append(key)
        return out

    order = interleave(dict(QUOTA))
    families = interleave(dict(FAMILY_WEIGHTS))
    positions = {record["uid"]: index for index, record in enumerate(records)}

    def work(record):
        nonlocal written
        if record["uid"] in done:
            return
        index = positions[record["uid"]]
        operation = order[index % len(order)]
        family = families[(index * 7) % len(families)]
        text = call(session, record["source_image"],
                    record.get("parent_condition") or "", operation, family)
        if args.smoke:
            with lock:
                print(json.dumps(dict(uid=record["uid"][:12], family=family,
                                      operation=operation, reply=text)), flush=True)
        if text and text.strip().upper().startswith("NO_FAILURE"):
            with lock:
                rejected["repair_no_failure"] = rejected.get("repair_no_failure", 0) + 1
            return
        row = parse(text or "")
        reason = valid(row, family) if text else "empty_reply"
        if reason is None and row["operation"] != operation:
            reason = "operation_mismatch"
        with lock:
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                return
            out.write(json.dumps(dict(
                uid=record["uid"],
                family=family,
                operation=row["operation"],
                failure=row["failure"],
                edit_instruction=row["instruction"],
                parent_condition=record.get("parent_condition") or "",
                tail_condition=row["tail_condition"],
                preserve="",
                edit_questions=row["edit_questions"],
                preserve_questions=row["preserve_questions"],
                edit_box=row["edit_box"],
                preserve_boxes=row["preserve"],
                task_type="active_edit",
                seen_image=True,
                source_image=record["source_image"],
                parent_marker=str(Path(record["_root"]) / "progress" / f"{record['uid']}.json"),
                split=record.get("split", "train"),
                source_family=record.get("source_family"),
            ), ensure_ascii=False) + "\n")
            out.flush()
            counts[row["operation"]] = counts.get(row["operation"], 0) + 1
            written += 1
            if written % 100 == 0:
                rate = written / max(time.time() - started, 1e-6)
                print(json.dumps(dict(written=written, per_min=round(rate * 60, 1),
                                      ops=counts, rejected=rejected)), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, records))
    out.close()
    print(json.dumps(dict(candidates=len(records), written=written,
                          operations=counts, rejected=rejected), indent=2))


if __name__ == "__main__":
    main()
