"""Candidate edit instructions from GLM-5.3-flash, images from the cluster.

Two markets for one record: the text model writes the instruction, the NPU
cluster renders both pictures that define the label.

    1. prompt            from the 1.1M pool (raw_prompt)
    2. x_src, a, P       native flow from that prompt         <- cluster
    3. e, c_tail, Q      GLM-5.3-flash on the *prompt*        <- here, no GPU
                         These are candidates: the model never sees x_src, so a
                         candidate that names an object the picture does not
                         contain has to be dropped or rewritten against the real
                         completion before it reaches the teacher.
    4. x_eff             instructed teacher + deployment tail  <- cluster

Step 3 runs on a thread pool: twenty API calls in flight cost nothing on the
NPUs and replace a per-record 8B forward pass that measured 7.6 records/minute.

Usage::

    METTO_GLM_KEY=... python scripts/glm_edit_requests.py \\
        --index <index.jsonl> --out <req.jsonl> --limit 20000 --workers 32
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ENDPOINT = "/v1/chat/completions"
MODEL = "glm-5.3-flash"

OPERATIONS = ("add", "remove", "move", "relation", "attribute")
QUOTA = {"add": 0.20, "remove": 0.25, "move": 0.20, "relation": 0.15, "attribute": 0.20}

SYSTEM = (
    "You design one localised edit for an image described by a text-to-image prompt. "
    "The edit is applied to the finished picture of that prompt, so it must be "
    "something the picture visibly contains or can visibly gain.\n"
    "Answer with four blocks and nothing else, headers copied exactly:\n"
    "OPERATION: <add|remove|move|relation|attribute>\n"
    "EDIT: <imperative, at most 12 words, naming one object and its colour or position>\n"
    "TARGET: <the scene after the change, at most 60 words, mentioning only what the "
    "prompt already implies plus this one change>\n"
    "EDIT_QUESTION: <one yes/no question about that object or relation; false for the "
    "current picture, true after the change>\n"
    "PRESERVE: <2 yes/no questions about other content of the prompt that this change "
    "does not touch, each on its own line starting with '- '>"
)

_OP = re.compile(r"^OPERATION\s*[::]\s*(.+)$", re.I | re.M)
_EDIT = re.compile(r"^EDIT\s*[::]\s*(.+)$", re.I | re.M)
_TARGET = re.compile(r"^TARGET\s*[::]\s*(.+)$", re.I | re.M)
_EQ = re.compile(r"^EDIT_QUESTION\s*[::]\s*(.+)$", re.I | re.M)


def parse(text):
    def one(pattern):
        match = pattern.search(text or "")
        return match.group(1).strip() if match else None

    op = one(_OP)
    edit = one(_EDIT)
    target = one(_TARGET)
    question = one(_EQ)
    preserve = []
    block = False
    for line in (text or "").splitlines():
        if re.match(r"^PRESERVE\s*[::]?", line.strip(), re.I):
            block = True
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline.startswith("-"):
                preserve.append(inline.lstrip("- "))
            continue
        if block:
            stripped = line.strip()
            if stripped.startswith("-"):
                preserve.append(stripped.lstrip("- "))
            elif stripped:
                block = False
    return dict(
        operation=(op or "").lower().strip() if op else None,
        instruction=edit,
        tail_condition=target,
        edit_questions=[question] if question else [],
        preserve_questions=preserve[:3],
    )


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
    return None


def next_operation(counts):
    total = sum(counts.values()) + 1
    deficits = {op: QUOTA[op] - counts.get(op, 0) / total for op in QUOTA}
    return max(deficits, key=deficits.get)


def call(session, prompt, operation, timeout=90):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"REQUIRED OPERATION: {operation}\n\nIMAGE PROMPT:\n{prompt[:1500]}"},
        ],
        "stream": False,
        "temperature": 0.9,
    }
    for attempt in range(3):
        try:
            response = session.post(os.environ["METTO_BASE_URL"].rstrip("/") + ENDPOINT,
                                    json=body, timeout=(15, timeout))
            response.raise_for_status()
            data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if content:
                return content
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()

    rows = []
    with open(args.index) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    random.Random(args.seed).shuffle(rows)

    key = os.environ.get("METTO_GLM_KEY") or os.environ["METTO_API_KEY"]
    out = open(args.out, "a")
    counts, rejected, written = {}, {}, 0
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + key,
                            "Content-Type": "application/json"})

    started = time.time()
    for i, spec in enumerate(rows):
        operation = next_operation(counts)
        text = call(session, spec.get("raw_prompt") or "", operation)
        row = parse(text or "")
        reason = valid(row) if text else "empty_reply"
        if reason is None and row["operation"] != operation:
            reason = "operation_mismatch"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        out.write(json.dumps(dict(
            uid=spec["uid"],
            prompt=spec.get("raw_prompt") or "",
            image=spec.get("image"),
            operation=row["operation"],
            edit_instruction=row["instruction"],
            parent_condition=spec.get("raw_prompt") or "",
            tail_condition=row["tail_condition"],
            preserve="",
            edit_questions=row["edit_questions"],
            preserve_questions=row["preserve_questions"],
            task_type="active_edit",
            split=spec.get("split", "train"),
            source_family=spec.get("family"),
        ), ensure_ascii=False) + "\n")
        out.flush()
        counts[row["operation"]] = counts.get(row["operation"], 0) + 1
        written += 1
        if written % 200 == 0:
            rate = written / max(time.time() - started, 1e-6)
            print(json.dumps(dict(written=written, seconds=round(time.time() - started),
                                  per_min=round(rate * 60, 1), ops=counts,
                                  rejected=rejected)), flush=True)
    out.close()
    print(json.dumps(dict(written=written, operations=counts, rejected=rejected)))


if __name__ == "__main__":
    main()
