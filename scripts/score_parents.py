"""Score every rendered parent completion for whether it is fit to edit.

Why this exists
---------------
An active edit says "change this one thing, leave the rest". If the source
picture already has melted tools and smeared shavings, that instruction asks the
teacher to do two jobs at once: make the requested change and silently repair
damage nobody asked about. The resulting label is ambiguous, and no amount of
loss weighting fixes an ambiguous target.

Repair tasks are the opposite case — they need a picture that is clean in body
but wrong in a stated requirement. So the score separates the two:

    body_ok     are the objects solid, unduplicated, unserrated?
    readable    can the main objects be told apart at all?

``body_ok`` gates active edits. A picture that fails it is not wrong for repair —
it is simply not in this round's pool.

Usage::

    METTO_GLM_KEY=... python scripts/score_parents.py \
        --parents mt mt_b --out parent_scores.jsonl --limit 2000 --workers 96
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

import requests

ENDPOINT = "/v1/chat/completions"
MODEL = "glm-5.3-flash"

SYSTEM = (
    "You look at one generated picture and judge whether its objects are "
    "physically well formed. Ignore style, colour choice, lighting and "
    "composition — only judge whether things are drawn correctly.\n"
    "Answer with four lines and nothing else:\n"
    "BODY_OK: yes|no -- are all visible objects solid and complete, with no "
    "fusion (two objects melted together), no smearing, no torn or duplicated "
    "parts, no missing limbs of a whole object?\n"
    "READABLE: yes|no -- can you still name the main objects and tell them apart?\n"
    "SUBJECTS: <comma-separated list of the main objects you can clearly see, "
    "at most 6>\n"
    "NOTE: <one short sentence; if BODY_OK is no, say exactly what is malformed>"
)

_FIELD = re.compile(r"^([A-Z_]+)\s*:\s*(.+)$", re.M)


def parse(text):
    return {m.group(1).strip().upper(): m.group(2).strip()
            for m in _FIELD.finditer(text or "")}


def call(session, image, timeout=150):
    try:
        encoded = base64.b64encode(open(image, "rb").read()).decode()
    except OSError:
        return None
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + encoded}},
                {"type": "text", "text": "Judge this picture."},
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
            time.sleep(1.5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    records, seen = [], set()
    for root in args.parents:
        for path in glob.glob(str(Path(root) / "progress" / "*.json")):
            try:
                record = json.loads(Path(path).read_text())
            except Exception:
                continue
            uid = record.get("uid")
            image = record.get("source_image")
            if not uid or uid in seen or not image or not Path(image).exists():
                continue
            # One score per parent, not per rendered time: all three times share
            # the same completion picture.
            seen.add(uid)
            record["_root"] = root
            records.append(record)
    if args.limit:
        records = records[: args.limit]

    done = set()
    if args.skip_existing and Path(args.out).exists():
        for line in open(args.out):
            if line.strip():
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass

    session = requests.Session()
    key = os.environ.get("METTO_GLM_KEY") or os.environ["METTO_API_KEY"]
    session.headers.update({"Authorization": "Bearer " + key,
                            "Content-Type": "application/json"})
    out = open(args.out, "a")
    lock = threading.Lock()
    counts, written = {}, 0
    started = time.time()

    def work(record):
        nonlocal written
        if record["uid"] in done:
            return
        text = call(session, record["source_image"])
        fields = parse(text or "")
        # The model sometimes appends a clause after the verdict, so only the
        # leading token is the answer.
        body = (fields.get("BODY_OK") or "").split()[0].lower() if fields.get("BODY_OK") else ""
        readable = (fields.get("READABLE") or "").split()[0].lower() if fields.get("READABLE") else ""
        with lock:
            key_name = ("body_ok" if body == "yes" else
                        "body_bad" if body == "no" else "unparsed")
            counts[key_name] = counts.get(key_name, 0) + 1
            if readable:
                counts["readable_" + readable] = counts.get("readable_" + readable, 0) + 1
            out.write(json.dumps(dict(
                uid=record["uid"],
                body_ok=(body == "yes"),
                readable=(readable == "yes"),
                subjects=[s.strip() for s in (fields.get("SUBJECTS") or "").split(",") if s.strip()][:6],
                note=fields.get("NOTE"),
                source_image=record["source_image"],
                parent_root=record["_root"],
                parent_condition=record.get("parent_condition") or "",
            ), ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if written % 500 == 0:
                rate = written / max(time.time() - started, 1e-6)
                print(json.dumps(dict(scored=written, per_min=round(rate * 60, 1),
                                      counts=counts)), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, records))
    out.close()
    print(json.dumps(dict(parents=len(records), scored=written, counts=counts), indent=2))


if __name__ == "__main__":
    main()
