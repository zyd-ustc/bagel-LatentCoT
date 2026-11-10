"""Judge produced pairs with GLM — the only judge in this pipeline.

The 8B yes/no head was removed from the data pipeline on instruction and the
measurement backs it: on pairs where the pictures differ by 0.11 of the display
range it returned identical answers, so it could not see that a picture had
changed at all.

This judge receives the source picture, the edited picture and the instruction,
and answers three things about the *pair*:

* did the requested change happen (and is it complete)?
* did anything the instruction was told to keep change?
* are both pictures usable (no fusion, no smearing, no torn edges)?

Usage::

    METTO_GLM_KEY=... python scripts/glm_verify.py --records <cache> [<cache>...] \\
        --out <jsonl> --workers 24
"""

import argparse
import base64
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENDPOINT = "/v1/chat/completions"
MODEL = "glm-5.3-flash"

SYSTEM = (
    "You compare two pictures of the same scene. Picture A is before the edit; "
    "picture B is after it. Decide three things.\n"
    "Answer with four lines and nothing else:\n"
    "EDIT_DONE: yes|partial|no -- did the instruction's change happen in B?\n"
    "EDIT_NOTE: <one short sentence on what actually changed, or what is still wrong>\n"
    "KEEP_OK: yes|no -- is everything the instruction did not ask to change still as in A?\n"
    "QUALITY_OK: yes|no -- are both pictures free of fusion, smearing, torn edges or "
    "duplicated objects? A pre-existing artefact in A alone does not count."
)


def encode(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def judge(session, before, after, instruction, preserve_text, timeout=150):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "PICTURE A (before):"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encode(before)}},
                {"type": "text", "text": "PICTURE B (after):"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encode(after)}},
                {"type": "text", "text":
                    "INSTRUCTION:\n" + (instruction or "") +
                    ("\n\nMUST STAY THE SAME:\n" + preserve_text if preserve_text else "")},
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
            return (response.json().get("choices") or [{}])[0].get("message", {}).get("content")
        except Exception:
            if attempt == 2:
                return None
    return None


def parse(text):
    fields = {}
    for line in (text or "").splitlines():
        for key in ("EDIT_DONE", "EDIT_NOTE", "KEEP_OK", "QUALITY_OK"):
            if line.strip().upper().startswith(key + ":"):
                value = line.split(":", 1)[1].strip()
                if key != "EDIT_NOTE":
                    value = value.split("--")[0].split()[0].strip(" .,").lower()
                fields[key] = value
    return fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, nargs="+")
    ap.add_argument("--requests", required=True, help="jsonl with instruction and preserve text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    wanted = {}
    for line in open(args.requests):
        if line.strip():
            row = json.loads(line)
            wanted[row["uid"]] = row

    pairs = []
    for root in args.records:
        for path in glob.glob(str(Path(root) / "progress" / "*.json")):
            record = json.loads(Path(path).read_text())
            if record.get("uid") in wanted and record.get("source_image") and record.get("target_image"):
                pairs.append(record)

    session = requests.Session()
    session.headers.update({
        "Authorization": "Bearer " + (os.environ.get("METTO_GLM_KEY") or os.environ["METTO_API_KEY"]),
        "Content-Type": "application/json"})
    counts = {}

    def work(record):
        spec = wanted[record["uid"]]
        text = judge(session, record["source_image"], record["target_image"],
                     spec.get("edit_instruction"), spec.get("preserve") or "")
        fields = parse(text or "")
        done = (fields.get("EDIT_DONE") or "").lower()
        keep = (fields.get("KEEP_OK") or "").lower()
        quality = (fields.get("QUALITY_OK") or "").lower()
        approved = done == "yes" and keep == "yes" and quality == "yes"
        reason = None if approved else (
            "quality" if quality != "yes" else
            "keep_broken" if keep != "yes" else
            "edit_" + (done or "unparsed"))
        return record, fields, approved, reason

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(work, pairs))

    out = open(args.out, "w")
    for record, fields, approved, reason in results:
        counts[reason or "approved"] = counts.get(reason or "approved", 0) + 1
        if approved:
            out.write(json.dumps(dict(record, glm_verdict=fields), ensure_ascii=False) + "\n")
    out.close()
    print(json.dumps(dict(pairs=len(pairs), approved=counts.get("approved", 0), counts=counts)))


if __name__ == "__main__":
    main()

# The 8B question set is no longer used anywhere in this pipeline. Scripts that
# still import it exist for the older repair branch only and are not part of the
# active-edit path: verify_candidates.py, rescore_existing.py and
# active_from_parents.py were superseded by this file and by verify_by_pixels.py.
