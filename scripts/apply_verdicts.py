"""Write the final verdict back into each record, without touching the trainer.

The contract is "one judge decides". The trainer predates that contract: it reads
``states[t].accepted`` from every record, in six places. Changing all six would
be a larger edit than the problem needs, and a trainer that can compute its own
verdict can drift from the judge that was supposed to own it.

So the judge writes its decision into the records instead. After this runs:

    states[t].accepted       True for approved pairs, False otherwise
    states[t].reject_reason  the judge's reason, or None
    acceptance               the full transport verdict, kept for diagnosis

The old in-loop acceptance is not consulted and is overwritten. Provenance of
who judged is recorded so a later run can tell pre- and post-contract caches
apart.

Usage::

    python scripts/apply_verdicts.py \\
        --approved v2_full_approved.jsonl --rejected v2_full_rejected.jsonl \\
        --cache v2_full
"""

import argparse
import glob
import json
import os
from pathlib import Path

JUDGE = "accept_final/glm4+pixels"


def load_verdicts(paths, approved):
    table = {}
    for path in paths:
        if not Path(path).exists():
            continue
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            reason = None if approved else (
                row.get("transport_verdict") or row.get("reason") or "unknown")
            table[row["uid"]] = dict(approved=approved, reason=reason)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approved", nargs="+", required=True)
    ap.add_argument("--rejected", nargs="+", default=[])
    ap.add_argument("--cache", required=True, help="coupled cache holding progress/*.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    verdicts = load_verdicts(args.approved, True)
    verdicts.update(load_verdicts(args.rejected, False))

    changed = skipped = missing = 0
    for path in glob.glob(str(Path(args.cache) / "progress" / "*.json")):
        try:
            record = json.loads(Path(path).read_text())
        except Exception:
            skipped += 1
            continue
        verdict = verdicts.get(record.get("uid"))
        if verdict is None:
            # A record with no verdict was never judged; leave it out of training
            # rather than guessing in its favour.
            missing += 1
            for item in (record.get("states") or {}).values():
                item["accepted"] = False
                item["reject_reason"] = "not_judged"
            record["acceptance"] = dict(approved=False, reason="not_judged", judge=None)
        else:
            for item in (record.get("states") or {}).values():
                item["accepted"] = bool(verdict["approved"])
                item["reject_reason"] = None if verdict["approved"] else verdict["reason"]
            record["acceptance"] = dict(approved=bool(verdict["approved"]),
                                        reason=verdict["reason"], judge=JUDGE)
        if not args.dry_run:
            Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        changed += 1

    # The trainer reads sharded manifests, not progress files. Those manifests
    # were written when production ended, before any verdict existed, so every
    # state in them says accepted=None and the trainer rejects the whole cache
    # with "no accepted training times". Rewrite them from the same verdicts.
    manifests = sorted(glob.glob(str(Path(args.cache) / "manifest*.json")))
    m_records = m_acc = 0
    for path in manifests:
        try:
            manifest = json.loads(Path(path).read_text())
        except Exception:
            continue
        for record in manifest.get("records") or []:
            verdict = verdicts.get(record.get("uid"))
            approved = bool(verdict["approved"]) if verdict else False
            reason = None if approved else (
                verdict["reason"] if verdict else "not_judged")
            for item in (record.get("states") or {}).values():
                item["accepted"] = approved
                item["reject_reason"] = reason
            record["acceptance"] = dict(approved=approved, reason=reason, judge=JUDGE)
            m_records += 1
            m_acc += 1 if approved else 0
        if not args.dry_run:
            Path(path).write_text(json.dumps(manifest, ensure_ascii=False) + "\n")

    print(json.dumps(dict(records=changed, no_verdict=missing, unreadable=skipped,
                          manifests=len(manifests), manifest_records=m_records,
                          manifest_approved=m_acc,
                          approved=sum(1 for v in verdicts.values() if v["approved"]),
                          rejected=sum(1 for v in verdicts.values() if not v["approved"]),
                          dry_run=bool(args.dry_run)), indent=2))


if __name__ == "__main__":
    main()
