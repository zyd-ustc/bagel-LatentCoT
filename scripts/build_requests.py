"""Join parent scores with planned instructions into production requests.

Task type is decided here, once, from evidence rather than from which prompt
happened to author the instruction:

``active_edit``  a body-clean parent plus an *invent* instruction. The source is
                 worth keeping, so the edit is the only thing that should change.
``repair``       a body-clean parent plus a *repair* instruction, i.e. the
                 planner found a requirement the picture missed. Its ``c_tail``
                 will be the original prompt, not a rewritten scene.

A parent whose body failed scoring is excluded from both: an ambiguous source
makes an ambiguous target, and this round needs unambiguous targets.

Usage::

    python build_requests.py --scores parent_scores.jsonl --plan plan/all42k.jsonl \
        --out requests_v2.jsonl --limit 5000
"""

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--active-share", type=float, default=0.7)
    ap.add_argument("--require-readable", action="store_true", default=True)
    ap.add_argument("--val-share", type=float, default=0.05,
                    help="Held-out share, assigned per parent trajectory so the same "
                         "parent never straddles train and val")
    ap.add_argument("--parent-cache", nargs="*", default=[],
                    help="Parent caches; a request whose uid has no state there cannot "
                         "be produced, so it is dropped here rather than failing later")
    args = ap.parse_args()

    # The renderers wrote more completion pictures than state markers (an
    # interrupted render leaves the picture but no marker). A request bound to
    # one of those has nothing for the teacher to start from, so filter on the
    # marker index instead of discovering it one skip at a time.
    available = None
    if args.parent_cache:
        available = set()
        for root in args.parent_cache:
            for path in glob.glob(str(Path(root) / "progress" / "*.json")):
                stem = Path(path).stem
                available.add(stem.rsplit("_", 1)[0] if "_" in stem else stem)

    clean = {}
    for line in open(args.scores):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("body_ok") and (row.get("readable") or not args.require_readable):
            clean[row["uid"]] = row

    wanted = {"active_edit": [], "repair": []}
    for line in open(args.plan):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["uid"] not in clean:
            continue
        if available is not None and row["uid"] not in available:
            continue
        row["task_type"] = "repair" if row.get("family") == "repair" else "active_edit"
        row["parent_condition"] = clean[row["uid"]].get("parent_condition") or row.get("parent_condition")
        row["parent_subjects"] = clean[row["uid"]].get("subjects")

        # Split by parent trajectory, not by record. One parent yields three read
        # times and possibly several instructions that all share one target
        # picture; if some of them landed in val, the val score would measure
        # memorisation of that picture. The hash makes the assignment stable
        # across rebuilds so a resumed run cannot reshuffle the split.
        bucket = int(hashlib.sha1(("split:" + row["uid"]).encode()).hexdigest()[:8], 16)
        row["split"] = "val" if (bucket % 10000) / 10000.0 < args.val_share else "train"
        # The trainer refuses a family that crosses splits, so every record from
        # one parent must carry that parent's identity in both family fields.
        row["source_family"] = row["uid"]
        row["prompt_family"] = row["uid"]
        wanted[row["task_type"]].append(row)

    total = sum(len(v) for v in wanted.values())
    if not total:
        raise SystemExit("no eligible requests: check that --scores and --plan share uids")

    if args.limit:
        # Proportional draw, so the requested share survives the cap instead of
        # one family consuming the whole budget.
        take_active = int(round(args.limit * args.active_share))
        take_repair = args.limit - take_active
        chosen = (wanted["active_edit"][:take_active] + wanted["repair"][:take_repair])
    else:
        chosen = wanted["active_edit"] + wanted["repair"]

    with open(args.out, "w") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = collections.Counter(row["task_type"] for row in chosen)
    ops = collections.Counter(row.get("operation") for row in chosen)
    print(json.dumps(dict(
        clean_parents=len(clean),
        parents_with_state=(len(available) if available is not None else None),
        eligible_active=len(wanted["active_edit"]),
        eligible_repair=len(wanted["repair"]),
        written=len(chosen), by_task=dict(counts), by_operation=dict(ops),
        by_split=dict(collections.Counter(row["split"] for row in chosen)),
        active_share=round(counts.get("active_edit", 0) / max(len(chosen), 1), 3),
    ), indent=2))


if __name__ == "__main__":
    main()
