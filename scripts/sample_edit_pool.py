"""Stratified, family-split sample of the qualified edit-pair pool.

The pool is one JSONL of ``uid, task, prompt, src, target`` rows. L training
wants breadth, R-keep wants ``(x_src, x_repair, direction)`` triples, so the
sample keeps the task mix explicit instead of taking a slice of the file.

Usage::

    python scripts/sample_edit_pool.py --pool <qualified_nonzero.jsonl> \
        --out-dir <dir> --per-task 3000 --target-size 30000 --max-per-family 2
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

L_TASKS = {
    "T1": "text", "T2": "color", "T3": "remove", "T4": "background",
    "T5": "material", "T6": "style", "T7": "action", "T8": "count",
    "T9": "spatial", "T11": "composite",
}
# Repair-shaped tasks first: a cancelled edit of these is a real "restore the intent" case.
REPAIR_TASKS = ("T8", "T9", "T3")


def family(uid):
    """Stable image family: the pool has ~1.2 edits per unique source."""
    return hashlib.sha1(uid.encode()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-task", type=int, default=3000)
    parser.add_argument("--target-size", type=int, default=30000)
    parser.add_argument("--max-per-family", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quota = collections.defaultdict(int)
    family_count = collections.Counter()
    records, stats = [], collections.Counter()
    with open(args.pool) as handle:
        for line in handle:
            if len(records) >= args.target_size:
                break
            row = json.loads(line)
            task = row.get("task")
            if task not in L_TASKS:
                continue
            if quota[task] >= args.per_task:
                continue
            key = family(str(row["uid"]))
            if family_count[key] >= args.max_per_family:
                continue
            family_count[key] += 1
            quota[task] += 1
            stats[task] += 1
            split = "val" if (hash(f"split:{row['uid']}") % 10**6) < args.val_fraction * 10**6 else "train"
            records.append(dict(
                uid=str(row["uid"]),
                split=split,
                prompt_family="pool_" + key,
                source_family="pool_" + key,
                source_image=row["src"],
                target_image=row["target"],
                edit_instruction=(row.get("prompt") or "").strip(),
                parent_condition=(row.get("prompt_draft") or row.get("prompt") or "").strip(),
                semantic_approved=True,
                task=task,
                task_kind=L_TASKS[task],
                repair_shaped=task in REPAIR_TASKS,
            ))

    manifest = dict(kind="linear_edit_pairs", pool=str(Path(args.pool).resolve()),
                    records=records)
    (out / "L_pool.json").write_text(json.dumps(manifest))
    repair = [row for row in records if row["repair_shaped"]]
    (out / "R_repair_seed.json").write_text(json.dumps(dict(kind="repair_seed", records=repair)))
    print(json.dumps(dict(rows=len(records), families=len(family_count),
                          val=sum(row["split"] == "val" for row in records),
                          text_len_warn=sum(len(row["edit_instruction"].split()) > 60 for row in records),
                          by_task=dict(stats),
                          repair_seed=len(repair)), indent=1))


if __name__ == "__main__":
    main()
