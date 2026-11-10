"""Build active-edit requests on real parent trajectories.

The version this replaces read a finished image from the text-to-image pool and
wrote an instruction for it. That binds nothing: the student's ``a`` and ``P``
are produced later by walking the flow again from the prompt, and a second run
of the same prompt does not reproduce the picture the instruction was written
for. The instruction, the preserve set and the teacher target then refer to a
scene the student never sees.

This producer reads the parent's *own* completion and binds the request to the
same trajectory:

    a      <- the cached state at the read time
    P      <- the cached preview of that state
    x_src  <- the same trajectory's completion
    e, c_tail, edit_questions, preserve_questions  <- read from x_src

Usage::

    python scripts/active_from_parents.py --parent <cache dir>[,<cache dir>] \\
        --out <jsonl> --limit 5000 --shard-index 0 --shard-count 16
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.coupled import _read_final
from qwen_latent_cot.u15_target_correction.diagnosis import decompose
from qwen_latent_cot.u15_target_correction.explicit import (
    OPERATION_QUOTA, propose_active_edit_multi,
)
from qwen_latent_cot.u15_target_correction.labels import score_questions
from qwen_latent_cot.u15_target_correction.loading import load_native


def records_of(root):
    """Every produced parent in one cache directory."""
    records = []
    for path in sorted(glob.glob(str(Path(root) / "manifest.shard_*.json"))) or \
            sorted(glob.glob(str(Path(root) / "manifest.json"))):
        records.extend(json.loads(Path(path).read_text()).get("records", []))
    return records


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def next_operation(counts, quota):
    total = sum(counts.values()) + 1
    deficits = {op: quota[op] - counts.get(op, 0) / total for op in quota}
    return max(deficits, key=deficits.get)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", required=True,
                        help="comma separated parent cache directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)

    rows = []
    for root in args.parent.split(","):
        root = Path(root.strip()).resolve()
        for record in records_of(root):
            if record.get("source_image") and record.get("states"):
                rows.append((root, record))
    rows.sort(key=lambda item: hashlib.sha1(("active:" + str(item[1]["uid"])).encode()).hexdigest())
    rows = rows[: args.limit]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name if args.shard_count <= 1 else f"{out.stem}.shard_{args.shard_index:03d}{out.suffix}"
    target = out.parent / name
    log = (out.parent / (name + ".log")).open("a")
    done = set()
    if args.skip_existing and target.exists():
        done = {json.loads(line)["uid"] for line in target.read_text().splitlines() if line.strip()}

    written, counts = 0, {}
    for position, (root, record) in enumerate(rows):
        if position % args.shard_count != args.shard_index:
            continue
        uid = str(record["uid"])
        if uid in done:
            continue
        completion_path = resolve(root, record["source_image"])
        if not completion_path.exists():
            log.write(json.dumps(dict(uid=uid, skipped="no_completion")) + "\n")
            log.flush()
            continue
        completion = _read_final(str(completion_path), rt.device)
        raw_request = record.get("parent_condition") or ""
        proposals = propose_active_edit_multi(rt, completion, raw_request, cfg, count=5)
        requirements, _ = decompose(rt, raw_request, cfg)

        usable = []
        for candidate in proposals:
            if candidate.get("error") is not None:
                continue
            asked = candidate["edit_questions"]
            passed_edit, per_edit = score_questions(rt, completion, asked, cfg)
            # Three states: only a clear "no" proves the edit is worth doing.
            unknown = [v for q, v in per_edit.items() if v is None]
            if unknown or all(q in passed_edit for q in asked):
                continue
            keep = [
                q for q in requirements
                if not any(w in candidate["instruction"].lower()
                           for w in q.lower().split() if len(w) > 4)
            ][:3]
            if not keep:
                continue
            passed_keep, per_keep = score_questions(rt, completion, keep, cfg)
            keep = [q for q in keep if q in passed_keep]
            if not keep:
                continue
            candidate["preserve_questions"] = keep
            candidate["scores"] = {q: v for q, v in {**per_edit, **per_keep}.items()}
            usable.append(candidate)
        if not usable:
            log.write(json.dumps(dict(uid=uid, skipped="no_usable_operation")) + "\n")
            log.flush()
            continue

        total = sum(counts.values()) + 1
        within = [c for c in usable
                  if counts.get(c["operation"], 0) / total < OPERATION_QUOTA.get(c["operation"], 0)]
        if not within:
            log.write(json.dumps(dict(uid=uid, skipped="quota_full")) + "\n")
            log.flush()
            continue
        wanted = next_operation(counts, OPERATION_QUOTA)
        proposal = next((c for c in within if c["operation"] == wanted), within[0])

        row = dict(
            uid=uid,
            parent_uid=uid,                       # same trajectory, not a lookalike
            operation=proposal["operation"],
            edit_instruction=proposal["instruction"],
            # An active edit states the scene *after* the change. Copying the raw
            # request here is the repair contract and would fight the editor.
            tail_condition=proposal["target_condition"] or raw_request,
            parent_condition=raw_request,
            preserve="",
            edit_questions=list(proposal["edit_questions"]),
            preserve_questions=list(proposal["preserve_questions"]),
            completion_scores=proposal["scores"],
            source_image=str(completion_path),
            parent_cache=str(root),
            task_type="active_edit",
            split=record.get("split", "train"),
            source_family=record.get("source_family") or hashlib.sha1(uid.encode()).hexdigest()[:12],
            prompt_family=record.get("prompt_family") or uid,
        )
        target.open("a").write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[row["operation"]] = counts.get(row["operation"], 0) + 1
        written += 1
    log.close()
    print(json.dumps(dict(shard=args.shard_index, written=written,
                          operations=counts, path=str(target))))


if __name__ == "__main__":
    main()
