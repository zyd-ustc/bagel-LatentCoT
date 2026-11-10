"""Build active-edit requests from the 1.1M text-to-image pool.

The pool ships a finished picture per prompt, so no parent trajectory has to be
generated: the student's ``a`` and ``P`` are produced afterwards by walking the
native flow to the read time, and the expert reads the *finished* picture to
decide what to change.

Three records per prompt are enough: the prompt, the finished image, and a
stable uid. Splitting is by prompt family, never by the new task id.

Usage::

    python scripts/active_from_pool.py --index out/active_index.jsonl \\
        --per-operation 120000 --out out/active_requests.jsonl \\
        --shard-index 0 --shard-count 16
"""

import argparse
import hashlib
import json
from pathlib import Path

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.explicit import (
    OPERATION_QUOTA, propose_active_edit_from_request, propose_active_edit_multi,
)
from qwen_latent_cot.u15_target_correction.diagnosis import decompose
from qwen_latent_cot.u15_target_correction.labels import score_questions
from qwen_latent_cot.u15_target_correction.loading import load_native


def read_image(path, rt):
    """Load onto the protocol canvas: the expert's grid is fixed at 512x512.

    The pool ships mixed aspect ratios (3:2, 1:1, 2:3 ...). Patching a 3:2 file
    at the 512 grid yields 4128 tokens against the 1024 the model expects, so
    the file has to go through the same canvas mapping training uses.
    """
    from qwen_latent_cot.u15_target_correction.images import load_canvas

    image, _, _ = load_canvas(path, rt.width, rt.height, rt.device)
    return image


def next_operation(counts, quota):
    total = sum(counts.values()) + 1
    deficits = {op: quota[op] - counts.get(op, 0) / total for op in quota}
    return max(deficits, key=deficits.get)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True,
                        help="prompt jsonl with uid, image, raw_prompt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-operation", type=int, default=100000,
                        help="rows taken per operation before sharding")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)

    rows = []
    with open(args.index) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: hashlib.sha1(("active:" + row["uid"]).encode()).hexdigest())
    rows = rows[: args.per_operation * len(OPERATION_QUOTA)]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name if args.shard_count <= 1 else f"{out.stem}.shard_{args.shard_index:03d}{out.suffix}"
    target = out.parent / name
    log = (out.parent / (name + ".log")).open("a")
    done = set()
    if args.skip_existing and target.exists():
        done = {json.loads(line)["uid"] for line in target.read_text().splitlines() if line.strip()}

    written, counts = 0, {}
    for position, spec in enumerate(rows):
        if position % args.shard_count != args.shard_index:
            continue
        uid = str(spec["uid"])
        if uid in done:
            continue
        image_path = spec.get("image")
        if not image_path or not Path(image_path).exists():
            log.write(json.dumps(dict(uid=uid, skipped="no_image")) + "\n")
            log.flush()
            continue
        completion = read_image(image_path, rt)
        operation = next_operation(counts, OPERATION_QUOTA)
        raw_request = spec.get("raw_prompt") or ""
        # Ask for several operations in one call, then take the one the quota needs.
        proposals = propose_active_edit_multi(rt, completion, raw_request, cfg, count=5)
        requirements, _ = decompose(rt, raw_request, cfg)
        usable = []
        for candidate in proposals:
            if candidate.get("error") is not None:
                continue
            asked = candidate["edit_questions"]
            passed_edit, per_edit = score_questions(rt, completion, asked, cfg)
            if all(q in passed_edit for q in asked):
                continue                      # already true on this picture
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
            candidate["scores"] = {q: round(float(v), 3) for q, v in
                                   {**per_edit, **per_keep}.items()}
            usable.append(candidate)
        if not usable:
            log.write(json.dumps(dict(uid=uid, skipped="no_usable_operation",
                                      ops=[c.get("operation") for c in proposals])) + "\n")
            log.flush()
            continue
        # Only now spend the quota. An operation already over its share is
        # refused even when a valid candidate exists, otherwise the pool drifts
        # towards whichever operation the expert writes most easily.
        total = sum(counts.values()) + 1
        within = [
            c for c in usable
            if counts.get(c["operation"], 0) / total < OPERATION_QUOTA.get(c["operation"], 0)
        ]
        if not within:
            log.write(json.dumps(dict(uid=uid, skipped="quota_full",
                                      ops=[c["operation"] for c in usable])) + "\n")
            log.flush()
            continue
        wanted = next_operation(counts, OPERATION_QUOTA)
        proposal = next((c for c in within if c["operation"] == wanted), within[0])
        operation = proposal["operation"]
        row = dict(
            uid=uid,
            parent_uid=uid,
            operation=proposal["operation"],
            edit_instruction=proposal["instruction"],
            tail_condition=raw_request,      # active edit keeps the scene request
            parent_condition=raw_request,
            preserve="",
            edit_questions=list(proposal["edit_questions"]),
            preserve_questions=list(proposal["preserve_questions"]),
            source_image=image_path,
            completion_scores=proposal["scores"],
            task_type="active_edit",
            split=spec.get("split", "train"),
            source_family=spec.get("family") or hashlib.sha1(str(uid).encode()).hexdigest()[:12],
        )
        target.open("a").write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[row["operation"]] = counts.get(row["operation"], 0) + 1
        written += 1
    log.close()
    print(json.dumps(dict(shard=args.shard_index, written=written,
                          operations=counts, path=str(target))))


if __name__ == "__main__":
    main()
