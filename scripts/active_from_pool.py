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
from qwen_latent_cot.u15_target_correction.explicit import OPERATION_QUOTA, propose_active_edit_from_request
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
        proposal = propose_active_edit_from_request(rt, completion, raw_request, cfg,
                                                    operation=operation)
        if proposal.get("error") == "operation_mismatch":
            # One retry on the same picture with the operation restated; the
            # expert drifts towards "add" and this is the cheapest fix.
            retry_op = next_operation(counts, OPERATION_QUOTA)
            proposal = propose_active_edit_from_request(rt, completion, raw_request, cfg,
                                                        operation=retry_op)
            operation = retry_op
        if proposal.get("error") is None:
            # Preserve set comes from the request text, not from a description of
            # the picture: the same model judging its own description rejected it
            # on 50 of 50 smoke attempts.
            requirements, _ = decompose(rt, raw_request, cfg)
            proposal["preserve_questions"] = [
                q for q in requirements
                if not any(w in proposal["instruction"].lower() for w in q.lower().split() if len(w) > 4)
            ][:3]
        if proposal.get("error"):
            log.write(json.dumps(dict(uid=uid, operation=proposal.get("operation"),
                                      skipped=proposal["error"],
                                      out=(proposal.get("raw_output") or "")[:1200]),
                                 ensure_ascii=False) + "\n")
            log.flush()
            continue
        # The edit question has to be false on the picture the expert just read.
        asked = proposal["edit_questions"] + proposal["preserve_questions"]
        passed, per = score_questions(rt, completion, asked, cfg)
        edit_true = all(q in passed for q in proposal["edit_questions"])
        protected = [q for q in proposal["preserve_questions"] if q in passed]
        if edit_true or not protected:
            log.write(json.dumps(dict(uid=uid, operation=proposal["operation"],
                                      skipped="edit_true" if edit_true else "no_protected",
                                      scores={q: round(float(v), 3) for q, v in per.items()}),
                                 ensure_ascii=False) + "\n")
            log.flush()
            continue
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
            completion_scores={q: round(float(v), 3) for q, v in per.items()},
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
