"""Turn produced parent trajectories into repair requests.

``build_explicit_requests`` asks the expert to *choose* one change; that is why
the pool it produced was 98% additions and never asked the model to undo
anything. This script asks the opposite question, against the scene the parent
trajectory actually rendered:

* what does the user's own request say that this picture gets wrong?
* what is the single instruction that removes that violation?
* which requirements does the fix have to leave alone?

Only requests whose focus requirement is verifiably false on the native
completion are kept, so the training pair never asks for a repair of something
that was never broken.

Usage::

    python scripts/repair_requests.py --parent <parent cache> --out <jsonl> \
        --limit 6000 --shard-index 0 --shard-count 16
"""

import argparse
import json
import glob
import hashlib
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.explicit import propose_repair
from qwen_latent_cot.u15_target_correction.loading import load_native


def parent_records(root):
    """Every produced parent, with its source completion path."""
    records = []
    for path in sorted(glob.glob(str(Path(root) / "manifest.shard_*.json"))) or \
            sorted(glob.glob(str(Path(root) / "manifest.json"))):
        records.extend(json.loads(Path(path).read_text()).get("records", []))
    return records


def completion_of(record, root, device):
    """The parent's own final picture: what the edit has to end up fixing."""
    from qwen_latent_cot.u15_target_correction.coupled import _read_final

    path = record.get("source_image")
    if not path:
        return None
    resolved = Path(path) if Path(path).is_absolute() else Path(root) / path
    if not resolved.exists():
        return None
    return _read_final(str(resolved), device)


def preview_of(record, root, device):
    states = record.get("states") or {}
    if not states:
        return None
    key = sorted(states)[0]
    item = states[key]
    path = Path(item.get("path"))
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        return None
    preview = torch.load(path, map_location="cpu", weights_only=False).get("preview")
    return None if preview is None else preview.float().to(device)


def focus_requirement(raw, target_count, family):
    """The clause the audit should verify, when the family makes it checkable."""
    if family in ("over", "count", "under") and target_count:
        return f"the image contains exactly {int(target_count)} of the requested objects"
    if family == "space":
        return "the stated left/right or above/below relation"
    if family == "attr":
        return "each stated attribute belongs to its own subject"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--prompts", required=True, help="the targeted prompt jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--observe", choices=("completion", "preview"), default="completion",
                        help="completion: judge the final picture (offline labels). "
                             "preview: the deployment view, which also produces the "
                             "false-trigger negatives.")
    parser.add_argument("--review", default="", help="directory for the split-by-verdict review files")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    root = Path(args.parent).resolve()
    prompts = {}
    for line in open(args.prompts):
        row = json.loads(line)
        prompts[row["uid"]] = row

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name if args.shard_count <= 1 else f"{out.stem}.shard_{args.shard_index:03d}{out.suffix}"
    target = out.parent / name
    log = (out.parent / (name + ".log")).open("a")
    review_path = Path(args.review) / name if args.review else out.parent / (name + ".review.jsonl")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review = review_path.open("a")
    done = set()
    if args.skip_existing and target.exists():
        for line in target.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uid"])

    records = parent_records(root)
    records.sort(key=lambda r: hashlib.sha1(("repair:" + str(r["uid"])).encode()).hexdigest())
    records = records[: args.limit]
    written = 0
    for position, record in enumerate(records):
        if position % args.shard_count != args.shard_index:
            continue
        uid = str(record["uid"])
        if uid in done:
            continue
        spec = prompts.get(uid)
        if spec is None:
            log.write(json.dumps(dict(uid=uid, skipped="prompt_missing")) + "\n")
            log.flush()
            continue
        if args.observe == "completion":
            observed = completion_of(record, root, rt.device)
            subject = "completion"
        else:
            observed = preview_of(record, root, rt.device)
            subject = "preview"
        if observed is None:
            log.write(json.dumps(dict(uid=uid, skipped="no_" + subject)) + "\n")
            log.flush()
            continue
        began = time.perf_counter()
        raw = record.get("parent_condition") or spec.get("raw_prompt")
        research = focus_requirement(raw, spec.get("target_count"), spec.get("family"))
        proposal = propose_repair(rt, observed, raw, cfg, requirement=research)
        error = proposal.get("error")
        # A missing preserve set is a second-stage gap, not a bad repair: the
        # instruction and its checkable question are already usable, and the
        # preserve set can be back-filled offline by decomposing the request.
        needs_backfill = error == "no_preserve_questions"
        if error and not needs_backfill:
            log.write(json.dumps(dict(uid=uid, skipped=error,
                                      out=(proposal.get("raw_output") or "")[:200]),
                                 ensure_ascii=False) + "\n")
            log.flush()
            continue
        if needs_backfill:
            log.write(json.dumps(dict(uid=uid, backfill="preserve_questions"),
                                 ensure_ascii=False) + "\n")
            log.flush()
        row = dict(
            uid=uid,
            source_family=spec.get("family"),
            family=spec.get("family"),
            target_count=spec.get("target_count"),
            parent_condition=raw,
            edit_instruction=proposal["instruction"],
            tail_condition=raw,
            preserve="",
            edit_questions=list(proposal["edit_questions"]),
            preserve_questions=list(proposal["preserve_questions"]),
            failed=list(proposal.get("failed") or []),
            source_image=record.get("source_image"),
            needs_preserve_backfill=bool(needs_backfill),
            observed=subject,
            task_type="repair",
            seconds=round(time.perf_counter() - began, 1),
        )
        # Writing a question about a picture does not make the picture fail it.
        # Score the completion here so the downstream gate is not the first time
        # anyone checks; false triggers are kept as diagnosis negatives.
        from qwen_latent_cot.u15_target_correction.coupled import _read_final
        from qwen_latent_cot.u15_target_correction.labels import score_questions

        final_path = record.get("source_image")
        resolved = Path(final_path) if Path(final_path).is_absolute() else root / str(final_path)
        if not resolved.exists():
            log.write(json.dumps(dict(uid=uid, skipped="no_completion")) + "\n")
            log.flush()
            continue
        completion = _read_final(str(resolved), rt.device)
        asked = row["edit_questions"] + row["preserve_questions"]
        passed_completion, per_completion = score_questions(rt, completion, asked, cfg)
        row["completion_scores"] = {q: round(float(v), 4) for q, v in per_completion.items()}
        row["edit_true_on_completion"] = bool(row["edit_questions"]) and all(
            q in passed_completion for q in row["edit_questions"]
        )
        row["protected_on_completion"] = sorted(q for q in row["preserve_questions"] if q in passed_completion)
        if row["edit_true_on_completion"] or not row["protected_on_completion"]:
            # The completion already satisfies the requirement, or nothing
            # could be shown to survive: a real failure to feed the editor.
            review.write(json.dumps(row, ensure_ascii=False) + "\n")
            review.flush()
            log.write(json.dumps(dict(uid=uid, skipped="false_trigger",
                                      edit_true=row["edit_true_on_completion"],
                                      protected=len(row["protected_on_completion"]))) + "\n")
            log.flush()
            continue
        target.open("a").write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
    log.close()
    review.close()
    print(json.dumps(dict(shard=args.shard_index, written=written, path=str(target),
                          review=str(review_path))))


if __name__ == "__main__":
    main()
