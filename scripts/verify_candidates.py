"""Check GLM candidates against the parent picture they will be applied to.

The instruction writer only sees the prompt, so it can name an object the
generation never drew. This stage reads the parent's own completion and keeps a
candidate only when

* its edit question is false on that picture (the change is worth doing), and
* at least one preserve question is true on it (there is something to keep).

A candidate that fails is dropped; rewriting it against the picture is a later
step and needs an image-capable call.

Usage::

    python scripts/verify_candidates.py --requests <glm jsonl> --parents <cache> \
        --out <verified jsonl> --shard-index 0 --shard-count 16
"""

import argparse
import glob
import json
import hashlib
import re
from pathlib import Path

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.coupled import load_source, _read_final
from qwen_latent_cot.u15_target_correction.labels import score_questions
from qwen_latent_cot.u15_target_correction.loading import load_native


_SUBJECT = re.compile(
    r"^(?:remove|delete|erase|move|change the (?:colou?r|colour|material) of|recolou?r|"
    r"replace|turn)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:to|into|from|so|and)\b.*)?$",
    re.I,
)


def _subject_phrase(instruction):
    """The object an instruction operates on, for the existence probe."""
    text = (instruction or "").strip().rstrip(".")
    match = _SUBJECT.match(text)
    if not match:
        return None
    # Drop trailing destination clauses ("the brass tray onto the shelf" -> "the brass tray").
    phrase = re.split(r"\s+(?:onto|into|to|towards|behind|above|below|beside|next to|in front of)\s+",
                      match.group(1).strip())[0]
    phrase = phrase.strip()
    return phrase if 1 <= len(phrase.split()) <= 6 else None


def _load_by_hash(cache, uid, read_key, rt):
    """Fallback for caches whose markers predate the _safe naming."""
    import torch
    marker = Path(cache) / "progress" / (hashlib.sha1(uid.encode()).hexdigest()[:16] + ".json")
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    item = (record.get("states") or {}).get(read_key)
    if not item:
        return None
    payload = torch.load(item["path"], map_location="cpu", weights_only=False)
    sibling = {k: v["path"] for k, v in (record.get("states") or {}).items()
               if k != read_key and Path(v["path"]).exists()}
    final = Path(cache) / record.get("source_image", "") if record.get("source_image") else None
    return dict(
        state_a=payload["state_a"].float(), preview=payload["preview"].float(),
        final_image=str(final) if final and final.exists() else None,
        seed=int(record.get("seed") or 0), parent_branch_id=record.get("parent_branch_id"),
        sibling_states=sibling,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--parents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, set_device(args.device), False)
    key = "0.360000"

    paths = sorted(glob.glob(args.requests))
    if not paths and Path(args.requests).exists():
        paths = [args.requests]
    if not paths:
        raise SystemExit("no request file matched: " + args.requests)
    rows = []
    for path in paths:
        for line in open(path):
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    rows.sort(key=lambda r: hashlib.sha1(("verify:" + r["uid"]).encode()).hexdigest())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name if args.shard_count <= 1 else f"{out.stem}.shard_{args.shard_index:03d}{out.suffix}"
    target = out.parent / name
    log = (out.parent / (name + ".log")).open("a")
    written, rejected = 0, {}
    for position, row in enumerate(rows):
        if position % args.shard_count != args.shard_index:
            continue
        src = load_source(args.parents, row["uid"], key)
        if src is None:
            # A cache written before the naming fix stores <hash16>.json; accept
            # both so the 6,900 renders already on disk stay usable.
            src = _load_by_hash(args.parents, row["uid"], key, rt)
        if src is None or not src.get("final_image"):
            rejected["no_parent"] = rejected.get("no_parent", 0) + 1
            continue
        picture = _read_final(src["final_image"], rt.device)
        asked = row["edit_questions"] + row["preserve_questions"]
        passed, per = score_questions(rt, picture, asked, cfg)
        unknown = [q for q, v in per.items() if v is None]
        edit_true = all(q in passed for q in row["edit_questions"])
        protected = [q for q in row["preserve_questions"] if q in passed]
        reason = ("unreadable" if unknown else "edit_true" if edit_true
                  else "no_protected" if not protected else None)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            log.write(json.dumps(dict(uid=row["uid"], skipped=reason)) + "\n")
            log.flush()
            continue
        # The object being removed, moved or recoloured has to be findable on
        # this picture, otherwise the instruction refers to something else.
        subject = _subject_phrase(row["edit_instruction"])
        row = dict(row, preserve_questions=protected,
                   completion_scores={q: v for q, v in per.items()},
                   sibling_states=src.get("sibling_states") or {},
                   verified_on=str(src["final_image"]))
        if subject:
            probe = f"Is there {subject} in the image?"
            found, per_subject = score_questions(rt, picture, [probe], cfg)
            value = per_subject.get(probe)
            row["subject_probe"] = dict(question=probe, p=value)
            if value is None:
                rejected["subject_unreadable"] = rejected.get("subject_unreadable", 0) + 1
                continue
            if probe not in found and row["operation"] in ("remove", "move", "attribute"):
                # remove/move/attribute need an existing referent; add does not.
                rejected["subject_absent"] = rejected.get("subject_absent", 0) + 1
                continue
        # c_tail must describe the scene after the change, not restate clauses the
        # generation never satisfied.
        row["tail_is_post_edit"] = row["tail_condition"] != row["parent_condition"]
        target.open("a").write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
    log.close()
    print(json.dumps(dict(shard=args.shard_index, written=written, rejected=rejected)))


if __name__ == "__main__":
    main()
