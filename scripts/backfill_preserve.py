"""Back-fill the preserve set of a repair request from its own requirement list.

``propose_repair`` sometimes returns a perfectly good instruction and edit
question but no preserve questions. That is a second-stage gap: the repair is
usable, and what it must not damage can be derived without looking at the image
again. This script decomposes the original request into checkable requirements,
drops the one the repair already covers, and keeps the rest as candidates. The
caller still has to verify them against the source image before trusting them.

Usage::

    python scripts/backfill_preserve.py --requests <repair requests> \\
        --out <jsonl> --shard-index 0 --shard-count 16
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.diagnosis import decompose
from qwen_latent_cot.u15_target_correction.loading import load_native


def _tokens(text):
    return {word.strip(".,;:!?").lower() for word in str(text).split() if len(word) > 3}


def overlaps(question, instruction, threshold=0.45):
    """True when a requirement is (partly) what the repair itself changes."""
    shared = _tokens(question) & _tokens(instruction)
    return len(shared) / max(1, len(_tokens(instruction))) >= threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True, help="repair request jsonl or a dir of shards")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)

    if Path(args.requests).is_dir():
        paths = sorted(glob.glob(str(Path(args.requests) / "requests*.shard_*.jsonl")))
    else:
        paths = sorted(glob.glob(str(args.requests)))
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in open(path) if line.strip()]
    rows.sort(key=lambda row: hashlib.sha1(("preserve:" + row["uid"]).encode()).hexdigest())
    rows = rows[: args.limit]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.name if args.shard_count <= 1 else f"{out.stem}.shard_{args.shard_index:03d}{out.suffix}"
    target = out.parent / name
    log = (out.parent / (name + ".log")).open("a")
    done = set()
    if args.skip_existing and target.exists():
        done = {json.loads(line)["uid"] for line in target.read_text().splitlines() if line.strip()}

    written = 0
    for position, row in enumerate(rows):
        if position % args.shard_count != args.shard_index:
            continue
        if row["uid"] in done or not row.get("needs_preserve_backfill"):
            continue
        questions, _ = decompose(rt, row["parent_condition"], cfg)
        wanted = [q for q in questions if not overlaps(q, row["edit_instruction"])]
        if not wanted:
            log.write(json.dumps(dict(uid=row["uid"], skipped="no_candidate_preserve")) + "\n")
            log.flush()
            continue
        row = dict(row, preserve_questions=wanted[:3],
                   preserve_source="decomposed_request", preserve_verified=False)
        target.open("a").write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
    log.close()
    print(json.dumps(dict(shard=args.shard_index, written=written, path=str(target))))


if __name__ == "__main__":
    main()
