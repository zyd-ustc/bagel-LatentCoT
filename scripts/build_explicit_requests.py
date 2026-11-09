"""Write explicit edit requests from real parent trajectories.

Each request reuses one cached parent trajectory: the parent state ``a``, its
preview, seed and branch stay untouched, while the understanding expert looks at
that preview and supplies one concrete edit, the description that should follow,
and the two question sets. Feeding these rows back through the target producer
builds R-edit records without depending on how often the diagnosis happens to
fail.

The pool is every recorded parent trajectory that carries a preview, which
includes scenes whose original prompt was *not* satisfied: the edit is grounded
in the image, not in the prompt's correctness. Records keep their diagnosis
verdict as ``preview_status`` so the two cases can be separated later.
"""

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.diagnosis import decompose
from qwen_latent_cot.u15_target_correction.explicit import (
    build_request, propose_edit, question_sets_error,
)
from qwen_latent_cot.u15_target_correction.images import load_canvas
from qwen_latent_cot.u15_target_correction.labels import score_questions
from qwen_latent_cot.u15_target_correction.loading import load_native


def _preview(record, device, root):
    """The preview the editor would see at the read time, loaded from the cache."""
    states = record.get("states") or {}
    if not states:
        return None
    key = sorted(states)[0]
    path = states[key].get("path")
    if not path:
        return None
    resolved = Path(path) if Path(path).is_absolute() else Path(root) / path
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    preview = payload.get("preview")
    return None if preview is None else preview.float().to(device)


def _cache_records(root):
    files = sorted(glob.glob(str(root / "manifest.shard_*.json"))) or sorted(glob.glob(str(root / "manifest.json")))
    if not files:
        raise ValueError("No shard manifest in " + str(root))
    records = []
    for path in files:
        records.extend(json.loads(Path(path).read_text()).get("records", []))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--include-failed", action="store_true",
                        help="Also draw from parents whose diagnosis failed (default: passed only)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    root = Path(args.cache).resolve()
    records = _cache_records(root)
    wanted = {"R_keep"} if not args.include_failed else {"R_keep", "R_edit"}
    pool = [record for record in records if record.get("role") in wanted and record.get("source_image")]
    pool.sort(key=lambda record: hashlib.sha1(("explicit:" + str(record["uid"])).encode()).hexdigest())
    pool = pool[: args.limit]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    name = output.name if args.shard_count <= 1 else f"{output.stem}.shard_{args.shard_index:03d}{output.suffix}"
    target = output.parent / name
    log = (output.parent / (name + ".log")).open("a")
    written = 0
    done = set()
    if args.skip_existing and target.exists():
        for line in target.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uid"])
    for position, record in enumerate(pool):
        if position % args.shard_count != args.shard_index:
            continue
        uid = str(record["uid"])
        if uid in done:
            continue
        began = time.perf_counter()
        raw = record.get("parent_condition") or ""
        preview = _preview(record, rt.device, root)
        if preview is None:
            log.write(json.dumps(dict(uid=uid, skipped="no_preview"), ensure_ascii=False) + "\n")
            log.flush()
            continue
        proposal = propose_edit(rt, preview, raw, cfg, require_preserve=False)
        if proposal.get("error"):
            log.write(json.dumps(dict(uid=uid, skipped=proposal["error"],
                                      output=(proposal.get("raw_output") or "")[:300]),
                                 ensure_ascii=False) + "\n")
            log.flush()
            continue
        # An edit question that already holds on the source cannot define a change.
        source_canvas, _, _ = load_canvas(str(root / record["source_image"]),
                                          rt.width, rt.height, rt.device)
        already_true, _ = score_questions(rt, source_canvas, proposal["edit_questions"], cfg)
        proposal["edit_questions"] = [
            question for question in proposal["edit_questions"] if question not in already_true
        ]
        if not proposal["edit_questions"]:
            log.write(json.dumps(dict(uid=uid, skipped="edit_question_true_on_source"),
                                 ensure_ascii=False) + "\n")
            log.flush()
            continue
        # Preserve set: decompose the *source* request and keep what the source
        # image already satisfies. That is what "do not damage" can mean; the
        # model's own preserve questions are only a fallback.
        try:
            source_questions, _ = decompose(rt, raw, cfg)
        except ValueError:
            source_questions = []
        keepable, _ = score_questions(rt, source_canvas, source_questions, cfg)
        preserve = [q for q in source_questions if q in keepable]
        preserve = [q for q in preserve if q not in set(proposal["edit_questions"])]
        proposal["preserve_questions"] = preserve or proposal["preserve_questions"]
        sets_error = question_sets_error(proposal["edit_questions"], proposal["preserve_questions"])
        if sets_error:
            log.write(json.dumps(dict(uid=uid, skipped=sets_error,
                                      output=(proposal.get("raw_output") or "")[:300]),
                                 ensure_ascii=False) + "\n")
            log.flush()
            continue
        row = build_request(
            uid=uid,
            seed=int(record.get("source_seed_or_linear_noise_seed") or 0),
            parent_branch_id=record.get("parent_branch_id") or f"{record.get('source_seed_or_linear_noise_seed')}:t2i",
            source_image=str(root / record["source_image"]),
            raw=raw,
            instruction=proposal["instruction"],
            target_condition=proposal["target_condition"],
            edit_questions=proposal["edit_questions"],
            preserve_questions=proposal["preserve_questions"],
        )
        row["preview_status"] = record.get("diagnosis_status")
        row["dropped_questions"] = dict(edit=proposal.get("dropped_edit", 0),
                                        preserve=proposal.get("dropped_preserve", 0))
        row["seconds"] = round(time.perf_counter() - began, 2)
        with target.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
        log.write(json.dumps(dict(uid=uid, written=True, seconds=row["seconds"]), ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(dict(uid=uid, edit=row["edit_instruction"][:60])), flush=True)
    print(json.dumps(dict(pool=len(pool), written=written, path=str(target))))


if __name__ == "__main__":
    main()
