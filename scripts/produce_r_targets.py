"""Produce one R-edit target per request with the frozen native IT2I branch.

Every request names the parent final image, the original prompt, and the
questions that failed on the parent preview. We re-render that image once under
the original prompt (optionally steered by the grounded edit request), then
score the result with the same questions so the target is approved only when it
fixes something without breaking something else. No seed search, one image per
request.
"""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.diagnosis import und_score_image
from qwen_latent_cot.u15_target_correction.images import load_canvas, save_image
from qwen_latent_cot.u15_target_correction.loading import load_native


def _prompt(request, mode):
    raw = " ".join(str(request.get("parent_condition") or "").split())
    edit = " ".join(str(request.get("edit_instruction") or "").split())
    target = " ".join(str(request.get("tail_condition") or "").split())
    if mode == "target" and target:
        return target
    if mode == "raw" or not edit:
        return raw
    return raw + "\nApply this correction while keeping everything else unchanged: " + edit


def _score(rt, image, questions, cfg):
    if not questions:
        return {}
    scores = und_score_image(rt, image, questions, None, None)
    per = scores["questions"]
    passed = [
        question for question, item in zip(questions, per)
        if item["p_yes"] >= float(cfg["pass_threshold"])
    ]
    return dict(
        questions=list(questions),
        passed=passed,
        p_yes={question: float(item["p_yes"]) for question, item in zip(questions, per)},
    )


def _explicit_approval(edit_questions, preserve_questions, before, after):
    """Strict rule for a specified edit: done, needed, and nothing damaged.

    ``edit_complete`` requires every edit question to pass on the target;
    ``edit_needed`` requires the source to fail at least one, so a no-op edit is
    rejected. Only preserve questions the source actually satisfied can be
    damaged.
    """
    passed_before = set(before.get("passed", []))
    passed_after = set(after.get("passed", []))
    edit_set = list(edit_questions)
    edit_complete = bool(edit_set) and set(edit_set) <= passed_after
    edit_needed = bool(edit_set) and not set(edit_set) <= passed_before
    protected = set(preserve_questions) & passed_before
    broken = sorted(protected - passed_after)
    return dict(
        edit_complete=edit_complete,
        edit_needed=edit_needed,
        repaired=sorted(set(edit_set) & (passed_after - passed_before)),
        broken=broken,
        protected=len(protected),
        approved=bool(edit_complete and edit_needed and not broken),
    )


def _approval(before_passed, after_passed):
    """Repaired and broken question sets for one edit target.

    ``before`` is the image that was actually edited (the parent final image),
    not the preview the diagnosis ran on: the preview and the final image can
    disagree about which requirements already hold.
    """
    before_passed, after_passed = set(before_passed), set(after_passed)
    repaired = sorted(after_passed - before_passed)
    broken = sorted(before_passed - after_passed)
    return dict(
        repaired=repaired,
        broken=broken,
        approved=bool(repaired) and not broken,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("raw", "raw_edit", "target"), default="raw_edit")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    device = set_device(args.device)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "targets"
    images_dir.mkdir(exist_ok=True)
    requests = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    if args.limit:
        requests = requests[: args.limit]
    targets = {}
    name = "targets.jsonl" if args.shard_count <= 1 else f"targets.shard_{args.shard_index:03d}.jsonl"
    existing = output / name
    if args.skip_existing and existing.exists():
        for line in existing.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                targets[row["uid"]] = row
    log = (output / "targets.log").open("a")
    seen = 0
    for position, request in enumerate(requests):
        if position % args.shard_count != args.shard_index:
            continue
        uid = str(request["uid"])
        if uid in targets:
            continue
        seen += 1
        began = time.perf_counter()
        source_image = request["source_image"]
        if not os.path.isabs(source_image):
            source_image = str(Path(args.requests).parent / source_image)
        with Image.open(source_image) as opened:
            rendered = rt.model.it2i_generate(
                rt.tokenizer,
                _prompt(request, args.mode),
                [opened.convert("RGB")],
                cfg_scale=float(cfg["cfg_scale"]),
                img_cfg_scale=float(cfg["image_cfg_scale"]),
                cfg_norm=cfg["cfg_norm"],
                image_size=(rt.width, rt.height),
                num_steps=int(cfg["num_steps"]),
                timestep_shift=float(cfg["timestep_shift"]),
                enable_timestep_shift=True,
                cfg_interval=tuple(float(t) for t in cfg["cfg_interval"]),
                t_eps=float(cfg["t_eps"]),
                think_mode=False,
                seed=int(request["seed"]),
            )
        if rendered.ndim == 3:
            rendered = rendered.unsqueeze(0)
        path = images_dir / f"{uid}_{args.mode}.png"
        save_image(path, rendered)
        source, _, _ = load_canvas(source_image, rt.width, rt.height, rt.device)
        task_type = request.get("task_type", "self_correct")
        if task_type == "explicit_edit":
            edit_questions = list(request.get("edit_questions") or [])
            preserve_questions = list(request.get("preserve_questions") or [])
            combined = list(dict.fromkeys(edit_questions + preserve_questions))
            before = _score(rt, source, combined, cfg)
            after = _score(rt, rendered, combined, cfg)
            verdict = _explicit_approval(edit_questions, preserve_questions, before, after)
            questions = combined
        else:
            questions = request.get("questions") or []
            before = _score(rt, source, questions, cfg)
            after = _score(rt, rendered, questions, cfg)
            verdict = _approval(before.get("passed", []), after.get("passed", []))
        preview_failed = set(request.get("failed_questions") or [])
        preview_agrees = preview_failed == set(before.get("questions", [])) - set(before.get("passed", []))
        row = dict(
            uid=uid,
            parent_branch_id=request["parent_branch_id"],
            seed=int(request["seed"]),
            edit_instruction=request["edit_instruction"],
            parent_condition=request.get("parent_condition"),
            tail_condition=request.get("tail_condition"),
            questions=list(questions),
            edit_questions=list(request.get("edit_questions") or []),
            preserve_questions=list(request.get("preserve_questions") or []),
            target_image=str(path),
            prompt_mode=args.mode,
            task_type=task_type,
            semantic_approved=verdict["approved"],
            edit_complete=verdict.get("edit_complete"),
            edit_needed=verdict.get("edit_needed"),
            repaired_questions=verdict.get("repaired", []),
            broken_questions=verdict.get("broken", []),
            passed_before_source=sorted(before.get("passed", [])),
            preview_failed_questions=sorted(preview_failed),
            preview_matches_source=bool(preview_agrees),
            p_yes_after=after.get("p_yes", {}),
            seconds=time.perf_counter() - began,
        )
        targets[uid] = row
        log.write(json.dumps(dict(uid=uid, mode=args.mode, approved=verdict["approved"],
                                  edit_complete=verdict.get("edit_complete"),
                                  edit_needed=verdict.get("edit_needed"),
                                  repaired=len(verdict.get("repaired", [])),
                                  broken=len(verdict.get("broken", [])),
                                  preview_matches_source=bool(preview_agrees),
                                  seconds=row["seconds"]), ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(dict(uid=uid, approved=verdict["approved"],
                              repaired=len(verdict.get("repaired", [])),
                              broken=len(verdict.get("broken", [])))), flush=True)
    log.close()
    partial = existing.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in targets.values()))
    partial.replace(existing)
    print(json.dumps(dict(requests=len(requests), produced=seen, approved=sum(
        1 for row in targets.values() if row["semantic_approved"]), path=str(existing))))


if __name__ == "__main__":
    main()
