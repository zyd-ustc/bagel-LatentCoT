"""Coupled pairs for R-edit: reuse the real source state, build the target forward.

Two target-branch constructions are supported.

``text``  — the earlier same-noise route: both branches run from one noise draw
            under their own text, so only the noise is shared. It produces
            native-quality images but the target branch never sees the scene, so
            composition drifts; keep it only as a comparison.

``it2i``  — the image-conditioned teacher: the target branch starts from the
            *source scene* (the deployment-visible preview), runs the frozen
            native image-conditioned path from the same initial noise up to the
            read time, and the deployment tail then continues that state under
            ``c_tail``. Whole target branch is forward-only; there is no
            inversion and no external editor.

In both cases ``a`` and ``P`` are read from the cache record the request was
built from, so the request, the preview and the paired state are the same scene
by construction instead of assuming a replay reproduces it.
"""

import json
import time
from pathlib import Path

import torch

from .conditions import compact, edit_prompt
from .field import (
    NativeT2I, euler, native_forward_states, phase_grid, segment_times, time_key,
)
from .labels import effective_acceptance, quality_record, score_questions
from .teachers import teacher_state

# Questions that already hold on the source image cannot define "what this round
# changes"; a pair whose protected set is empty cannot show damage either.
MIN_PROTECTED = 1


def run_trajectory(rt, cfg, prompt, noise, grid, state_times, read_time):
    """One frozen native forward from a given initial noise (text route)."""
    velocity = NativeT2I(rt, prompt, cfg)
    return dict(velocity=velocity, **native_forward_states(noise, grid, state_times, velocity))


def continue_from(rt, cfg, state, prompt, grid, read_time):
    """Deployment continuation of a state under a given condition."""
    velocity = NativeT2I(rt, prompt, cfg)
    return euler(state.float(), segment_times(grid, read_time, 1.0), velocity)


def load_source(cache, uid, read_key):
    """The cached source state, preview and final image for one uid.

    A parent rendered at several read times carries one state per time. The
    teacher runs at one of them, but the record must keep every sibling so the
    same target can supervise the other times later.
    """
    from .prepare_state_pairs import _safe

    marker = Path(cache) / "progress" / f"{_safe(uid)}.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    item = (record.get("states") or {}).get(read_key)
    if not item:
        return None
    payload = torch.load(Path(cache) / item["path"], map_location="cpu", weights_only=False)
    final = Path(cache) / record.get("source_image", "") if record.get("source_image") else None
    sibling = {}
    for key, other in (record.get("states") or {}).items():
        if key == read_key:
            continue
        path = Path(cache) / other["path"]
        if path.exists():
            sibling[key] = str(path)
    return dict(
        state_a=payload["state_a"].float(),
        preview=payload["preview"].float(),
        final_image=str(final) if final and final.exists() else None,
        seed=int(record.get("source_seed_or_linear_noise_seed") or 0),
        parent_branch_id=record.get("parent_branch_id"),
        sibling_states=sibling,
    )


def _read_final(path, device):
    import numpy as np
    from PIL import Image

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device)


def _explicit_acceptance(rt, before, effective, request, cfg):
    """Two-question-set rule, with an explicit guard on the protected set."""
    verdict = effective_acceptance(rt, before, effective, request, cfg)
    if verdict["approved"] and int(verdict.get("protected", 0)) < MIN_PROTECTED:
        return dict(verdict, approved=False, reason="no_usable_preserve_questions")
    return verdict


def build_pair(rt, cfg, grid, request, source, *, teacher="it2i", teacher_image="preview",
               teacher_prompt="joint"):
    """Target state for one request, using the cached source state as the scene."""
    read = float(cfg["read_time"])
    key = time_key(read)
    parent = compact(request["parent_condition"])
    task_type = request.get("task_type", "explicit_edit")
    seed = source["seed"] or int(request.get("seed", int(cfg["seed"])))
    generator = torch.Generator(device=rt.device).manual_seed(seed)
    noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()

    a = source["state_a"].to(rt.device)
    preview = source["preview"].to(rt.device)
    before = (
        _read_final(source["final_image"], rt.device) if source.get("final_image")
        else euler(a.float(), segment_times(grid, read, 1.0), NativeT2I(rt, parent, cfg))
    )

    # A repair is an instructed edit too: the teacher has to see the very
    # instruction the expert derived, otherwise it falls into the text-only
    # rewrite branch and never renders the specific fix.
    # Cheap rejections first: they need neither the teacher nor the tail.
    questions = list(request.get("edit_questions") or []) + list(request.get("preserve_questions") or [])
    passed_before, _ = score_questions(rt, before, questions, cfg)
    edit_set = set(request.get("edit_questions") or [])
    if edit_set and edit_set <= passed_before:
        return None, "edit_not_needed"      # the request already holds: nothing to repair
    if task_type in ("explicit_edit", "repair") and not (
        set(request.get("preserve_questions") or []) & passed_before
    ):
        return None, "no_protected_requirement"   # nothing could be shown to survive

    if task_type in ("explicit_edit", "repair", "active_edit"):
        tail = compact(request["tail_condition"])
        if teacher == "text":
            velocity = NativeT2I(rt, tail, cfg)
            b = euler(noise.float(), segment_times(grid, 0.0, read), velocity)
            calls = velocity.calls
        else:
            image = preview if teacher_image == "preview" else before
            if teacher_prompt == "instruction":
                instruction = compact(request["edit_instruction"])
            elif teacher_prompt == "target":
                # Same text as the tail: only the image condition differs at the
                # handoff, so a texture jump cannot come from a text mismatch.
                instruction = tail
            else:
                instruction = edit_prompt(
                    request["edit_instruction"], tail, request.get("preserve", "")
                )
            b, calls = teacher_state(rt, cfg, instruction, image, noise, grid, read)
        effective = euler(b.float(), segment_times(grid, read, 1.0), NativeT2I(rt, tail, cfg))
    else:
        from .teachers import propose_teacher  # self-correction target prefix
        proposal = propose_teacher(rt, parent, cfg)
        if proposal["error"]:
            return None, proposal["error"]
        tail = parent
        velocity = NativeT2I(rt, proposal["condition"], cfg)
        b = euler(noise.float(), segment_times(grid, 0.0, read), velocity)
        calls = velocity.calls
        effective = euler(b.float(), segment_times(grid, read, 1.0), NativeT2I(rt, tail, cfg))

    return dict(a=a, b=b.float(), preview=preview, before=before.float(),
                effective=effective.float(), calls=calls, seed=seed, tail=tail,
                passed_before=sorted(passed_before)), None


def produce(rt, cfg, args):
    """Write a state-pair cache of coupled pairs, one shard per process."""
    from .data import read_rows
    from .images import save_image
    from .prepare_state_pairs import (
        _cache_header, _safe, _save_payload, _write_jsonl, _write_manifest,
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    requests = read_rows(args.requests)
    grid = phase_grid(rt, cfg)
    read = float(cfg["read_time"])
    key = time_key(read)
    header = _cache_header(rt, cfg, "prepare-state-pairs/coupled", args)
    progress = output / "progress"
    progress.mkdir(exist_ok=True)
    continuations = output / "continuations"
    continuations.mkdir(exist_ok=True)
    log = output / "production.jsonl"
    rows, written = [], 0
    limit = getattr(args, "limit", None)
    total = min(len(requests), int(limit)) if limit else len(requests)
    teacher = getattr(args, "teacher", "it2i")
    teacher_image = getattr(args, "teacher_image", "preview")
    teacher_prompt = getattr(args, "teacher_prompt", "joint")
    for index in range(args.shard_index, total, args.shard_count):
        request = requests[index]
        uid = str(request["uid"])
        marker = progress / f"{_safe(uid)}.json"
        if getattr(args, "skip_existing", False) and marker.exists():
            rows.append(json.loads(marker.read_text()))
            continue
        began = time.perf_counter()
        source = load_source(args.source_cache, uid, key)
        if source is None:
            _write_jsonl(log, dict(uid=uid, skipped="source_state_missing"))
            continue
        pair, error = build_pair(
            rt, cfg, grid, request, source,
            teacher=teacher, teacher_image=teacher_image, teacher_prompt=teacher_prompt,
        )
        if pair is None:
            _write_jsonl(log, dict(uid=uid, skipped=error))
            continue
        # build_pair already scored the source image; do not pay for it twice.
        verdict = _explicit_acceptance(
            rt, pair["before"], pair["effective"],
            dict(request, passed_before_source=pair["passed_before"]), cfg,
        )
        reason = None if verdict["approved"] else "effective_" + str(verdict["reason"])
        finite = bool(
            torch.isfinite(pair["a"]).all() and torch.isfinite(pair["b"]).all()
            and torch.isfinite(pair["effective"]).all()
        )
        if reason is None and not finite:
            reason = "nonfinite_state"
        accepted = reason is None
        ctx = _safe(uid)
        save_image(continuations / f"{ctx}_source.png", pair["before"])
        save_image(continuations / f"{ctx}_effective.png", pair["effective"])
        valid = torch.ones(1, 1, rt.height, rt.width, device=rt.device)
        payload = dict(
            time=read, state_a=pair["a"].float().cpu(), state_b=pair["b"].cpu(),
            preview=pair["preview"].cpu(), base_prediction=None, valid=valid.float().cpu(),
            source_reconstruction={}, target_reconstruction={}, accepted=accepted,
        )
        state_file = _save_payload(output, uid, read, payload)
        record = dict(
            uid=uid, source_kind="native_parent", split=request.get("split", "train"),
            parent_branch_id=f"{pair['seed']}:coupled",
            # The role decides which loss trains this record, so it has to come
            # from the task, not from a constant.
            role={"repair": "R_repair", "active_edit": "R_active"}.get(
                request.get("task_type", "explicit_edit"), "R_edit"),
            task_type=request.get("task_type", "explicit_edit"),
            diagnosis_status="coupled_pair",
            teacher=teacher,
            teacher_image=teacher_image,
            teacher_calls=pair["calls"],
            source_family=request.get("source_family", uid),
            prompt_family=request.get("prompt_family", uid),
            source_image=str(continuations / f"{ctx}_source.png"),
            target_image=str(continuations / f"{ctx}_effective.png"),
            edit_instruction=compact(request["edit_instruction"]),
            parent_condition=compact(request["parent_condition"]),
            tail_condition=pair["tail"],
            preserve=compact(request.get("preserve")),
            # The acceptance questions have to travel with the record: without
            # them a produced pair cannot be re-judged, and the first teacher pass
            # lost all 674 of them this way.
            edit_questions=list(request.get("edit_questions") or []),
            preserve_questions=list(request.get("preserve_questions") or []),
            edit_box=request.get("edit_box"),
            preserve_boxes=request.get("preserve_boxes"),
            condition_protocol=cfg["condition_protocol"],
            valid_region=None,
            source_seed_or_linear_noise_seed=pair["seed"],
            states={key: dict(
                path=state_file, accepted=accepted, reject_reason=reason,
                effective=verdict, effective_quality=quality_record(pair["effective"], pair["before"]),
                source_reconstruction={}, target_reconstruction={},
            )},
            source_reconstruction_metrics={}, target_reconstruction_metrics={},
            semantic_quality_source="coupled_forward",
            target_quality_source="coupled_forward",
        )
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        rows.append(record)
        written += 1
        _write_jsonl(log, dict(uid=uid, accepted=accepted, reason=reason,
                               teacher_calls=pair["calls"],
                               seconds=round(time.perf_counter() - began, 1)))
    path = _write_manifest(output, header, rows,
                           args.shard_index if args.shard_count > 1 else None)
    return path, written
