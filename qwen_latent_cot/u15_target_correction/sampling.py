"""Read one native state, edit it at fixed t, and continue with frozen T2I."""

import torch

from .conditions import edit_prompt
from .diagnosis import review_preview
from .field import NativeT2I, edit_state, euler, phase_grid, preview_image, protocol, segment_times


@torch.no_grad()
def prepare(rt, raw, seed, cfg, *, diagnose=True):
    native = NativeT2I(rt, raw, cfg)
    grid = phase_grid(rt, cfg)
    read = float(cfg["read_time"])
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn(
        (1, 3, rt.height, rt.width), device=rt.device, generator=generator
    ) * rt.noise_scale()
    before = native.calls
    state = euler(noise.float(), segment_times(grid, 0.0, read), native)
    parent_calls = native.calls - before

    before_preview = native.calls
    preview = preview_image(state, read, grid, native)
    preview_calls = native.calls - before_preview
    if not (torch.isfinite(state).all() and torch.isfinite(preview).all()):
        raise RuntimeError("Native preparation produced a nonfinite state")

    if diagnose:
        review = review_preview(rt, preview, raw, cfg)
    else:
        review = dict(
            questions=[], decomposition={}, preview_score={"questions": []},
            failed=[], edit_request="", preserve="", rejected=[],
        )

    tail_condition = raw
    edit_request = review["edit_request"]
    condition = edit_prompt(edit_request, tail_condition, review["preserve"]) if edit_request else None
    record = dict(
        protocol=protocol(rt, cfg),
        raw_prompt=raw,
        seed=int(seed),
        read_time=read,
        parent_condition=raw,
        tail_condition=tail_condition,
        preview_score=review["preview_score"],
        decomposition=review["decomposition"],
        questions=review["questions"],
        failed_questions=review["failed"],
        rejected_questions=review["rejected"],
        edit_request=edit_request,
        preserve=review["preserve"],
        edit_condition=condition,
        early_exit=condition is None,
        nfe=dict(
            text_prefix=1 + int(native.negative is not None),
            parent_trajectory=parent_calls,
            preview=preview_calls,
            prefill=0,
            baseline=0,
            state_edit=0,
            tail_prefix=0,
            tail=0,
        ),
    )
    return native, grid, state, preview, record


@torch.no_grad()
def run(rt, raw, seed, cfg, *, editor=None, explicit=None):
    explicit = dict(explicit or {})
    native, grid, state, preview, record = prepare(
        rt, raw, seed, cfg, diagnose=not bool(explicit)
    )
    read = float(cfg["read_time"])
    images = {"preview": preview}

    if explicit:
        record.update(
            edit_request=explicit["edit_request"],
            tail_condition=explicit["tail_condition"],
            preserve=explicit.get("preserve", ""),
            early_exit=False,
            route="explicit_state_edit",
        )
        record["edit_condition"] = edit_prompt(
            record["edit_request"], record["tail_condition"], record["preserve"]
        )
    elif record["early_exit"]:
        record["route"] = "native_passthrough"
        before = native.calls
        result = euler(state, segment_times(grid, read, 1.0), native)
        record["nfe"]["tail"] = native.calls - before
        if not torch.isfinite(result).all():
            raise RuntimeError("Native continuation produced a nonfinite image")
        images["result"] = result
        return images, record
    else:
        record["route"] = "self_correct_state_edit"

    if editor is None:
        raise ValueError("An edit at the read state requires the state editor")

    before_prefix, before_baseline, before_state = (
        editor.prefix_calls,
        editor.baseline_calls,
        editor.state_calls,
    )
    prefixes = editor.prefixes(
        preview, record["edit_request"], record["tail_condition"], record["preserve"]
    )
    base = editor.image_clean(state, read, prefixes["image"])
    edited, _ = edit_state(
        editor,
        state,
        read,
        prefixes,
        int(cfg["edit_steps"]),
        base_prediction=base,
    )
    if not torch.isfinite(edited).all():
        raise RuntimeError("The state edit produced a nonfinite state")
    record["nfe"]["prefill"] = editor.prefix_calls - before_prefix
    record["nfe"]["baseline"] = editor.baseline_calls - before_baseline
    record["nfe"]["state_edit"] = editor.state_calls - before_state
    del prefixes

    if record["tail_condition"] == raw:
        tail = native
    else:
        tail = NativeT2I(rt, record["tail_condition"], cfg)
        record["nfe"]["tail_prefix"] = 1 + int(tail.negative is not None)
    before = tail.calls
    result = euler(edited, segment_times(grid, read, 1.0), tail)
    record["nfe"]["tail"] = tail.calls - before
    if not torch.isfinite(result).all():
        raise RuntimeError("Native continuation produced a nonfinite image")
    images["result"] = result
    return images, record
