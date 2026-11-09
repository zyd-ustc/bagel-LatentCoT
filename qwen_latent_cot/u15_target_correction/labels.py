"""Shared label acceptance: what counts as a usable (a, b) pair.

The semantic verdict is always computed on the *continued* image, ``x_eff`` —
what the frozen tail actually renders from the label state — never on a
candidate image that some editor proposed.
"""

from .diagnosis import und_score_image
from .loss import high_frequency as _per_sample_high_frequency


def score_questions(rt, image, questions, cfg):
    """Which yes/no questions the given image passes."""
    questions = [q for q in questions if q]
    if not questions:
        return set(), {}
    # Module-level import kept as the seam tests replace; production reaches the
    # generated-answer scorer through it.
    # Pass the config positionally-friendly form: the scorer replaced it with a
    # keyword-only cfg, older stubs accept yes_ids/no_ids. Fall back so both work.
    try:
        scores = und_score_image(rt, image, questions, cfg=cfg)
    except TypeError:
        scores = und_score_image(rt, image, questions)
    per = scores["questions"]
    passed = {
        question for question, item in zip(questions, per)
        if float(item["p_yes"]) >= float(cfg["pass_threshold"])
    }
    return passed, {q: float(item["p_yes"]) for q, item in zip(questions, per)}


def high_frequency(image):
    """Laplacian energy; a texture proxy for human review, never a gate."""
    return float(_per_sample_high_frequency(image.detach().float()).mean())


def effective_acceptance(rt, before_image, effective, request, cfg):
    """Judge the continued image against the request's two question sets.

    ``before_image`` is the image that would have been produced without this
    round's change (the same trajectory's continuation), so both sides of the
    comparison are outputs of the same model.
    """
    edit_questions = list(request.get("edit_questions") or [])
    preserve_questions = list(request.get("preserve_questions") or [])
    questions = list(request.get("questions") or (edit_questions + preserve_questions))
    # Repairs are judged with the same two-set rule as explicit edits: the
    # edit question must be false before and true after, and the requirements
    # the fix does not touch must survive. The self-correction branch below has
    # no notion of "protected", which would reject every repair.
    explicit = request.get("task_type", "explicit_edit") in ("explicit_edit", "repair")
    after, p_yes = score_questions(rt, effective, questions, cfg)
    before = {str(q) for q in (request.get("passed_before_source") or [])}
    if not before:
        before, _ = score_questions(rt, before_image, questions, cfg)
    if explicit:
        edit_set = set(edit_questions)
        complete = bool(edit_set) and edit_set <= after
        needed = bool(edit_set) and not edit_set <= before
        protected = set(preserve_questions) & before
        broken = sorted(protected - after)
        approved = complete and needed and not broken
        reason = None if approved else (
            "edit_incomplete" if not complete else "edit_not_needed" if not needed else "preserve_broken"
        )
        return dict(approved=approved, reason=reason, edit_complete=complete, edit_needed=needed,
                    broken_questions=broken, protected=len(protected), p_yes_after=p_yes)
    repaired = sorted((after - before) & set(questions))
    broken = sorted((before & set(questions)) - after)
    approved = bool(repaired) and not broken
    return dict(approved=approved,
                reason=None if approved else ("no_repair" if not repaired else "prompt_broken"),
                repaired_questions=repaired, broken_questions=broken, p_yes_after=p_yes)


def quality_record(effective, before_image, candidate=None):
    """Diagnostics for the separate form/texture review; never a gate."""
    record = dict(pending_review=True,
                  hf_effective=high_frequency(effective),
                  hf_before=high_frequency(before_image))
    if candidate is not None:
        record["l1_vs_candidate"] = float((effective - candidate).abs().mean())
    return record

