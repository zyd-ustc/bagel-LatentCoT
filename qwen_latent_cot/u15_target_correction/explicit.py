"""Explicit edit requests for real parent trajectories.

Self-correction waits for the diagnosis to find a failure, which fires on only a
few percent of prompts and caps how much real-entry edit data exists. This
module specifies the edit instead: given the preview the editor would actually
see, the understanding expert proposes one checkable change, the description
that follows from it, and two separate question sets.

``edit_questions`` cover what this round must achieve (they should be false
before the edit and true after). ``preserve_questions`` cover what the edit must
not damage, drawn from the parent request minus what the edit changes; only the
ones the source image already satisfies are used for the damage check.
"""

import re

from .conditions import compact
from .diagnosis import ask_with_image, question_grounded

SYSTEM = (
    "You design exactly one concrete image edit for the scene in the attached "
    "image, then state the target description as a MINIMAL update of the source "
    "description.\n"
    "Answer with four blocks and nothing else:\n"
    "EDIT: <imperative, at most 12 words, naming one visible change that this "
    "image currently does not show>\n"
    "TARGET: <the source description with only the edited clause replaced; copy "
    "every other word of the source description verbatim, at most 60 words>\n"
    "EDIT_QUESTIONS: <exactly one yes/no question, starting with '- ', true only "
    "after the edit and false for the current image>\n"
    "PRESERVE_QUESTIONS: <1 to 3 yes/no questions about content of the source "
    "description that this edit does not touch>\n"
    "Rules:\n"
    "Change one thing only: add or remove one object, change one attribute, or "
    "change one spatial relation.\n"
    "Do not add clothing, props, actions, viewpoints, materials or mood that the "
    "source description does not already contain.\n"
    "PRESERVE_QUESTIONS must not ask about anything the edit changes.\n"
    "Base the edit on what the image actually shows, not only on the request."
)

_RULES = (
    "One change only. EDIT must be false for the current image. TARGET must keep "
    "the source description's other content word for word. One EDIT question only."
)

_QUESTION_BLOCK = re.compile(
    r"^(EDIT_QUESTIONS?|EDIT_QS?|PRESERVE_QUESTIONS?|PRESERVE_QS?)\s*[::]?\s*(.*)$", re.I
)
_FIELD = re.compile(r"^(EDIT|TARGET)\s*[::]\s*(.+)$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _question_text(line):
    """One question, cut at its first question mark.

    The expert often appends an explanation after the question, for example
    ``Is the mug blue? (false before the edit)``. Keeping only the part up to
    the question mark preserves the question and drops the annotation.
    """
    text = _BULLET.sub("", line).strip()
    text = re.sub(r"^Q\d*\s*[.:)]\s*", "", text, flags=re.I)
    text = compact(text)
    if not text:
        return None
    ends = [index for index in (text.find("?"), text.find("？")) if index >= 0]
    if not ends:
        return None
    return compact(text[: min(ends) + 1])


def parse_proposal(text):
    """Split the labelled blocks; empty values mean the reply was unusable."""
    instruction = target = None
    edit_questions, preserve_questions = [], []
    block = None
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        header = _QUESTION_BLOCK.match(stripped)
        if header:
            block = "edit" if header.group(1).lower().startswith("edit") else "preserve"
            inline = _question_text(header.group(2)) if header.group(2).strip() else None
            if inline:
                (edit_questions if block == "edit" else preserve_questions).append(inline)
            continue
        field = _FIELD.match(stripped)
        if field:
            block = None
            if field.group(1).lower() == "edit":
                instruction = compact(field.group(2))
            else:
                target = compact(field.group(2))
            continue
        if block:
            question = _question_text(stripped)
            if question:
                (edit_questions if block == "edit" else preserve_questions).append(question)
    return dict(
        instruction=instruction,
        target_condition=target,
        edit_questions=_dedupe(edit_questions),
        preserve_questions=_dedupe(preserve_questions),
    )


def _dedupe(items):
    seen, out = set(), []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def proposal_error(raw, instruction, target_condition):
    """Why this proposal cannot be used, or ``None`` when it can."""
    raw_text = compact(raw)
    if not instruction or not target_condition:
        return "unparsed"
    if not 2 <= len(instruction.split()) <= 16:
        return "instruction_length"
    if instruction.rstrip().endswith("?") or instruction.lower().startswith("keep"):
        return "instruction_form"
    if not 6 <= len(target_condition.split()) <= 80:
        return "target_length"
    if compact(target_condition).casefold() == raw_text.casefold():
        return "target_equals_request"
    if compact(instruction).casefold() in compact(target_condition).casefold():
        return "target_repeats_instruction"
    return target_rewrite_error(raw, target_condition)


STOPWORDS = frozenset(
    "the and with from into onto over under that this these those there their its "
    "his her they them then than while which what where when who whom whose are was "
    "were been being have has had does did doing very more most some such only also".split()
)


def _tokens(text):
    return {
        word for word in re.findall(r"[a-z0-9]+", compact(text).casefold())
        if len(word) > 2 and word not in STOPWORDS
    }


def target_rewrite_error(source, target, *, min_recall=0.5):
    """A minimal update keeps the source description's content words."""
    source_tokens = _tokens(source)
    if not source_tokens:
        return None
    kept = source_tokens & _tokens(target)
    return None if len(kept) / len(source_tokens) >= min_recall else "target_rewrite"


def question_sets_error(edit_questions, preserve_questions):
    """Why the two question sets cannot carry the acceptance, or ``None``."""
    if not edit_questions:
        return "no_edit_questions"
    if len(edit_questions) > 2:
        return "too_many_edit_questions"
    if not preserve_questions:
        return "no_preserve_questions"
    for question in edit_questions + preserve_questions:
        if not 4 <= len(question.split()) <= 40:
            return "question_length"
    edit_keys = {question.casefold() for question in edit_questions}
    if edit_keys & {question.casefold() for question in preserve_questions}:
        return "question_overlap"
    return None


def edit_question_grounded(target, question):
    """An edit question is grounded when its content words appear in the target.

    The strict relational grounding used for diagnosis rejects questions whose
    subject phrase carries a determiner ("...the bench") even when the target
    says "park bench"; for a single edit question that is too brittle, so the
    check is a content-word overlap instead.
    """
    words = _tokens(question)
    if not words:
        return False
    return len(words & _tokens(target)) >= max(1, len(words) // 3)


def filter_questions(proposal, raw):
    """Drop questions that are not grounded in the text they claim to describe."""
    edit_kept = [
        question for question in proposal["edit_questions"]
        if edit_question_grounded(proposal["target_condition"] or "", question)
    ]
    preserve_kept = [
        question for question in proposal["preserve_questions"]
        if question_grounded(raw, question)
    ]
    return dict(proposal, edit_questions=edit_kept, preserve_questions=preserve_kept,
                dropped_edit=len(proposal["edit_questions"]) - len(edit_kept),
                dropped_preserve=len(proposal["preserve_questions"]) - len(preserve_kept))


def propose_edit(rt, preview, raw, cfg, *, require_preserve=True):
    """Ask the understanding expert, looking at the preview the editor would see."""
    question = (
        "USER REQUEST:\n" + compact(raw) + "\n\n"
        "The attached image is the current state of this scene at generation time.\n"
        "Propose one change this image needs, the description after the change, and "
        "the two question sets.\nFINAL RULES:\n" + _RULES
    )
    answer = ask_with_image(rt, preview, question, cfg, system_message=SYSTEM)
    proposal = parse_proposal(answer)
    proposal["raw_output"] = answer
    proposal = filter_questions(proposal, raw)
    error = proposal_error(raw, proposal["instruction"], proposal["target_condition"])
    if error is None and require_preserve:
        error = question_sets_error(proposal["edit_questions"], proposal["preserve_questions"])
    proposal["error"] = error
    return proposal


def build_request(*, uid, seed, parent_branch_id, source_image, raw, instruction,
                  target_condition, edit_questions, preserve_questions, preserve=""):
    """One row in the shape the target producer and the R producer expect."""
    return dict(
        uid=str(uid),
        parent_branch_id=str(parent_branch_id),
        seed=int(seed),
        source_image=str(source_image),
        parent_condition=compact(raw),
        edit_instruction=compact(instruction),
        tail_condition=compact(target_condition),
        preserve=compact(preserve),
        edit_questions=list(edit_questions),
        preserve_questions=list(preserve_questions),
        task_type="explicit_edit",
    )

# ---------------------------------------------------------------------------
# Repair requests
#
# ``propose_edit`` lets the expert choose a change. Repair needs the opposite:
# the failing requirement is not a free choice, it has to come from the user's
# own request, and the fix has to be the one that removes *that* violation.
# Both entry points share the parsing and validation above.
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = (
    "You audit the partially generated image against the user's original "
    "request and name ONE requirement that is currently violated.\n"
    "Answer with four blocks and nothing else:\n"
    "FAILED: <the specific requirement from the request that this image gets "
    "wrong, at most 12 words>\n"
    "EDIT: <imperative fix, at most 12 words, false for the current image>\n"
    "TARGET: <the original request, unchanged>\n"
    "EDIT_QUESTIONS: <exactly one yes/no question, true only after the fix and "
    "false for the current image>\n"
    "PRESERVE_QUESTIONS: <1 to 3 yes/no questions about requirements of the "
    "original request that this fix does not touch>\n"
    "Rules:\n"
    "The failing requirement must come from the original request; do not invent "
    "a new one.\n"
    "Name the failure mode: wrong count (too many or too few), a missing object, "
    "an extra object, a swapped attribute, or a wrong spatial relation.\n"
    "If the count is wrong, EDIT must say whether to add or to remove, and by how "
    "many.\n"
    "If an attribute belongs to the wrong subject, EDIT must move it, not "
    "duplicate it.\n"
    "TARGET must be the original request verbatim."
)

REPAIR_RULES = (
    "One violated requirement only. EDIT must repair it and be false for the "
    "current image. TARGET is the original request, unchanged."
)

_REVERSE_STEP = re.compile(r"\b(add|increase|remove|delete)\s+(\d+)\b", re.I)


def propose_repair(rt, preview, raw, cfg, *, requirement=None, require_preserve=True):
    """Ask the understanding expert what the current picture gets *wrong*.

    ``requirement`` may pin one clause (for example "exactly five laptops") so
    the call becomes a verdict on that clause rather than an open audit.
    """
    focus = ""
    if requirement:
        focus = "FOCUS ON THIS REQUIREMENT:\n" + compact(requirement) + "\n\n"
    question = (
        "USER REQUEST:\n" + compact(raw) + "\n\n" + focus +
        "The attached image is the current state of this scene at generation time.\n"
        "Which ONE requirement of the request does this image break?\n"
        "FINAL RULES:\n" + REPAIR_RULES
    )
    answer = ask_with_image(rt, preview, question, cfg, system_message=REPAIR_SYSTEM)
    proposal = parse_proposal(answer)
    proposal["raw_output"] = answer
    proposal = filter_questions(proposal, raw)
    error = proposal_error(raw, proposal["instruction"], proposal["target_condition"])
    if error is None and require_preserve:
        error = question_sets_error(proposal["edit_questions"], proposal["preserve_questions"])
    # The repair target is the original request; a rewrite would let the expert
    # do the editor's job for it.
    if error is None and compact(proposal["target_condition"]) != compact(raw):
        proposal["target_condition"] = compact(raw)
    proposal["error"] = error
    return proposal


def reverse_instruction(task, forward_prompt):
    """A real edit instruction for the flipped image pair.

    A reversed pair must never be trained with "keep the scene unchanged": the
    target picture differs from the start picture, so that pairing would teach
    the model that subjects may disappear without being asked. ``task`` selects
    the wording; anything unmapped returns ``None`` and the pair is dropped.
    """
    text = compact(forward_prompt)
    if task == "T8":
        found = _REVERSE_STEP.search(text)
        if not found:
            return None
        verb, step = found.group(1).lower(), found.group(2)
        if verb in ("add", "increase"):
            return f"Remove {step}, keeping the same scene otherwise unchanged."
        return f"Add {step} back, keeping the rest of the scene unchanged."
    if task == "T9":
        return "Undo the rearrangement: restore the objects' previous spatial relation."
    if task == "T3":
        return "Put the removed object back where it was and blend it into the scene."
    if task == "T1":
        return "Restore the original text in the same place, font and colour."
    return None
