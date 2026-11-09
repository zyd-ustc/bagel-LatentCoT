"""The frozen text template shared by training, cache production, and inference."""


def compact(text):
    return " ".join(str(text or "").split())


def edit_prompt(edit_request, tail_condition, preserve=""):
    """Text seen by the joint UND prefix. It never contains target pixels."""
    request = compact(edit_request)
    target = compact(tail_condition)
    if not request:
        raise ValueError("An edit request is required")
    if not target:
        raise ValueError("A full generation target is required")
    lines = [
        "Edit request: " + request,
        "Generation target after editing: " + target,
    ]
    kept = compact(preserve)
    if kept:
        lines.append("Preserve: " + kept)
    return "\n".join(lines)


def derived_tail_condition(parent_condition, edit_request):
    """Explicit legacy fallback for pairs without a full target description.

    This concatenation can retain old semantics; formal L/I data must provide
    an independent ``tail_condition``. The loader only reaches this function
    when ``allow_derived_tail`` is set explicitly.
    """
    parent = compact(parent_condition)
    request = compact(edit_request)
    if not parent:
        raise ValueError("A parent scene description is required")
    if not request:
        raise ValueError("An edit request is required")
    return parent + "\nApply this edit to the scene: " + request
