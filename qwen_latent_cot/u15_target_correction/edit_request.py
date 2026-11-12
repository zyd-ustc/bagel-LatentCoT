"""One atomic edit, shared by diagnosis and data production."""

from .conditions import compact
from .judge import _object

TASKS = ("count", "spatial", "attribute")
RELATIONS = ("left of", "right of", "in front of", "behind", "above", "below")
SCHEMA = (
    'Return one JSON object with task ("count", "spatial", or "attribute"), '
    'subject (an object or group phrase), and target. For count, target is the exact '
    'nonnegative integer. For spatial, target is one of left of, right of, in front of, '
    'behind, above, below, and reference names the other object. For attribute, '
    'target is the desired color, material, or other single attribute. '
    'Include preserve, a short phrase naming unrelated content to keep. '
    'Return {"task":"none"} if there is no clear applicable change.'
)


def parse_edit(reply):
    value = _object(reply) if isinstance(reply, str) else reply
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON edit object")
    task = value.get("task")
    if task == "none":
        return None
    subject = value.get("subject")
    target = value.get("target")
    if not isinstance(subject, str) or not compact(subject):
        raise ValueError("An edit needs a subject")
    subject = compact(subject)
    if task == "count" and type(target) is int and target >= 0:
        instruction = f"Set the number of {subject} to {target}."
    elif task == "spatial" and target in RELATIONS:
        reference = value.get("reference")
        if not isinstance(reference, str) or not compact(reference):
            raise ValueError("A spatial edit needs a reference object")
        instruction = f"Place {subject} {target} {compact(reference)}."
    elif task == "attribute" and isinstance(target, str) and compact(target):
        instruction = f"Make {subject} {compact(target)}."
    else:
        raise ValueError("Invalid atomic edit target")
    return dict(task=task, subject=subject, target=target, instruction=instruction,
                reference=value.get("reference"), preserve=compact(value.get("preserve")))
