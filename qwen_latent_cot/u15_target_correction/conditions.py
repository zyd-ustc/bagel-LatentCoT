"""The frozen text template shared by training, cache production, and inference."""


def compact(text):
    return " ".join(str(text or "").split())


def edit_prompt(edit_request):
    """The editor's only text condition: the short instruction, nothing else.

    No target scene, no preserve list, no preview caption. Deployment hands the
    diagnoser's short instruction and the state; anything more here would be
    information the closed loop never has, so the editor would learn to read a
    condition that disappears at test time.
    """
    request = compact(edit_request)
    if not request:
        raise ValueError("An edit request is required")
    return "Edit request: " + request
