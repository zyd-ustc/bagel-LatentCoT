"""Which GEN KV a format-4 checkpoint actually trained the writer to read.

The stage string is not the source. ``visual_objectives.training_objective``
returns on ``row['task'] == 'observe'`` before any prompt is passed, including
when the saved stage is C2-feedback or C3-loop. ``CoverageSampler`` forces
that single task for C2-observe and otherwise keeps ``task_mix`` entries whose
loss weight is active. Endpoint and trajectory rows are materialized as
``feedback`` and, on C2-feedback, call ``write(..., prompt=prompt_kv)``.
Every later non-observe task goes through ``memory_loop``, which always
passes prompt KV. Those two writes are ``prompt_and_gen``.

``prompt_and_gen`` is a named mode so a config can ask for it. The reader
does not concatenate prompt KV, so that request fails closed.
"""
from __future__ import annotations

from .config import SOURCE_MODES


def active_training_tasks(config: dict) -> dict[str, float]:
    """Tasks ``CoverageSampler`` would draw, with inactive losses removed."""
    if not isinstance(config, dict):
        raise ValueError("observation training config must be a dict")
    try:
        stage = config["stage"]
        weights = config["loss_weights"]
        requested = {"observe": 1.0} if stage == "C2-observe" else dict(config["task_mix"])
    except KeyError as error:
        raise ValueError("observation training config is missing " + str(error.args[0])) from error
    if not isinstance(weights, dict):
        raise ValueError("observation training config is missing loss_weights")
    for task in list(requested):
        try:
            # Same activation rule as CoverageSampler.__init__.
            active = (
                weights["sem"] if task == "observe"
                else weights["sem"] + weights["teacher"] if task in ("endpoint", "trajectory")
                else weights["flow"]
            )
        except KeyError as error:
            raise ValueError(
                "observation training config is missing loss weight " + str(error.args[0])
            ) from error
        if not active:
            requested[task] = 0
    return {str(task): float(weight) for task, weight in requested.items() if weight > 0}


def trained_source_modes(config: dict) -> frozenset[str]:
    """Writer sources this config trained. An active non-observe task saw prompt KV."""
    tasks = active_training_tasks(config)
    modes: set[str] = set()
    if "observe" in tasks:
        modes.add("gen_full_query")
    if any(task != "observe" for task in tasks):
        modes.add("prompt_and_gen")
    if not modes:
        raise ValueError("observation training config activates no writer task")
    return frozenset(modes)


def require_source_mode(saved: dict, requested_mode: str) -> dict:
    """Accept a checkpoint only when its training tasks are exactly the wired reader.

    The parent path is recorded and is not used to choose the mode.
    """
    if requested_mode not in SOURCE_MODES:
        raise ValueError(f"unknown observation source mode {requested_mode}")
    if requested_mode != "gen_full_query":
        raise ValueError(
            "observation_source_mode prompt_and_gen is reserved and not wired. "
            "The reader does not concatenate prompt KV onto GEN KV."
        )
    config = saved.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "observation checkpoint has no training config; "
            "source mode cannot be inferred from the stage name"
        )
    if config.get("stage") != saved.get("stage"):
        raise ValueError("checkpoint stage disagrees with its training config stage")
    policy = config.get("visual_source_policy")
    if policy != "full_query":
        raise ValueError(
            "observation reader requires visual_source_policy full_query; "
            f"checkpoint trained with {policy}"
        )
    depth = saved.get("depth")
    layers = config.get("source_layers")
    if isinstance(layers, (str, bytes)) or not isinstance(layers, (list, tuple)) or not isinstance(depth, int):
        raise ValueError("observation training config must read GEN KV at every layer")
    if [int(index) for index in layers] != list(range(depth)):
        raise ValueError("observation training config must read GEN KV at every layer")
    trained = trained_source_modes(config)
    if trained != frozenset({"gen_full_query"}):
        found = ", ".join(sorted(trained))
        raise ValueError(
            f"checkpoint training sources are {found}; refusing to read them as gen_full_query"
        )
    return {
        "trained_sources": ["gen_full_query"],
        "resolved_mode": "gen_full_query",
        "stage": saved["stage"],
        "parent": saved.get("parent"),
    }
