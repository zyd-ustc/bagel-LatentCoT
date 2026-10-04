"""Step-based depth curriculum with explicit training coverage."""

import math

import torch


def resolve_depth_curriculum(curriculum, max_depth, total_steps):
    if total_steps < 1:
        raise ValueError("steps must be positive")
    if not curriculum:
        raise ValueError("depth curriculum must be nonempty")
    if all(isinstance(value, int) for value in curriculum):
        curriculum = [{"until": 1.0, "depths": curriculum}]
    phases, start, previous_until = [], 0, 0.0
    for phase in curriculum:
        until = float(phase["until"])
        depths = sorted(set(phase["depths"]))
        if not previous_until < until <= 1.0:
            raise ValueError("curriculum phase boundaries must increase to 1.0")
        if not depths or any(
            not isinstance(depth, int) or not 1 <= depth <= max_depth
            for depth in depths
        ):
            raise ValueError("depth curriculum must lie within max_train_loop_depth")
        end = math.ceil(until * total_steps)
        if end <= start:
            raise ValueError(
                "steps must give every curriculum phase at least one optimizer step"
            )
        phases.append({"start_step": start, "end_step": end, "depths": depths})
        start, previous_until = end, until
    if previous_until != 1.0:
        raise ValueError("last curriculum boundary must be 1.0")
    if max(depth for phase in phases for depth in phase["depths"]) != max_depth:
        raise ValueError(
            "max_train_loop_depth must equal the maximum trained depth; reduce the training maximum or extend the curriculum"
        )
    return phases


def sample_curriculum_depth(phases, step):
    previous_max = 0
    for index, phase in enumerate(phases):
        if phase["start_step"] <= step < phase["end_step"]:
            depths = phase["depths"]
            # Exercise a newly introduced depth immediately, including short
            # jobs. Subsequent steps sample uniformly from the current set.
            if step == phase["start_step"] and max(depths) > previous_max:
                return max(depths), index
            return depths[int(torch.randint(len(depths), ()).item())], index
        previous_max = max(previous_max, max(phase["depths"]))
    raise ValueError("step lies outside the curriculum")
