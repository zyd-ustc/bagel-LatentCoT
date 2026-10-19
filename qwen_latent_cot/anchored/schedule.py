"""Sampler-progress schedule. BAGEL flow time decreases from noise to data."""
from __future__ import annotations

from .config import LoopConfig


def bagel_flow_time_to_progress(flow_time: float) -> float:
    """Map BAGEL ``t`` to progress ``u``.

    ``t = 1`` is pure noise and ``t = 0`` is data, so ``u = 1 - t``.
    A threshold on the raw ``t`` value points at the opposite end of sampling.
    """
    if not 0.0 <= float(flow_time) <= 1.0:
        raise ValueError("BAGEL flow time must be in [0, 1]")
    return 1.0 - float(flow_time)


def _progresses(num_steps: int) -> list[float]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_steps == 1:
        return [0.0]
    return [index / (num_steps - 1) for index in range(num_steps)]


def _contains(
    schedule: str,
    fraction: float,
    progress: float,
    *,
    start: float,
    end: float,
) -> bool:
    if schedule == "early":
        return progress <= fraction
    if schedule == "late":
        return progress >= 1.0 - fraction
    if schedule == "mid":
        half = fraction / 2.0
        return (0.5 - half) <= progress <= (0.5 + half)
    if schedule == "interval":
        return start <= progress <= end
    raise ValueError(f"schedule {schedule} needs a step index, not a single progress value")


def selected_steps(
    schedule: str,
    fraction: float,
    num_steps: int,
    *,
    start: float = 0.15,
    end: float = 0.65,
) -> list[int]:
    """Step indexes that receive a correction, counted in sampler progress."""
    progress = _progresses(num_steps)
    if schedule in ("early", "mid", "late", "interval"):
        return [
            index for index, value in enumerate(progress)
            if _contains(schedule, fraction, value, start=start, end=end)
        ]
    if schedule != "uniform":
        raise ValueError(f"unknown schedule {schedule}")
    # Match the early window's extra body calls, spread across the trajectory.
    target = len(selected_steps("early", fraction, num_steps))
    if target <= 0:
        return []
    if target >= num_steps:
        return list(range(num_steps))
    if target == 1:
        return [0]
    gap = (num_steps - 1) / (target - 1)
    return sorted({int(round(slot * gap)) for slot in range(target)})


def in_loop_schedule(
    config: LoopConfig,
    *,
    flow_time: float | None = None,
    step_index: int | None = None,
    num_steps: int | None = None,
) -> bool:
    """Return whether this sampling step runs a within-step correction.

    ``step_index`` is sampler progress and wins when both are supplied.
    ``flow_time`` alone is converted from the BAGEL direction.
    """
    if not config.enabled:
        return False
    if step_index is not None:
        if num_steps is None:
            raise ValueError("step_index requires num_steps")
        if not 0 <= step_index < num_steps:
            raise ValueError("step_index is outside the sampler")
        if config.schedule == "uniform":
            return step_index in selected_steps(
                config.schedule, config.schedule_fraction, num_steps,
                start=config.schedule_start, end=config.schedule_end,
            )
        progress = 0.0 if num_steps == 1 else step_index / (num_steps - 1)
        return _contains(
            config.schedule, config.schedule_fraction, progress,
            start=config.schedule_start, end=config.schedule_end,
        )
    if flow_time is None:
        raise ValueError("the loop schedule needs step_index or BAGEL flow time")
    if config.schedule == "uniform":
        raise ValueError("uniform schedule needs step_index and num_steps")
    return _contains(
        config.schedule, config.schedule_fraction, bagel_flow_time_to_progress(flow_time),
        start=config.schedule_start, end=config.schedule_end,
    )
