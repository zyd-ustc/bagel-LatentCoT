"""Loop configuration for the anchored dual-state experiment line."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


FEEDBACK_MODES = ("correct", "zero", "mean", "shuffled", "frozen")
SCHEDULES = ("early", "mid", "late", "uniform")
STATE_MODES = ("anchored", "direct_hidden")
VARIANTS = ("dual", "gen_only")
OBSERVERS = ("verified_gen_hidden_to_und",)


@dataclass
class LoopConfig:
    """Controls one within-step correction loop.

    ``entry_scale`` is the inference gate. ``0`` is the native bypass.
    Training uses the learned gate parameter instead of this constant.
    ``[body_start, body_end)`` is a candidate span, not a proven optimum.
    """

    enabled: bool = False
    body_start: int = 20
    body_end: int = 28
    correction_rounds: int = 1
    schedule: str = "early"
    schedule_fraction: float = 0.35
    entry_scale: float = 0.0
    max_delta_rms_ratio: float = 0.02
    observer: str = "verified_gen_hidden_to_und"
    feedback_mode: str = "correct"
    state_mode: str = "anchored"
    variant: str = "dual"
    slots: int = 16
    rank: int = 32
    writer_position_start: int = 2048
    lambda_anchor: float = 0.01
    lambda_entry: float = 0.001
    log_intermediates: bool = False
    force_full_zero_path: bool = False
    verify_prompt_cache: bool = True

    def validate(self, depth: int) -> None:
        if not 0 <= self.body_start < self.body_end <= depth:
            raise ValueError(
                f"body span [{self.body_start}, {self.body_end}) does not fit depth {depth}"
            )
        if self.correction_rounds not in (1, 2):
            raise ValueError("the first line trains 1 correction and may extend to 2")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {self.schedule}")
        if not 0.0 < self.schedule_fraction <= 1.0:
            raise ValueError("schedule_fraction must be in (0, 1]")
        if self.entry_scale < 0.0:
            raise ValueError("entry_scale must be non-negative")
        if self.max_delta_rms_ratio <= 0.0:
            raise ValueError("max_delta_rms_ratio must be positive")
        if self.observer not in OBSERVERS:
            raise ValueError(f"unknown observer {self.observer}")
        if self.feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"unknown feedback mode {self.feedback_mode}")
        if self.state_mode not in STATE_MODES:
            raise ValueError(f"unknown state mode {self.state_mode}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant}")
        if self.slots <= 0 or self.rank <= 0:
            raise ValueError("slots and rank must be positive")
        if self.writer_position_start < 0:
            raise ValueError("writer positions must be non-negative")
        if self.lambda_anchor < 0.0 or self.lambda_entry < 0.0:
            raise ValueError("loss weights must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "LoopConfig":
        known = {field for field in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown loop config fields: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "LoopConfig":
        payload = json.loads(Path(path).read_text())
        payload.pop("experiment", None)
        payload.pop("native_weights", None)
        payload.pop("rejects_legacy_memory_sft_checkpoints", None)
        return cls.from_dict(payload)
