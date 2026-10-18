"""Anchored Dual-State Loop: within-step GEN feedback on a frozen BAGEL."""

from .config import LoopConfig
from .loop import AnchoredDualStateLoop, LoopOutput

__all__ = ["AnchoredDualStateLoop", "LoopConfig", "LoopOutput"]
