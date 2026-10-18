"""Keep the old Memory-SFT line from starting unless it is explicitly unlocked."""
from __future__ import annotations

import os


LEGACY_UNLOCK = "BAGEL_ALLOW_LEGACY_MEMORY_SFT"


def refuse_legacy_memory_sft(entry: str) -> None:
    if os.environ.get(LEGACY_UNLOCK) == "1":
        return
    raise RuntimeError(
        f"{entry} is the retired Observation-Memory-SFT route. "
        "The active line is Anchored Dual-State Loop: "
        "python -m qwen_latent_cot.anchored.cli self-check. "
        f"Set {LEGACY_UNLOCK}=1 only to reproduce a historical run."
    )


def refuse_legacy_checkpoint(blob: dict) -> None:
    """ADSL checkpoints contain only the gate, adapter, and feedback updater."""
    if not isinstance(blob, dict):
        raise ValueError("anchored checkpoint must be a dict")
    if blob.get("experiment") != "anchored_dual_state_loop":
        raise ValueError("refusing a checkpoint from outside the anchored experiment line")
    weights = blob.get("model", {})
    leaked = [
        key for key in weights
        if "moe_gen" in key or key.startswith("layers.") or key.startswith("writer")
        or "language_model" in key or key.startswith("und")
    ]
    if leaked:
        raise ValueError("refusing native or Memory-SFT weights in an anchored checkpoint: " + leaked[0])
