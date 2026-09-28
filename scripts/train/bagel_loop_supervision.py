#!/usr/bin/env python3
"""BAGEL v2 loop stage; see docs/experiments/memory_grounding_v2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen_latent_cot.bagel.memory_stage_runner import main

if __name__ == "__main__":
    main("loop")
