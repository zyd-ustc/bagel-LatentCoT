# Copyright 2025 CoRT Project.
# SPDX-License-Identifier: Apache-2.0

"""BAGEL model components, ported from https://github.com/bytedance/Bagel.

Package layout:
    bagel/      - Core BAGEL model (BagelConfig, Bagel, NaiveCache, etc.)
    qwen2/      - Qwen2 tokenizer and config
    siglip/     - SigLIP vision encoder
    cache_utils/ - TaylorSeer KV cache
    autoencoder  - VAE encoder/decoder
"""

from .bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from .bagel.qwen2_navit import NaiveCache
from .autoencoder import load_ae

__all__ = [
    "BagelConfig",
    "Bagel",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "SiglipVisionConfig",
    "SiglipVisionModel",
    "NaiveCache",
    "load_ae",
]
