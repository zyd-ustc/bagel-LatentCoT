from __future__ import annotations

import torch
import pytest

pytest.importorskip("einops")

from qwen_latent_cot.bagel.modeling.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2RotaryEmbedding,
    ROPE_INIT_FUNCTIONS,
)


def test_default_rope_works_without_transformers_registry_entry(monkeypatch) -> None:
    monkeypatch.delitem(ROPE_INIT_FUNCTIONS, "default", raising=False)
    config = Qwen2Config(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )

    rotary = Qwen2RotaryEmbedding(config=config)

    assert rotary.inv_freq.shape == (2,)
    assert torch.isfinite(rotary.inv_freq).all()
    assert rotary.attention_scaling == 1.0
