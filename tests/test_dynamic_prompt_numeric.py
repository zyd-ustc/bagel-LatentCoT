from __future__ import annotations

import torch

from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import NaiveCache


def test_prompt_value_residual_is_exactly_affine_and_anchor_is_immutable():
    cache = NaiveCache(2)
    cache.key_cache[1] = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    cache.value_cache[1] = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    anchor = cache.value_cache[1].clone()
    delta = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])
    prompt_index = torch.tensor([1])

    plus = cache.with_value_residuals(
        start_layer=1,
        prompt_indexes=prompt_index,
        residuals=(delta,),
        alpha=0.25,
    )
    minus = cache.with_value_residuals(
        start_layer=1,
        prompt_indexes=prompt_index,
        residuals=(delta,),
        alpha=-0.25,
    )

    torch.testing.assert_close(cache.value_cache[1], anchor, rtol=0, atol=0)
    torch.testing.assert_close(
        plus.value_cache[1][1] - anchor[1], 0.25 * delta[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        plus.value_cache[1][1] - anchor[1],
        -(minus.value_cache[1][1] - anchor[1]),
        rtol=0,
        atol=0,
    )


def test_prompt_prefill_metadata_stays_aligned_after_append():
    cache = NaiveCache(1)
    cache.record_prompt_segment(
        torch.tensor([2, 3]), torch.tensor([0, 1]), torch.tensor([], dtype=torch.long)
    )
    cache.record_prompt_segment(
        torch.tensor([7]), torch.tensor([2]), torch.tensor([0, 1])
    )

    torch.testing.assert_close(
        cache.position_ids, torch.tensor([2, 3, 7]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        cache.prompt_mask, torch.tensor([True, True, True]), rtol=0, atol=0
    )
