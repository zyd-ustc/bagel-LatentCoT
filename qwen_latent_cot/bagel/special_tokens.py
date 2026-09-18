"""BAGEL special-token registration (v3 native multi-turn CoRT protocol).

BAGEL exposes a raw `Qwen2Tokenizer`, so this shim:

1. ensures the BAGEL built-in special tokens (`<|im_start|>`, `<|im_end|>`,
   `<|vision_start|>`, `<|vision_end|>`) exist for model compatibility;
2. registers the CoRT schema and reserved checkpoint-compatibility tokens;
3. resolves a `BagelSpecialTokenIds` bundle that downstream code
   (collator + loss + trainer + probe) consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qwen_latent_cot.constants import SPECIAL_TOKEN_ORDER, SPECIAL_TOKENS


_BAGEL_CHAT_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


@dataclass
class BagelSpecialTokenIds:
    im_start: int
    im_end: int
    vision_start: int
    vision_end: int
    cort_start: int
    cort_end: int
    cort_continue: int
    latent_start: int
    latent_end: int
    latent_pad: int
    boa: int
    eoa: int
    bof: int
    eof: int
    answer_start_pattern: List[int]


def add_bagel_special_tokens(tokenizer) -> int:
    """Register BAGEL chat tokens and the checkpoint-stable CoRT token schema.

    Returns the number of newly added tokens.
    """
    before = len(tokenizer)

    chat_to_add = []
    existing = set()
    special_map = tokenizer.special_tokens_map or {}
    for value in special_map.values():
        if isinstance(value, str):
            existing.add(value)
        elif isinstance(value, list):
            existing.update(value)
    for token in _BAGEL_CHAT_TOKENS:
        if token not in existing:
            chat_to_add.append(token)
    if chat_to_add:
        tokenizer.add_tokens(chat_to_add, special_tokens=True)

    for key in SPECIAL_TOKEN_ORDER:
        tokenizer.add_tokens(SPECIAL_TOKENS[key], special_tokens=True)

    return len(tokenizer) - before


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids, f"tokenizer returned no ids for {text!r}"
    return int(ids[-1])


def resolve_bagel_special_token_ids(tokenizer) -> BagelSpecialTokenIds:
    answer_start = tokenizer.encode(SPECIAL_TOKENS["cort_start"], add_special_tokens=False)
    return BagelSpecialTokenIds(
        im_start=_single_token_id(tokenizer, "<|im_start|>"),
        im_end=_single_token_id(tokenizer, "<|im_end|>"),
        vision_start=_single_token_id(tokenizer, "<|vision_start|>"),
        vision_end=_single_token_id(tokenizer, "<|vision_end|>"),
        cort_start=_single_token_id(tokenizer, SPECIAL_TOKENS["cort_start"]),
        cort_end=_single_token_id(tokenizer, SPECIAL_TOKENS["cort_end"]),
        cort_continue=_single_token_id(tokenizer, SPECIAL_TOKENS["cort_continue"]),
        latent_start=_single_token_id(tokenizer, SPECIAL_TOKENS["latent_start"]),
        latent_end=_single_token_id(tokenizer, SPECIAL_TOKENS["latent_end"]),
        latent_pad=_single_token_id(tokenizer, SPECIAL_TOKENS["latent_pad"]),
        boa=_single_token_id(tokenizer, SPECIAL_TOKENS["boa"]),
        eoa=_single_token_id(tokenizer, SPECIAL_TOKENS["eoa"]),
        bof=_single_token_id(tokenizer, SPECIAL_TOKENS["bof"]),
        eof=_single_token_id(tokenizer, SPECIAL_TOKENS["eof"]),
        answer_start_pattern=[int(t) for t in answer_start],
    )
