"""Shared constants for Qwen-LatentCoT."""

# ---------------------------------------------------------------------------
# Special-token schema v3 (2026-05-31) — native multi-turn CoRT protocol
# ---------------------------------------------------------------------------
# The three dino_* entries are reserved but not emitted by v3.1. Keeping their
# insertion slots preserves embedding/lm-head shapes for existing checkpoints.
#
# Per-turn grammar:
#   <|cort_start|>
#     <|vlat_start|> <|vlat_pad|>×N <|vlat_end|>
#     <|cort_boa|> analysis <|cort_eoa|>
#     [ <|cort_bof|> fix <|cort_eof|> ]          # only when continuing
#     ( <|cort_continue|> -> next vlat | <|cort_end|> -> stop )
#   ...
#
# The continue/stop decision is an explicit single token (`<|cort_continue|>`
# vs `<|cort_end|>`), which makes the multi-turn branch directly supervisable
# (the old schema inferred it implicitly from whatever followed the fix).
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {
    "cort_start": "<|cort_start|>",
    "cort_end": "<|cort_end|>",
    "cort_continue": "<|cort_continue|>",
    "latent_pad": "<|vlat_pad|>",
    "latent_start": "<|vlat_start|>",
    "latent_end": "<|vlat_end|>",
    "boa": "<|cort_boa|>",
    "eoa": "<|cort_eoa|>",
    "bof": "<|cort_bof|>",
    "eof": "<|cort_eof|>",
    "dino_start": "<|dino_start|>",
    "dino_pad": "<|dino_pad|>",
    "dino_end": "<|dino_end|>",
}

# Insertion order used when registering tokens on the tokenizer.
SPECIAL_TOKEN_ORDER = (
    "cort_start",
    "cort_end",
    "cort_continue",
    "latent_pad",
    "latent_start",
    "latent_end",
    "boa",
    "eoa",
    "bof",
    "eof",
    "dino_start",
    "dino_pad",
    "dino_end",
)

IGNORE_TOKEN_ID = -100
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_CORT_TURNS = 16
DEFAULT_MAX_CORT_TOKENS = 4096
DEFAULT_TERMINAL_LOSS_WEIGHT = 0.2
DEFAULT_CORT_END_LOSS_WEIGHT = 0.1

REVIEW_SYSTEM_PROMPT = (
    "You are a strict prompt-to-image consistency reviewer.\n"
    "Your job is to judge whether the current image state satisfies the given prompt, "
    "and to point out only the mismatches that block prompt fidelity.\n\n"
    "Before listing mismatches, first ground yourself in what is actually in the image: "
    "name the concrete objects, counts, colors, and notable attributes you can see. "
    "Then compare that against the prompt and report only the discrepancies that matter.\n\n"
    "Check the image against the prompt using these criteria:\n"
    "- the main subject and required objects\n"
    "- counts, attributes, colors, materials, and identity cues\n"
    "- spatial relations, layout, and scene setting when explicitly requested\n"
    "- style or mood only when the prompt clearly requires them\n"
    "- missing, incorrect, or contradictory content\n\n"
    "Do not suggest optional beautification, stylistic upgrades, camera changes, "
    "or extra details that are not required by the prompt.\n"
    "Do not rewrite the whole prompt.\n"
    "If the image is already correct, state that there is no blocking mismatch and no change is needed.\n"
    "If the image is not correct, describe the concrete errors and give the smallest set of edits that would fix them.\n\n"
    "Always use exactly one of these two protocols, with the special tokens verbatim:\n"
    "If a blocking mismatch exists:\n"
    "<|cort_boa|>analysis<|cort_eoa|>"
    "<|cort_bof|>minimal fix<|cort_eof|><|cort_continue|>\n"
    "If no blocking mismatch exists:\n"
    "<|cort_boa|>analysis<|cort_eoa|><|cort_end|>\n"
    "Never emit a fix block for an already-correct image."
)
