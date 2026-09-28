"""BAGEL T0 Text-CoT cache contract and conservative local quality gate."""

import hashlib
import json
import re
from pathlib import Path


TEMPLATE_VERSION = "bagel_t0_v1"
SECTIONS = ("Objects", "Counts", "Attributes", "Spatial relations",
            "Actions", "Global layout", "Critical constraints")
REFUSALS = ("i cannot", "i can't", "i'm sorry", "as an ai", "cannot assist")
SEMANTIC_CATEGORIES = frozenset(("count", "spatial_relation", "attribute_binding",
    "multi_object_composition", "action_relation", "rare_concept", "reasoning_heavy_t2i"))
NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def teacher_instruction(prompt):
    return (f"Prompt: {prompt}\n\nBefore image generation, write a concise scene plan "
            "using exactly these seven headings, each on its own line: "
            + ", ".join(SECTIONS) + ". Preserve every object, count, attribute, "
            "spatial and action relation. Do not add objects or contradict the prompt. "
            "Use 80 to 160 tokens total. Return only the scene plan.")


def teacher_condition(prompt, reasoning):
    return f"{prompt.strip()}\n\nGeneration plan:\n{reasoning.strip()}"


def _numbers(text):
    result = set()
    for token in re.findall(r"\b(?:\d+|[a-z]+)\b", text.lower()):
        if token.isdigit():
            result.add(int(token))
        elif token in NUMBERS:
            result.add(NUMBERS[token])
    return result


def validate_teacher_record(row, *, tokenizer=None, min_tokens=80, max_tokens=160):
    if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
        raise ValueError("teacher cache row needs nonempty prompt")
    if not str(row.get("prompt_id", "")).strip() or not str(row.get("category", "")).strip():
        raise ValueError("teacher cache row needs prompt_id and category")
    if row["category"] not in SEMANTIC_CATEGORIES:
        raise ValueError("Phase 1A T0 needs a curated T2I semantic category")
    reasoning = row.get("reasoning_text")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("teacher reasoning is empty")
    if row.get("teacher_template_version") != TEMPLATE_VERSION:
        raise ValueError("unknown teacher template version")
    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    headings = [line.split(":", 1)[0].strip() for line in lines]
    if headings != list(SECTIONS) or any(not line.partition(":")[2].strip() for line in lines):
        raise ValueError("teacher reasoning must contain exactly seven nonempty schema sections")
    if any(word in reasoning.lower() for word in REFUSALS):
        raise ValueError("teacher refusal detected")
    if tokenizer is not None:
        count = len(tokenizer.encode(reasoning, add_special_tokens=False))
        if not min_tokens <= count <= max_tokens:
            raise ValueError(f"teacher reasoning length {count} not in [{min_tokens},{max_tokens}]")
    # Conservative explicit-number check. This is not a full semantic judge;
    # semantic quality still requires the independent pre-training gate.
    source_counts = _numbers(row["prompt"])
    plan_counts = _numbers(next(line for line in lines if line.startswith("Counts:")))
    if source_counts and not plan_counts <= source_counts:
        raise ValueError("teacher Counts section contradicts an explicit prompt count")
    return row


def load_teacher_cache(path, *, tokenizer=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("teacher cache is empty")
    for row in rows:
        validate_teacher_record(row, tokenizer=tokenizer)
    keys = [str(row["prompt_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prompt_id in teacher cache")
    return rows


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
