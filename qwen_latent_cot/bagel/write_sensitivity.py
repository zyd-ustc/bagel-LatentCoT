"""Read-to-Write memory interventions for the paired T2I causal probe."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


WRITE_MEMORY_SOURCES = ("correct", "shuffle", "m0", "zero")
PROMPT_MEMORY_SLOT_BETA = 0.05


def prompt_memory_init(
    prompt_body_entry: torch.Tensor,
    *,
    slots: int,
    beta: float = PROMPT_MEMORY_SLOT_BETA,
) -> torch.Tensor:
    """Nonparametric [B*K,D] memory at the same depth as the loop body.

    Each row of ``prompt_body_entry`` is the causal final prompt token after
    prefix layers [0,s).  The centered sinusoidal offsets distinguish slots
    without adding trainable parameters or changing the prompt KV cache.
    """

    if prompt_body_entry.ndim != 2 or slots < 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError("expected prompt body entry [B,D], K>0 and finite beta>=0")
    batch, width = prompt_body_entry.shape
    if batch < 1 or width < 1:
        raise ValueError("prompt body entry must be nonempty")
    anchor = prompt_body_entry.float()
    if not bool(torch.isfinite(anchor).all()):
        raise ValueError("prompt body entry must be finite")
    slot_ids = torch.arange(1, slots + 1, device=anchor.device, dtype=torch.float32)[:, None]
    features = torch.arange(1, width + 1, device=anchor.device, dtype=torch.float32)[None, :]
    offsets = torch.sin(slot_ids * features * 0.013) + torch.cos(slot_ids * features * 0.017)
    offsets = offsets - offsets.mean(dim=0, keepdim=True)
    # One shared scale preserves the zero-mean-across-slots invariant. A
    # separate RMS for each slot would undo the centering above.
    offsets = offsets / offsets.square().mean().sqrt().clamp_min(1e-6)
    if bool((anchor.square().sum(dim=1) == 0).any()):
        raise ValueError("prompt body entry must be nonzero")
    memory = anchor[:, None, :] + beta * offsets[None, :, :]
    return memory.to(dtype=prompt_body_entry.dtype).reshape(batch * slots, width)


def select_write_memory(
    read_memory: torch.Tensor,
    initial_memory: torch.Tensor,
    *,
    source: str,
    batch_size: int,
) -> torch.Tensor:
    """Select only the memory fed into Write; never modify the Read result."""

    if source not in WRITE_MEMORY_SOURCES:
        raise ValueError(f"unknown Write memory source: {source}")
    if read_memory.ndim != 2 or initial_memory.shape != read_memory.shape:
        raise ValueError("Read and m0 memory must have matching [B*K,D] shapes")
    if batch_size < 1 or int(read_memory.shape[0]) % batch_size:
        raise ValueError("memory slots must divide equally among batch samples")
    if source == "correct":
        return read_memory
    if source == "m0":
        return initial_memory
    if source == "zero":
        return torch.zeros_like(read_memory)
    if batch_size < 2:
        raise ValueError("shuffle requires batch size > 1")
    slots = int(read_memory.shape[0]) // batch_size
    # Roll by one is a deterministic derangement for every B>1.  It never
    # shuffles slots within a sample or draws from outside the current batch.
    return torch.roll(read_memory.reshape(batch_size, slots, -1), 1, dims=0).reshape_as(
        read_memory
    )


def append_write_probe(
    rows: list[dict[str, Any]],
    read_memory: torch.Tensor,
    initial_memory: torch.Tensor,
    used_memory: torch.Tensor,
    *,
    batch_size: int,
) -> None:
    """Record cheap CPU scalars; avoid Ascend's optional SVD/TBE path."""

    read = read_memory.detach().float().cpu().reshape(batch_size, -1)
    initial = initial_memory.detach().float().cpu().reshape(batch_size, -1)
    used = used_memory.detach().float().cpu().reshape(batch_size, -1)

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
        if float(a.norm()) == 0.0 or float(b.norm()) == 0.0:
            return None
        value = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0])
        return value if math.isfinite(value) else None

    for sample in range(batch_size):
        rows.append(
            {
                "sample_in_pair": sample,
                "read_l2": float(read[sample].norm()),
                "m0_l2": float(initial[sample].norm()),
                "read_minus_m0_l2": float((read[sample] - initial[sample]).norm()),
                "read_m0_cos": cosine(read[sample], initial[sample]),
                "read_write_input_cos": cosine(read[sample], used[sample]),
            }
        )
