"""Minimal runtime device abstraction for CUDA and Ascend NPU.

The model code is device-agnostic as long as it asks this module for the
device, autocast context, and random seeding instead of hardcoding ``cuda``.
``import torch_npu`` registers the ``npu`` privateuse1 backend, so importing
this module is enough to make ``torch.device("npu:0")`` valid.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch

_NPU_IMPORTED = False
try:  # pragma: no cover - environment dependent
    import torch_npu  # noqa: F401

    _NPU_IMPORTED = True
except Exception:  # noqa: BLE001 - torch_npu is optional
    _NPU_IMPORTED = False


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def npu_available() -> bool:
    return _NPU_IMPORTED and hasattr(torch, "npu") and bool(torch.npu.is_available())


def default_device_type() -> str:
    if cuda_available():
        return "cuda"
    if npu_available():
        return "npu"
    return "cpu"


def resolve_device(spec=None) -> torch.device:
    """Accept ``cuda:0``/``npu:0``/``auto`` and map to the available backend."""

    text = str(spec or os.environ.get("LCOT_DEVICE") or "auto")
    if text == "auto":
        text = default_device_type()
    if text.startswith("cuda") and not cuda_available() and npu_available():
        text = "npu" + text[len("cuda") :]
    elif text.startswith("npu") and not npu_available() and cuda_available():
        text = "cuda" + text[len("npu") :]
    return torch.device(text)


def is_accelerator(device) -> bool:
    return str(getattr(device, "type", device)) in ("cuda", "npu")


def autocast_for(device):
    kind = str(getattr(device, "type", device))
    if kind == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if kind == "npu":
        return torch.npu.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if cuda_available():
        torch.cuda.manual_seed_all(int(seed))
    if npu_available():
        torch.npu.manual_seed_all(int(seed))


def empty_cache() -> None:
    if cuda_available():
        torch.cuda.empty_cache()
    if npu_available():
        torch.npu.empty_cache()


def enable_dynamo_flex_attention() -> bool:
    """``torch.compile(flex_attention)`` is unstable on Ascend; only CUDA default."""

    if os.environ.get("LCOT_DISABLE_FLEX", "").strip() in ("1", "true", "True"):
        return False
    return cuda_available()
