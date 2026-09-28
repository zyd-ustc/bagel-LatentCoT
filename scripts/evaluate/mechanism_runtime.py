"""Strict device selection for the memory-mechanism experiment launcher."""
from __future__ import annotations

import argparse
import os

import torch


def backend_api(backend):
    if backend == "npu":
        import torch_npu  # noqa: F401 - registers the optional backend
    if backend not in ("cuda", "npu"):
        raise ValueError("backend must be cuda or npu")
    api = getattr(torch, backend)
    if not api.is_available():
        raise RuntimeError(f"{backend} unavailable in this Python environment; no backend fallback")
    return api


def visible_devices(backend):
    """Preserve scheduler/container visibility tokens, including GPU UUIDs."""
    api = backend_api(backend)
    count = api.device_count()
    variable = "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"
    mask = os.environ.get(variable)
    devices = mask.split(",") if mask is not None else [str(i) for i in range(count)]
    devices = [token.strip() for token in devices]
    if not devices or any(not token or token == "-1" for token in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"invalid {variable}")
    if len(devices) != count:
        raise ValueError(f"{variable} lists {len(devices)} devices but PyTorch sees {count}")
    return devices


def prepare_runtime(spec):
    """Fail before loading weights; explicit CUDA never silently becomes NPU."""
    if spec == "auto":
        spec = "cuda:0" if torch.cuda.is_available() else "npu:0"
    kind = spec.split(":")[0]
    api = backend_api(kind)
    device = torch.device(spec)
    index = device.index if device.index is not None else 0
    if not 0 <= index < api.device_count():
        raise ValueError(f"device index out of range: {spec}")
    api.set_device(index)
    if kind == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("memory mechanism inference requires CUDA BF16 support")
    info = {"device": f"{kind}:{index}", "device_name": api.get_device_name(index),
            "torch_version": str(torch.__version__), "dtype": "bfloat16"}
    if kind == "cuda":
        info.update(cuda_version=torch.version.cuda,
                    capability=list(torch.cuda.get_device_capability(index)),
                    total_memory_bytes=torch.cuda.get_device_properties(index).total_memory,
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), default="cuda")
    args = parser.parse_args()
    print("\n".join(visible_devices(args.backend)))
