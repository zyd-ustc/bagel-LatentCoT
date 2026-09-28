from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/evaluate"))
import mechanism_runtime as runtime


def mock_cuda(monkeypatch, *, available=True, bf16=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA H200")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=123))


def test_cuda_runtime_checks_and_metadata(monkeypatch):
    mock_cuda(monkeypatch)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    info = runtime.prepare_runtime("cuda:1")
    assert selected == [1]
    assert info["device"] == "cuda:1"
    assert info["device_name"] == "NVIDIA H200"
    assert info["capability"] == [9, 0]
    assert info["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="out of range"):
        runtime.prepare_runtime("cuda:2")


def test_cuda_unavailable_never_falls_back(monkeypatch):
    mock_cuda(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match="no backend fallback"):
        runtime.prepare_runtime("cuda:0")


def test_cuda_rejects_no_bf16(monkeypatch):
    mock_cuda(monkeypatch, bf16=False)
    with pytest.raises(RuntimeError, match="BF16"):
        runtime.prepare_runtime("cuda:0")


def test_device_inventory_preserves_allocation(monkeypatch):
    mock_cuda(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb")
    assert runtime.visible_devices("cuda") == ["GPU-aaa", "GPU-bbb"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    assert runtime.visible_devices("cuda") == ["3", "7"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert runtime.visible_devices("cuda") == ["0", "1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,1")
    with pytest.raises(ValueError, match="invalid"):
        runtime.visible_devices("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    with pytest.raises(ValueError, match="PyTorch sees"):
        runtime.visible_devices("cuda")


def launcher(tmp_path, **overrides):
    """Exercise the real shell with a fake interpreter, never start a GPU job."""
    fake = tmp_path / "fake-python"
    fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
args = sys.argv[1:]
with open(os.environ["TEST_CALLS"], "a") as handle:
    handle.write(json.dumps({"args": args, "cuda": os.getenv("CUDA_VISIBLE_DEVICES"),
                             "npu": os.getenv("ASCEND_RT_VISIBLE_DEVICES")}) + "\\n")
if args[0].endswith("mechanism_runtime.py"):
    print("3\\n7")
elif "--dry-run" in args:
    print("{}")
elif "--shard-id" in args and os.getenv("TEST_FAIL"):
    sys.exit(1)
''')
    fake.chmod(0o755)
    env = dict(os.environ)
    for name in ("NUM_SHARDS", "MAX_PROMPTS", "BACKEND", "TEST_FAIL"):
        env.pop(name, None)
    env.update(PYTHON_BIN=str(fake), TEST_CALLS=str(tmp_path / "calls.jsonl"),
               CUDA_VISIBLE_DEVICES="3,7", MAX_PROMPTS="4", **overrides)
    result = subprocess.run(["bash", str(ROOT / "scripts/evaluate/run_bagel_memory_mechanism.sh"),
                             str(tmp_path / "out")], env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    return result, calls


def test_launcher_cuda_isolates_allocated_gpus(tmp_path):
    result, calls = launcher(tmp_path)
    assert result.returncode == 0, result.stderr
    workers = [call for call in calls if "--shard-id" in call["args"]]
    assert len(workers) == 2
    assert {call["cuda"] for call in workers} == {"3", "7"}
    for call in workers:
        assert call["args"][call["args"].index("--device") + 1] == "cuda:0"
    assert "--merge-only" in calls[-1]["args"]


def test_launcher_keeps_explicit_npu_support(tmp_path):
    result, calls = launcher(tmp_path, BACKEND="npu")
    assert result.returncode == 0, result.stderr
    workers = [call for call in calls if "--shard-id" in call["args"]]
    assert {call["npu"] for call in workers} == {"3", "7"}
    assert all("npu:0" in call["args"] for call in workers)


def test_launcher_rejects_gpu_oversubscription(tmp_path):
    result, calls = launcher(tmp_path, NUM_SHARDS="3")
    assert result.returncode == 2
    assert not (tmp_path / "out").exists()
    assert not any("--shard-id" in call["args"] for call in calls)


def test_launcher_worker_failure_skips_merge(tmp_path):
    result, calls = launcher(tmp_path, TEST_FAIL="1")
    assert result.returncode == 1
    assert not any("--merge-only" in call["args"] for call in calls)
