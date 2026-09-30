"""CPU-only continuation regression; never loads full BAGEL or launches H200 training."""
import json
import os
import random
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from qwen_latent_cot.bagel.reader_warmup import (
    ReaderWarmupRuntime, capture_warmup_rng, inspect_warmup_resume,
    restore_warmup_rng, train_warmup)
from test_reader_warmup_runner import TinyWarmupRuntime


def resume_fixture(tmp_path):
    train = [dict(prompt_id=f"t{i}", prompt=f"{i+2} cubes", category="count", split="train")
             for i in range(5)]
    heldout = [dict(prompt_id=f"h{i}", prompt=f"{i+8} spheres", category="count", split="heldout")
               for i in range(2)]
    source = tmp_path / "train.jsonl"
    validation = tmp_path / "heldout.jsonl"
    for path, rows in ((source, train), (validation, heldout)):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    config = dict(model_path=str(tmp_path), prompt_data=str(source), heldout_prompt_data=str(validation),
        output_dir=str(tmp_path/"parent"), height=32, width=32, num_steps=4, timestep_shift=3.,
        seed=42, states_per_rollout=2, num_loop_tokens=2, memory_loop_start_layer=1,
        memory_loop_end_layer=3, o_adapter_rank=2, o_adapter_alpha=2, learning_rate=1e-4,
        max_grad_norm=1., max_steps=2, save_steps=1, eval_steps=1, eval_max_prompts=2,
        timestep_bucket_weights=[.5,.4,.1])
    return config, train, heldout


def seed_all():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def run_tiny(monkeypatch, config, train, heldout, runtime_type=TinyWarmupRuntime):
    seed_all()
    runtime = runtime_type(config)
    monkeypatch.setattr(ReaderWarmupRuntime, "load_model", classmethod(lambda cls, cfg: runtime))
    train_warmup(config, train, heldout)
    return Path(config["output_dir"])


def checkpoint(root, step=2):
    return root / f"reader_warmup_step_{step:07d}.safetensors"


def assert_same_optimizer(left, right, step):
    a = torch.load(checkpoint(left, step).with_suffix(".optimizer.pt"), weights_only=True)
    b = torch.load(checkpoint(right, step).with_suffix(".optimizer.pt"), weights_only=True)
    assert a["step"] == b["step"] == step
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["optimizer"]["state"][key][name])


class StochasticTinyRuntime(TinyWarmupRuntime):
    def warmup_forward(self, state, *, memory_override=None):
        velocity, bank, sink = super().warmup_forward(state, memory_override=memory_override)
        if state.condition.record.get("split") == "train":
            # Exercise all three checkpointed RNG streams, not just explicit rollout seeds.
            scale = 1.+.01*(torch.rand(()).item()+random.random()+float(np.random.rand()))
            sink = {layer:(readout*scale, target) for layer,(readout,target) in sink.items()}
        return velocity, bank, sink


@pytest.mark.parametrize("resume_step", [1,2])
def test_resume_matches_uninterrupted_weights_optimizer_samples_rng_and_baseline(tmp_path, monkeypatch, resume_step):
    config, train, heldout = resume_fixture(tmp_path)
    full = run_tiny(monkeypatch, {**config,"max_steps":4,"output_dir":str(tmp_path/"full")},
                    train, heldout, StochasticTinyRuntime)
    parent = run_tiny(monkeypatch, config, train, heldout, StochasticTinyRuntime)
    original_bytes = {p.name:p.read_bytes() for p in parent.iterdir()}
    child_config = {**config,"max_steps":4,"output_dir":str(tmp_path/"child"),
                    "resume_checkpoint":str(checkpoint(parent, resume_step))}
    info = inspect_warmup_resume(child_config, world_size=1)
    assert info["step"] == resume_step and info["mode"] == "per_rank_rng"
    child = run_tiny(monkeypatch, child_config, train, heldout, StochasticTinyRuntime)
    for name, value in load_file(str(checkpoint(full, 4))).items():
        assert torch.equal(value, load_file(str(checkpoint(child, 4)))[name])
    assert_same_optimizer(full, child, 4)
    full_rows = [json.loads(x) for x in (full/"metrics.jsonl").read_text().splitlines()]
    child_rows = [json.loads(x) for x in (child/"metrics.jsonl").read_text().splitlines()]
    assert child_rows == full_rows[resume_step:]
    assert [r["step"] for r in child_rows] == list(range(resume_step+1,5))
    assert (child/"heldout_diagnostics.jsonl").read_text().splitlines()[0] == (parent/"heldout_diagnostics.jsonl").read_text().splitlines()[0]
    report = json.loads((child/"warmup_gate.json").read_text())
    assert report["initial_error_mse"] == json.loads((parent/"warmup_gate.json").read_text())["initial_error_mse"]
    assert {p.name:p.read_bytes() for p in parent.iterdir()} == original_bytes
    assert json.loads((child/"run_manifest.json").read_text())["start_step"] == resume_step


def make_legacy(parent):
    # Simulate the exact old three-file checkpoint layout and original run provenance.
    manifest = parent/"run_manifest.json"
    row = json.loads(manifest.read_text())
    row.pop("resume_schema")
    manifest.write_text(json.dumps(row))
    for suffix in (".resume.pt", ".resume.json"):
        checkpoint(parent).with_suffix(suffix).unlink()


def test_legacy_checkpoint_restores_optimizer_step_and_old_initial_baseline(tmp_path, monkeypatch):
    config, train, heldout = resume_fixture(tmp_path)
    full = run_tiny(monkeypatch, {**config,"max_steps":4,"output_dir":str(tmp_path/"full")}, train, heldout)
    parent = run_tiny(monkeypatch, config, train, heldout)
    make_legacy(parent)
    child_config = {**config,"max_steps":4,"output_dir":str(tmp_path/"child"),
                    "resume_checkpoint":str(checkpoint(parent))}
    assert inspect_warmup_resume(child_config)["mode"] == "legacy_seeded_rollout"
    child = run_tiny(monkeypatch, child_config, train, heldout)
    assert_same_optimizer(full, child, 4)
    assert (child/"metrics.jsonl").read_text() == "\n".join((full/"metrics.jsonl").read_text().splitlines()[2:])+"\n"


@pytest.fixture
def saved_resume(tmp_path, monkeypatch):
    config, train, heldout = resume_fixture(tmp_path)
    parent = run_tiny(monkeypatch, config, train, heldout)
    return {**config,"max_steps":4,"output_dir":str(tmp_path/"child"),
            "resume_checkpoint":str(checkpoint(parent))}, parent


@pytest.mark.parametrize("key,value", [("seed",43),("num_steps",5),("learning_rate",2e-4),
    ("max_grad_norm",2.),("eval_max_prompts",3),("timestep_bucket_weights",[.4,.5,.1])])
def test_resume_rejects_changed_protocol(saved_resume, key, value):
    config, _ = saved_resume
    with pytest.raises(ValueError, match="config mismatch"):
        inspect_warmup_resume({**config,key:value})


def test_resume_rejects_world_data_output_and_total_budget_mismatch(saved_resume):
    config, parent = saved_resume
    with pytest.raises(ValueError, match="world_size/global batch"):
        inspect_warmup_resume(config, world_size=2)
    with pytest.raises(FileExistsError, match="fresh output"):
        inspect_warmup_resume({**config,"output_dir":str(parent)})
    with pytest.raises(ValueError, match="total max_steps"):
        inspect_warmup_resume({**config,"max_steps":2})
    source = Path(config["prompt_data"])
    source.write_text(source.read_text()+"\n")
    with pytest.raises(ValueError, match="dataset hash"):
        inspect_warmup_resume(config)


@pytest.mark.parametrize("kind", ["missing", "counter", "nonfinite", "moments", "order", "lr"])
def test_resume_rejects_invalid_optimizer(saved_resume, kind):
    config, parent = saved_resume
    path = checkpoint(parent).with_suffix(".optimizer.pt")
    if kind == "missing":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            inspect_warmup_resume(config)
        return
    state = torch.load(path, weights_only=True)
    if kind == "counter": state["optimizer"]["state"][0]["step"].fill_(1)
    if kind == "nonfinite": state["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
    if kind == "moments": state["optimizer"]["state"].clear()
    if kind == "order": state["optimizer"]["param_groups"][0]["params"].reverse()
    if kind == "lr": state["optimizer"]["param_groups"][0]["lr"] *= 2
    torch.save(state, path)
    with pytest.raises(ValueError, match="optimizer|AdamW"):
        inspect_warmup_resume(config)


def test_resume_rejects_incomplete_sidecars_and_missing_initial(saved_resume):
    config, parent = saved_resume
    path = checkpoint(parent).with_suffix(".resume.json")
    marker = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="completion marker"):
        inspect_warmup_resume(config)
    path.write_bytes(marker)
    rng = checkpoint(parent).with_suffix(".resume.pt")
    rng.write_bytes(rng.read_bytes()+b"tamper")
    with pytest.raises(ValueError, match="sidecar hash"):
        inspect_warmup_resume(config)
    (parent/"heldout_diagnostics.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="step-zero"):
        inspect_warmup_resume(config)


@pytest.mark.parametrize("target", ["initial", "source", "rng"])
def test_resume_rejects_changed_provenance_or_rng(saved_resume, target):
    config, parent = saved_resume
    if target == "initial":
        p = parent/"heldout_diagnostics.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[0]["heldout_error_mse"]["correct"] *= 2
        p.write_text("".join(json.dumps(row)+"\n" for row in rows))
        message = "baseline hash"
    elif target == "source":
        p = parent/"run_manifest.json"
        p.write_text(p.read_text()+"\n")
        message = "provenance hash"
    else:
        from qwen_latent_cot.bagel.cot_teacher import sha256_file
        p = checkpoint(parent).with_suffix(".resume.pt")
        states = torch.load(p, weights_only=True)
        states[0]["torch_cpu"] = torch.zeros(1)
        torch.save(states, p)
        marker = checkpoint(parent).with_suffix(".resume.json")
        row = json.loads(marker.read_text())
        row["hashes"][".resume.pt"] = sha256_file(p)
        marker.write_text(json.dumps(row))
        message = "RNG state"
    with pytest.raises(ValueError, match=message):
        inspect_warmup_resume(config)


def test_rng_round_trip_preserves_cpu_python_numpy():
    seed_all()
    state = capture_warmup_rng("cpu")
    expected = (torch.rand(5), random.random(), np.random.rand(5))
    restore_warmup_rng(state, "cpu")
    actual = (torch.rand(5), random.random(), np.random.rand(5))
    assert torch.equal(expected[0],actual[0]) and expected[1] == actual[1]
    assert np.array_equal(expected[2], actual[2])


def test_resume_cli_validation_does_not_load_model_or_create_output(tmp_path):
    # Full Phase1A.0 contract using lightweight synthetic adapter files only.
    from test_reader_warmup_contract import warmup_evidence
    from qwen_latent_cot.bagel.cot_teacher import sha256_file
    from qwen_latent_cot.bagel.reader_warmup import validate_warmup_config
    model = tmp_path/"model"
    model.mkdir()
    fixture, report = warmup_evidence(tmp_path, model)
    path = Path(fixture["reader_warmup_checkpoint"])
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["step"] = 1
    path.with_suffix(".json").write_text(json.dumps(meta))
    source_config = {key:fixture[key] for key in ("model_path","prompt_data","heldout_prompt_data","output_dir")}
    cfg = validate_warmup_config(source_config)
    (tmp_path/"resolved_config.json").write_text(json.dumps(cfg))
    (tmp_path/"trainable_routes.json").write_text(json.dumps(meta["trainable_names"]))
    manifest = dict(world_size=8,effective_batch_size=8,per_rank_batch_size=1,
        prompt_data_sha256=sha256_file(cfg["prompt_data"]),
        heldout_prompt_data_sha256=sha256_file(cfg["heldout_prompt_data"]))
    (tmp_path/"run_manifest.json").write_text(json.dumps(manifest))
    params = [torch.nn.Parameter(tensor.clone()) for tensor in load_file(str(path)).values()]
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9,.95), weight_decay=0.)
    sum(p.square().sum() for p in params).backward()
    optimizer.step()
    torch.save(dict(step=1,optimizer=optimizer.state_dict()), path.with_suffix(".optimizer.pt"))
    report.update(step=0,per_state=[dict(prompt_id="h0"),dict(prompt_id="h1")])
    (tmp_path/"heldout_diagnostics.jsonl").write_text(json.dumps(report)+"\n")
    command = [sys.executable,"scripts/train/bagel_memory_reader_warmup.py",
        "--model-path",str(model),"--prompt-data",cfg["prompt_data"],
        "--heldout-prompt-data",cfg["heldout_prompt_data"],"--output-dir",cfg["output_dir"],
        "--resume-checkpoint",str(path),"--validate-only"]
    result = subprocess.run(command, capture_output=True, text=True,
        env={**os.environ,"PYTHONPATH":str(Path.cwd()),"WORLD_SIZE":"8"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resume"] == dict(step=1,world_size=8,mode="legacy_seeded_rollout")
    assert not Path(cfg["output_dir"]).exists()
    rejected = subprocess.run(command, capture_output=True, text=True,
        env={**os.environ,"PYTHONPATH":str(Path.cwd()),"WORLD_SIZE":"4"})
    assert rejected.returncode != 0 and "world_size/global batch" in rejected.stderr
    assert not Path(cfg["output_dir"]).exists()


def distributed_resume_worker(rank, rendezvous, config, train, heldout):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        parent_config = {**config,"output_dir":str(Path(config["output_dir"])/"parent")}
        for phase, cfg in (("full",{**config,"max_steps":4,"output_dir":str(Path(config["output_dir"])/"full")}),
            ("parent",parent_config), ("child",{**config,"max_steps":4,
            "output_dir":str(Path(config["output_dir"])/"child"),
            "resume_checkpoint":str(checkpoint(Path(parent_config["output_dir"]))) } )):
            seed_all()
            runtime = StochasticTinyRuntime(cfg)
            # Distinct RNG per rank verifies that RNG is not just broadcast from rank zero.
            torch.manual_seed(42+rank); random.seed(42+rank); np.random.seed(42+rank)
            ReaderWarmupRuntime.load_model = classmethod(lambda cls, current: runtime)
            train_warmup(cfg, train, heldout)
            params = torch.cat([p.detach().flatten() for p in runtime.model.parameters() if p.requires_grad])
            if phase == "full": expected = params
            if phase == "child": assert torch.equal(params,expected)
    finally:
        dist.destroy_process_group()


def test_two_rank_resume_matches_global_optimizer_and_correct_next_samples(tmp_path):
    config, train, heldout = resume_fixture(tmp_path)
    config["output_dir"] = str(tmp_path/"distributed")
    mp.spawn(distributed_resume_worker,
        args=("file://"+str(tmp_path/"rendezvous"),config,train,heldout), nprocs=2, join=True)
    root = Path(config["output_dir"])
    assert_same_optimizer(root/"full", root/"child", 4)
    rows = [json.loads(x) for x in (root/"child/metrics.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 3 and rows[0]["prompt_ids"] == ["t4","t0"]
    assert rows[1]["step"] == 4 and rows[1]["prompt_ids"] == ["t1","t2"]
