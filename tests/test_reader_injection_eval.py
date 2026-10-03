"""Fixed-gate diagnostics: CPU contracts and actual tiny MoT velocity paths."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image
from safetensors.torch import save_file
import torch

from qwen_latent_cot.bagel.reader_injection_eval import (
    ARMS, build_plan, digest, generate_prompt, merge_shards, reader_arm,
    select_hard_records, write_json)
from qwen_latent_cot.bagel.cot_teacher import sha256_file
from test_reader_warmup_runner import TinyWarmupRuntime


@pytest.fixture
def runtime():
    torch.manual_seed(7)
    config = dict(height=32, width=32, num_steps=4, timestep_shift=3.,
                  num_loop_tokens=2, memory_loop_start_layer=1, memory_loop_end_layer=3,
                  o_adapter_rank=2, o_adapter_alpha=2)
    runtime = TinyWarmupRuntime(config)
    with torch.no_grad():
        for layer in runtime.model.language_model.model.layers[1:3]:
            layer.memory_reader.output_adapter.B.weight.normal_(std=.05)
    runtime.model.requires_grad_(False)
    return runtime


def trained_state(runtime):
    return {name: value.detach().clone() for name, value in runtime.model.named_parameters()
            if "memory_reader.output_adapter." in name}


def test_tiny_gate_zero_matches_native_and_each_nonzero_arm_recomputes_read(runtime, monkeypatch):
    calls = []
    read = runtime.model.forward_memory_read_bank
    def capture(**kwargs):
        calls.append(kwargs["x_t"].detach().clone())
        return read(**kwargs)
    monkeypatch.setattr(runtime.model, "forward_memory_read_bank", capture)
    row, samples = generate_prompt(runtime, dict(prompt_id="p", prompt="three cubes"),
        seed=42, gate_scale=1., trained=trained_state(runtime))
    assert row["gate_zero_native_parity_max_abs"] == 0.
    assert len({value["initial_noise_sha256"] for value in row["arms"].values()}) == 1
    assert len({value["schedule_sha256"] for value in row["arms"].values()}) == 1
    assert len(calls) == 1 + 2 * 3  # Parity Read, then every Euler step of two arms.
    assert not torch.equal(samples["native"], samples["step5000_reader"])
    assert not torch.equal(samples["untrained_reader"], samples["step5000_reader"])
    assert not any(value.requires_grad for value in runtime.model.parameters())


def test_zero_injection_full_trajectories_match_and_are_seed_reproducible(runtime):
    args = dict(seed=44, gate_scale=0., trained=trained_state(runtime))
    row, samples = generate_prompt(runtime, dict(prompt_id="p", prompt="cubes"), **args)
    repeated, _ = generate_prompt(runtime, dict(prompt_id="p", prompt="cubes"), **args)
    assert row == repeated
    assert all(torch.equal(samples["native"], value) for value in samples.values())


def test_untrained_is_exact_b_zero_function_and_parameters_restore_on_exception(runtime):
    trained = trained_state(runtime)
    params = dict(runtime.model.named_parameters())
    reader = runtime.model.language_model.model.layers[1].memory_reader
    q = torch.randn(2, 2, 4).bfloat16()
    k = torch.randn(2, 1, 4).bfloat16()
    v = torch.randn(2, 1, 4).bfloat16()
    with pytest.raises(RuntimeError, match="fixture"):
        with reader_arm(runtime, trained, "untrained_reader", 1.):
            expected = reader(gen_query=q, memory_key=k, memory_value=v)
            with torch.no_grad():
                reader.output_adapter.A.weight.normal_(std=10.)
            assert torch.equal(expected, reader(gen_query=q, memory_key=k, memory_value=v))
            assert all(not value.any() for name, value in params.items() if name.endswith(".B.weight"))
            raise RuntimeError("fixture")
    assert all(torch.equal(params[name], value) for name, value in trained.items())
    assert reader.injection_gate.item() == 0.


def make_plan_inputs(tmp_path):
    model = tmp_path / "model"; model.mkdir()
    main = tmp_path / "main"; main.mkdir()
    train, heldout = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    train.write_text(json.dumps(dict(prompt_id="t", prompt="train cubes", category="count")) + "\n")
    heldout.write_text("".join(json.dumps(dict(prompt_id=f"h{i}", prompt=f"heldout cubes {i}", category="count")) + "\n" for i in range(2)))
    config = dict(model_path=str(model), prompt_data=str(train), heldout_prompt_data=str(heldout),
                  output_dir=str(main), seed=42, height=512, width=512, num_steps=50,
                  num_loop_tokens=8, memory_loop_start_layer=12, memory_loop_end_layer=20,
                  o_adapter_rank=8, o_adapter_alpha=16)
    write_json(main / "resolved_config.json", config)
    names = {f"language_model.model.layers.{layer}.memory_reader.output_adapter.{route}.weight":
             torch.ones((8, 16) if route == "A" else (16, 8)) for layer in range(12, 20) for route in ("A", "B")}
    checkpoint = main / "reader_warmup_step_0005000.safetensors"
    save_file(names, str(checkpoint))
    meta = dict(schema="bagel-memory-reader-warmup-v1", stage="Phase 1A.0", step=5000,
                generation_injection=False, K=8, body=[12,20], adapter_rank=8, adapter_alpha=16,
                model_path=str(model), trainable_names=sorted(names),
                memory_init="prompt_hidden_uniform", query_source="native_gen_q",
                memory_kv_source="frozen_read_native_kv", prompt_target="native_prompt_bank",
                prompt_data_sha256=sha256_file(train), heldout_prompt_data_sha256=sha256_file(heldout))
    write_json(checkpoint.with_suffix(".json"), meta)
    benchmark = tmp_path / "benchmark.jsonl"
    rows = [dict(prompt_id=f"p{a}_{i}", prompt=f"benchmark {a} cubes {i}", atom_count=a,
                 skills=["count"], vqa_list=[["How many?", "two"]])
            for a in (7, 8, 9, 10) for i in range(4)]
    benchmark.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return checkpoint, benchmark, config


def test_plan_inherits_training_geometry_and_selects_balanced_independent_hard16(tmp_path):
    checkpoint, benchmark, config = make_plan_inputs(tmp_path)
    plan = build_plan(checkpoint, benchmark, tmp_path / "out")
    assert plan["gate_scale"] == 1. and plan["arms"] == list(ARMS)
    assert plan["prompt_seeds"] == list(range(42, 58))
    assert [row["atom_count"] for row in plan["records"]] == [7]*4 + [8]*4 + [9]*4 + [10]*4
    assert plan["config"]["height"] == config["height"]
    assert plan["config"]["num_steps"] == 50 and plan["config"]["cfg_text_scale"] == 1.
    assert not (tmp_path / "out").exists()
    assert build_plan(checkpoint, benchmark, tmp_path/"other", gate_scale=.1)["gate_scale"] == .1


def test_cpu_cli_preflight_never_creates_output_or_loads_gpu_model(tmp_path):
    checkpoint, benchmark, _ = make_plan_inputs(tmp_path)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, "scripts/evaluate/bagel_reader_injection.py",
        "--checkpoint", str(checkpoint), "--benchmark-data", str(benchmark),
        "--output-dir", str(output), "--dry-run"], cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["gate_scale"] == 1.
    assert not output.exists()


def test_launcher_rejects_missing_checkpoint_before_creating_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "out"
    result = subprocess.run(["bash", "scripts/evaluate/run_bagel_reader_injection.sh", str(output)],
        cwd=root, env={**os.environ, "PYTHON_BIN": sys.executable,
                       "READER_CHECKPOINT": str(tmp_path / "missing.safetensors"),
                       "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["nan", "model", "geometry", "source", "overlap", "step", "reader_contract"])
def test_plan_rejects_invalid_or_mismatched_inputs(tmp_path, mutation):
    checkpoint, benchmark, config = make_plan_inputs(tmp_path)
    args = {}
    if mutation == "nan": args["gate_scale"] = float("nan")
    elif mutation == "model": args["model_path"] = tmp_path / "another_model"
    elif mutation == "geometry":
        config["memory_loop_start_layer"] = 11
        write_json(checkpoint.parent/"resolved_config.json", config)
    elif mutation == "source": Path(config["prompt_data"]).write_text(Path(config["prompt_data"]).read_text() + "\n")
    elif mutation == "overlap":
        rows = [json.loads(line) for line in benchmark.read_text().splitlines()]
        rows[0]["prompt"] = " TRAIN   CUBES "
        benchmark.write_text("".join(json.dumps(row)+"\n" for row in rows))
    else:
        meta = json.loads(checkpoint.with_suffix(".json").read_text())
        if mutation == "step": meta["step"] = 4750
        else: meta["query_source"] = "changed_query"
        write_json(checkpoint.with_suffix(".json"), meta)
    with pytest.raises((ValueError, FileNotFoundError)):
        build_plan(checkpoint, benchmark, tmp_path/"out", **args)


def create_fake_shards(plan):
    root = Path(plan["output"]); root.mkdir()
    for shard_id in range(plan["num_shards"]):
        shard = root/"shards"/f"s{shard_id:03d}"; shard.mkdir(parents=True)
        rows=[]
        for index in range(shard_id, plan["max_prompts"], plan["num_shards"]):
            folder=root/f"p{index:03d}"; folder.mkdir()
            record=plan["records"][index]
            (folder/"prompt.txt").write_text(record["prompt"]+"\n")
            for arm in ARMS: Image.new("RGB", (512,512)).save(folder/f"{arm}.png")
            row=dict(index=index, prompt_id=record["prompt_id"], prompt=record["prompt"],
                seed=plan["prompt_seeds"][index],gate_scale=plan["gate_scale"],
                gate_zero_native_parity_max_abs=0.,
                arms={arm:dict(initial_noise_sha256="noise", schedule_sha256="schedule",euler_steps=49) for arm in ARMS},
                image_sha256={arm:sha256_file(folder/f"{arm}.png") for arm in ARMS})
            rows.append(row); write_json(folder/"generation.json",row)
        write_json(shard/"status.json",dict(status="complete"))
        write_json(shard/"manifest.json",dict(plan_sha256=digest(plan),shard_id=shard_id,rows=rows))


def test_merge_requires_all_eight_shards_and_publishes_aligned_scoring_maps(tmp_path):
    checkpoint, benchmark, _ = make_plan_inputs(tmp_path)
    plan=build_plan(checkpoint,benchmark,tmp_path/"out")
    create_fake_shards(plan)
    merge_shards(plan)
    root=Path(plan["output"])
    assert json.loads((root/"status.json").read_text())["images"] == 48
    assert len(json.loads((root/"step5000_reader_image_map.json").read_text())) == 16
    assert "step5000_reader" in (root/"index.html").read_text()
    with pytest.raises(FileExistsError): merge_shards(plan)


@pytest.mark.parametrize("mutation", ["noise", "seed", "image", "plan", "missing", "parity"])
def test_merge_refuses_corrupt_or_noncomparable_results(tmp_path, mutation):
    checkpoint,benchmark,_=make_plan_inputs(tmp_path)
    plan=build_plan(checkpoint,benchmark,tmp_path/"out")
    create_fake_shards(plan)
    root=Path(plan["output"]); manifest=root/"shards/s000/manifest.json"
    data=json.loads(manifest.read_text()); row=data["rows"][0]
    if mutation=="noise": row["arms"]["step5000_reader"]["initial_noise_sha256"]="other"
    elif mutation=="seed": row["seed"]+=1
    elif mutation=="image": (root/"p000/native.png").write_bytes(b"broken")
    elif mutation=="plan": data["plan_sha256"]="wrong"
    elif mutation=="missing": data["rows"].pop()
    else: row["gate_zero_native_parity_max_abs"] = .01
    if mutation in ("noise","seed","parity"): write_json(root/"p000/generation.json",row)
    write_json(manifest,data)
    with pytest.raises(ValueError): merge_shards(plan)
    assert not (root/"index.html").exists()
