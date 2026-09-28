import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from qwen_latent_cot.bagel.cot_teacher import (
    SECTIONS, TEMPLATE_VERSION, validate_teacher_record, teacher_condition)
from qwen_latent_cot.bagel.opd_training import (
    SCHEMA, check_teacher_baseline, load_training_records, validate_opd_config)
from qwen_latent_cot.bagel.cot_teacher import sha256_file
from qwen_latent_cot.bagel.opd_evaluation import fixed_state_metrics


def _row(prompt_id="x", prompt="Three red cubes and one blue sphere"):
    return dict(prompt_id=prompt_id, prompt=prompt, category="count",
        teacher_template_version=TEMPLATE_VERSION,
        reasoning_text="\n".join(f"{heading}: " + (
            "three red cubes and one blue sphere" if heading == "Counts" else "preserve scene")
            for heading in SECTIONS))


def test_teacher_cache_schema_and_contradictory_count():
    row=_row()
    validate_teacher_record(row)
    assert row["prompt"] in teacher_condition(row["prompt"],row["reasoning_text"])
    row["reasoning_text"]=row["reasoning_text"].replace("three red cubes", "four red cubes")
    with pytest.raises(ValueError,match="contradicts"):
        validate_teacher_record(row)


def test_opd_rejects_pair_ranking_and_mismatched_teacher_gate(tmp_path):
    model=tmp_path/"model"
    model.mkdir()
    source=tmp_path/"prompts.jsonl"
    source.write_text(json.dumps(dict(id="x", prompt=_row()["prompt"],category="count"))+"\n")
    cache=tmp_path/"teacher.jsonl"
    cache.write_text(json.dumps(_row())+"\n")
    base=dict(model_path=str(model),prompt_data=str(source),teacher_cot_data=str(cache),
              output_dir=str(tmp_path/"output"))
    config=validate_opd_config(base)
    assert load_training_records(config)[0]["prompt_id"] == "x"
    with pytest.raises(ValueError,match="forbids"):
        validate_opd_config({**base,"lambda_dep_shuffle":0.5})
    gate=tmp_path/"baseline.json"
    gate.write_text(json.dumps(dict(schema=SCHEMA,kind="teacher_baseline",
        teacher_cache_sha256=sha256_file(cache),model_path=str(model.resolve()),
        num_steps=50,cfg=1.0,prompt_ids=[str(i) for i in range(8)],
        field_rms_by_seed={"42":.02,"43":.03})))
    assert check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})
    cache.write_text(cache.read_text()+"\n")
    with pytest.raises(ValueError,match="does not match"):
        check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})


def test_evaluation_controls_use_same_state_and_never_enter_loss():
    class Runtime:
        def rollout(self, row, seed):
            from types import SimpleNamespace
            condition=SimpleNamespace(record=row)
            return [SimpleNamespace(condition=condition, sample=torch.tensor([1.]),
                                    timestep=.5, step_index=3)]
        def student_velocity(self, state, *, return_memory=False, memory_override=None):
            memory=torch.tensor([1. if state.condition.record["prompt_id"]=="a" else 2.])
            if return_memory:
                return state.sample+memory, memory
            return state.sample+(memory if memory_override is None else memory_override)
        def teacher_velocity(self, state):
            return state.sample+1
        def native_velocity(self, state):
            return state.sample
    rows=[dict(prompt_id="a",prompt="a prompt"),dict(prompt_id="b",prompt="b prompt")]
    result=fixed_state_metrics(Runtime(),rows,seed=42)
    assert result["per_prompt"][0]["errors"] == dict(native=1.,correct=0.,shuffled=1.,zero=1.)
    assert result["per_prompt"][1]["errors"] == dict(native=1.,correct=1.,shuffled=0.,zero=1.)
    many=rows+[dict(prompt_id="c",prompt="c prompt"),dict(prompt_id="d",prompt="d prompt")]
    assert len(fixed_state_metrics(Runtime(),many,seed=42)["per_prompt"])==4
    with pytest.raises(ValueError,match="even number"):
        fixed_state_metrics(Runtime(),many[:3],seed=42)


def test_opd_cli_preflight_does_not_load_bagel_weights(tmp_path):
    model=tmp_path/"empty_model_directory"
    model.mkdir()
    source=tmp_path/"prompts.jsonl"
    cache=tmp_path/"cot.jsonl"
    source.write_text(json.dumps(dict(id="x",prompt=_row()["prompt"],category="count"))+"\n")
    cache.write_text(json.dumps(_row())+"\n")
    command=[sys.executable,"scripts/train/bagel_memory_opd.py",
        "--config","configs/training/memory_opd_t0.yaml",
        "--model-path",str(model),"--prompt-data",str(source),
        "--teacher-cot-data",str(cache),"--output-dir",str(tmp_path/"out"),
        "--validate-only"]
    result=subprocess.run(command,capture_output=True,text=True,check=True,
        env={**os.environ,"PYTHONPATH":str(Path.cwd())})
    assert '"records": 1' in result.stdout
    assert not (tmp_path/"out").exists()
    builder=[sys.executable,"scripts/data/build_bagel_cot_teacher.py",
        "--model-path",str(model),"--prompt-data",str(source),
        "--output",str(tmp_path/"new_cot.jsonl"),"--validate-only"]
    checked=subprocess.run(builder,capture_output=True,text=True,check=True,
        env={**os.environ,"PYTHONPATH":str(Path.cwd())})
    assert '"records": 1' in checked.stdout
    assert not (tmp_path/"new_cot.jsonl").exists()
