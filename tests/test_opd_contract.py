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
from qwen_latent_cot.bagel.memory_read_bank import LayerMemoryState,MemoryReadBank
from test_reader_warmup_contract import warmup_evidence


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
    row=_row(prompt="One cube and three spheres")
    row["reasoning_text"]=row["reasoning_text"].replace(
        "three red cubes and one blue sphere", "three spheres")
    with pytest.raises(ValueError, match="omits"):
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
    config=validate_opd_config(base,require_warmup=False)
    assert load_training_records(config)[0]["prompt_id"] == "x"
    with pytest.raises(ValueError,match="forbids"):
        validate_opd_config({**base,"lambda_dep_shuffle":0.5},require_warmup=False)
    gate=tmp_path/"baseline.json"
    gate.write_text(json.dumps(dict(schema=SCHEMA,kind="teacher_baseline",
        teacher_cache_sha256=sha256_file(cache),model_path=str(model.resolve()),
        num_steps=50,cfg=1.0,prompt_ids=[str(i) for i in range(8)],
        field_rms_by_seed={"42":.02,"43":.03})))
    with pytest.raises(ValueError, match="held-out semantic_evidence"):
        check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})
    debug=validate_opd_config({**base,"max_steps":10,"allow_field_only_debug":True},require_warmup=False)
    assert check_teacher_baseline({**debug,"teacher_baseline_json":str(gate)})[
        "training_gate"] == "field_only_debug_not_semantic_improvement"
    with pytest.raises(ValueError, match="at most 10 steps"):
        validate_opd_config({**base,"max_steps":11,"allow_field_only_debug":True},require_warmup=False)
    heldout=tmp_path/"heldout.jsonl"
    heldout.write_text("".join(json.dumps({"prompt":f"held-out semantic prompt {i}"})+"\n"
                               for i in range(8)))
    report=tmp_path/"scores.json"
    report.write_text(json.dumps({"note":"independent semantic scoring report"}))
    evidence=dict(heldout_prompt_data=str(heldout),heldout_prompt_sha256=sha256_file(heldout),
        score_report=str(report),score_report_sha256=sha256_file(report),scorer="human",
        model_path=str(model.resolve()),num_steps=50,cfg=1.0,prompt_count=8,
        native_score=.50,teacher_score=.60)
    baseline=json.loads(gate.read_text())
    baseline["semantic_evidence"]=evidence
    gate.write_text(json.dumps(baseline))
    assert check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})[
        "semantic_teacher_minus_native"] == pytest.approx(.10)
    baseline["semantic_evidence"]["teacher_score"] = .49
    gate.write_text(json.dumps(baseline))
    with pytest.raises(ValueError,match="must exceed native"):
        check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})
    baseline["semantic_evidence"]["teacher_score"] = .60
    heldout.write_text(heldout.read_text()+json.dumps({"prompt":_row()["prompt"]})+"\n")
    baseline["semantic_evidence"]["heldout_prompt_sha256"] = sha256_file(heldout)
    baseline["semantic_evidence"]["prompt_count"] = 9
    gate.write_text(json.dumps(baseline))
    with pytest.raises(ValueError,match="distinct held-out"):
        check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})
    cache.write_text(cache.read_text()+"\n")
    with pytest.raises(ValueError,match="does not match"):
        check_teacher_baseline({**config,"teacher_baseline_json":str(gate)})


def test_evaluation_controls_use_same_state_and_never_enter_loss():
    class Runtime:
        calls=[]
        def rollout(self, row, seed):
            from types import SimpleNamespace
            condition=SimpleNamespace(record=row)
            return [SimpleNamespace(condition=condition, sample=torch.tensor([1. if row["prompt_id"]=="a" else 3.]),
                                    timestep=.5, step_index=3)]
        def read_bank(self,state):
            self.calls.append((state.condition.record["prompt_id"],float(state.sample),state.timestep))
            value=1. if state.condition.record["prompt_id"]=="a" else 2.
            return MemoryReadBank({0:LayerMemoryState(0,torch.tensor([[value]]),
                torch.tensor([[[value]]]),torch.tensor([[[value]]]))})
        def student_velocity(self, state, *, return_memory=False, memory_override=None):
            memory=self.read_bank(state) if memory_override is None else memory_override
            if return_memory:
                return state.sample+memory.states[0].value.mean(),memory
            return state.sample+memory.states[0].value.mean()
        def teacher_velocity(self, state):
            return state.sample+1
        def native_velocity(self, state):
            return state.sample
    rows=[dict(prompt_id="a",prompt="a prompt"),dict(prompt_id="b",prompt="b prompt")]
    runtime=Runtime()
    result=fixed_state_metrics(runtime,rows,seed=42)
    assert result["per_prompt"][0]["errors"] == dict(native=1.,correct=0.,shuffled=1.,zero=1.)
    assert result["per_prompt"][1]["errors"] == dict(native=1.,correct=1.,shuffled=0.,zero=1.)
    assert runtime.calls==[("a",1.,.5),("b",1.,.5),("b",3.,.5),("a",3.,.5)]
    many=rows+[dict(prompt_id="c",prompt="c prompt"),dict(prompt_id="d",prompt="d prompt")]
    assert len(fixed_state_metrics(Runtime(),many,seed=42)["per_prompt"])==4
    with pytest.raises(ValueError,match="even number"):
        fixed_state_metrics(Runtime(),many[:3],seed=42)


def test_opd_cli_preflight_does_not_load_bagel_weights(tmp_path):
    model=tmp_path/"empty_model_directory"
    model.mkdir()
    warmup,_=warmup_evidence(tmp_path,model)
    source=tmp_path/"prompts.jsonl"
    cache=tmp_path/"cot.jsonl"
    source.write_text(json.dumps(dict(id="x",prompt=_row()["prompt"],category="count"))+"\n")
    cache.write_text(json.dumps(_row())+"\n")
    command=[sys.executable,"scripts/train/bagel_memory_opd.py",
        "--config","configs/training/memory_opd_t0.yaml",
        "--model-path",str(model),"--prompt-data",str(source),
        "--teacher-cot-data",str(cache),"--output-dir",str(tmp_path/"out"),
        "--reader-warmup-checkpoint",warmup["reader_warmup_checkpoint"],
        "--reader-warmup-eval-json",warmup["reader_warmup_eval_json"],
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


def test_formal_opd_requires_warmup_artifact_gate_and_gate_only_controls(tmp_path):
    model=tmp_path/"model"
    model.mkdir()
    warmup,_=warmup_evidence(tmp_path,model)
    source=tmp_path/"opd_train.jsonl"
    teacher=tmp_path/"cot.jsonl"
    source.write_text(json.dumps(dict(prompt_id="x",prompt=_row()["prompt"],category="count"))+"\n")
    teacher.write_text(json.dumps(_row())+"\n")
    base=dict(model_path=str(model),prompt_data=str(source),teacher_cot_data=str(teacher),
              output_dir=str(tmp_path/"opd"))
    with pytest.raises(ValueError,match="reader_warmup_checkpoint"):
        validate_opd_config(base)
    controls=dict(load_warmup=True,injection_gate_init=0.,train_gate=True,train_adapter=False,train_q_lora=False)
    valid=validate_opd_config({**base,"reader_warmup_checkpoint":warmup["reader_warmup_checkpoint"],
                              "reader_warmup_eval_json":warmup["reader_warmup_eval_json"],"memory_reader":controls})
    assert valid["learning_rate"]==1e-3
    for key,value in (("load_warmup",False),("injection_gate_init",.01),("train_gate",False),
                      ("train_adapter",True),("train_q_lora",True)):
        with pytest.raises(ValueError,match="requires memory_reader"):
            validate_opd_config({**valid,"memory_reader":{**controls,key:value}})
