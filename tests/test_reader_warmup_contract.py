import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from qwen_latent_cot.bagel.cot_teacher import sha256_file
from qwen_latent_cot.bagel.memory_reader import install_memory_readers
from qwen_latent_cot.bagel.reader_warmup import (
    SCHEMA, check_warmup_gate, inspect_warmup_checkpoint, load_warmup_checkpoint,
    load_warmup_records, save_warmup_checkpoint, validate_warmup_config)
from test_memory_grounding import tiny_bagel


def warmup_evidence(tmp_path,model):
    """Synthetic provenance fixture; these values are not experiment results."""
    train=tmp_path/"reader_train.jsonl"
    heldout=tmp_path/"reader_heldout.jsonl"
    train.write_text(json.dumps(dict(prompt_id="t",prompt="Two red cubes",category="count"))+"\n")
    heldout.write_text("".join(json.dumps(dict(prompt_id=f"h{i}",prompt=f"{i+3} blue spheres",
        category="count"))+"\n" for i in range(2)))
    checkpoint=tmp_path/"warmup.safetensors"
    tensors={f"language_model.model.layers.{layer}.memory_reader.output_adapter.{route}.weight":torch.zeros(8,8)
        for layer in range(12,20) for route in ("A","B")}
    save_file(tensors,str(checkpoint))
    metadata=dict(schema=SCHEMA,K=8,body=[12,20],memory_init="prompt_hidden_uniform",
        query_source="native_gen_q",memory_kv_source="frozen_read_native_kv",
        prompt_target="native_prompt_bank",generation_injection=False,
        adapter_rank=8,adapter_alpha=16,trainable_names=sorted(tensors),
        model_path=str(model.resolve()),prompt_data=str(train),heldout_prompt_data=str(heldout),
        prompt_data_sha256=sha256_file(train),heldout_prompt_data_sha256=sha256_file(heldout))
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    report=tmp_path/"warmup_gate.json"
    content=dict(schema=SCHEMA,kind="heldout_reader_gate",K=8,body=[12,20],cfg=1.0,
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_metadata_sha256=sha256_file(checkpoint.with_suffix(".json")),
        model_path=str(model.resolve()),num_steps=50,height=512,width=512,timestep_shift=3.,
        prompt_data=str(train),prompt_data_sha256=sha256_file(train),
        heldout_prompt_data=str(heldout),heldout_prompt_data_sha256=sha256_file(heldout),
        prompt_ids=["h0","h1"],initial_error_mse=2.,
        heldout_error_mse=dict(correct=1.,shuffled=3.,zero=4.),native_parity_max_abs=0.,
        per_layer={str(layer):dict(slot_mass_max=.3,slot_effective_count=6.) for layer in range(12,20)})
    report.write_text(json.dumps(content))
    return dict(model_path=str(model),prompt_data=str(train),heldout_prompt_data=str(heldout),
        output_dir=str(tmp_path/"warmup_output"),reader_warmup_checkpoint=str(checkpoint),
        reader_warmup_eval_json=str(report)),content


def test_warmup_prompt_split_validation_and_rejected_controls(tmp_path):
    model=tmp_path/"model"
    model.mkdir()
    config,_=warmup_evidence(tmp_path,model)
    source={key:value for key,value in config.items() if not key.startswith("reader_warmup")}
    valid=validate_warmup_config(source)
    train,heldout=load_warmup_records(valid)
    assert len(train)==1 and len(heldout)==2
    for overrides in (dict(loss={"type":"ranking"}),dict(loss={"lambda_shuffle":1.}),
                      dict(reader={"inject_into_generation":True}),dict(lambda_dep_shuffle=1.),
                      dict(reader={"query_source":"raw_hidden"})):
        with pytest.raises(ValueError):
            validate_warmup_config({**source,**overrides})
    alias=validate_warmup_config({**source,"num_memory_slots":8,"memory_body_start":12,"memory_body_end":20})
    assert alias["memory_loop_start_layer"]==12
    Path(config["heldout_prompt_data"]).write_text(Path(config["prompt_data"]).read_text())
    with pytest.raises(ValueError,match="disjoint"):
        load_warmup_records(valid)


@pytest.mark.parametrize("change,reason",[
    ({"heldout_error_mse":dict(correct=2.,shuffled=3.,zero=4.)},"readability"),
    ({"heldout_error_mse":dict(correct=1.,shuffled=.5,zero=4.)},"natural_specificity"),
    ({"native_parity_max_abs":.001},"native_parity"),
    ({"per_layer":{str(layer):dict(slot_mass_max=1.,slot_effective_count=1.) for layer in range(12,20)}},"slot_utilization")])
def test_opd_warmup_gate_rejects_failed_conditions(tmp_path,change,reason):
    model=tmp_path/"model"
    model.mkdir()
    config,report=warmup_evidence(tmp_path,model)
    config.update(num_steps=50,height=512,width=512,timestep_shift=3.)
    assert check_warmup_gate(config)["initial_error_mse"]==2.
    Path(config["reader_warmup_eval_json"]).write_text(json.dumps({**report,**change}))
    with pytest.raises(ValueError,match=reason):
        check_warmup_gate(config)


def test_warmup_gate_rejects_changed_checkpoint_and_prompt_provenance(tmp_path):
    model=tmp_path/"model"
    model.mkdir()
    config,_=warmup_evidence(tmp_path,model)
    config.update(num_steps=50,height=512,width=512,timestep_shift=3.)
    checkpoint=Path(config["reader_warmup_checkpoint"])
    original=checkpoint.read_bytes()
    checkpoint.write_bytes(original+b"changed")
    with pytest.raises(ValueError):
        check_warmup_gate(config)
    checkpoint.write_bytes(original)
    train=Path(config["prompt_data"])
    train.write_text(train.read_text()+"\n")
    with pytest.raises(ValueError,match="provenance"):
        check_warmup_gate(config)


def test_warmup_checkpoint_load_preserves_trained_adapter_and_zero_opd_gates(tmp_path):
    model=tiny_bagel()
    names=install_memory_readers(model,start=1,end=3,rank=2,alpha=2)
    config=dict(model_path=str(tmp_path),prompt_data=str(tmp_path/"train"),
        heldout_prompt_data=str(tmp_path/"heldout"),o_adapter_rank=2,o_adapter_alpha=2)
    Path(config["prompt_data"]).write_text("train")
    Path(config["heldout_prompt_data"]).write_text("heldout")
    runtime=SimpleNamespace(model=model,trainable_names=names,slots=2,body_start=1,body_end=3,config=config)
    with torch.no_grad():
        for name,param in model.named_parameters():
            if name in names:
                param.fill_(.25)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    path=save_warmup_checkpoint(runtime,tmp_path,7,optimizer,config)
    assert inspect_warmup_checkpoint(path)["step"]==7
    opd_model=tiny_bagel()
    gates=install_memory_readers(opd_model,start=1,end=3,rank=2,alpha=2,stage="opd")
    opd=SimpleNamespace(model=opd_model,config=config,slots=2,body_start=1,body_end=3)
    load_warmup_checkpoint(opd,path)
    named=dict(opd_model.named_parameters())
    assert all(torch.all(named[name]==.25) and not named[name].requires_grad for name in names)
    assert all(named[name]==0 and named[name].requires_grad for name in gates)
    with pytest.raises(ValueError,match="base model mismatch"):
        load_warmup_checkpoint(SimpleNamespace(**{**opd.__dict__,"config":{**config,"model_path":"wrong"}}),path)


def test_warmup_cli_preflight_does_not_load_bagel(tmp_path):
    model=tmp_path/"model"
    model.mkdir()
    config,_=warmup_evidence(tmp_path,model)
    command=[sys.executable,"scripts/train/bagel_memory_reader_warmup.py",
        "--model-path",str(model),"--prompt-data",config["prompt_data"],
        "--heldout-prompt-data",config["heldout_prompt_data"],
        "--output-dir",config["output_dir"],"--validate-only"]
    result=subprocess.run(command,capture_output=True,text=True,check=True,
        env={**os.environ,"PYTHONPATH":str(Path.cwd())})
    assert '"heldout_records": 2' in result.stdout
    assert not Path(config["output_dir"]).exists()
    command[1]="scripts/evaluate/bagel_memory_reader_warmup_eval.py"
    command += ["--checkpoint",config["reader_warmup_checkpoint"]]
    subprocess.run(command,capture_output=True,text=True,check=True,
        env={**os.environ,"PYTHONPATH":str(Path.cwd())})
    assert not Path(config["output_dir"]).exists()
