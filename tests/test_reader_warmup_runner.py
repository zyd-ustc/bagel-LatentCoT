"""CPU runner wiring smoke with real tiny MoT; no full-model result claim."""

import json
from pathlib import Path

import torch

from qwen_latent_cot.bagel.memory_reader import install_memory_readers
from qwen_latent_cot.bagel.opd_runtime import OPDCondition,OPDState
from qwen_latent_cot.bagel.reader_warmup import (
    ReaderWarmupRuntime,inspect_warmup_checkpoint,train_warmup)
from test_memory_grounding import tiny_bagel
from test_memory_read_bank import one_sample_flow,reader_condition


class TinyWarmupRuntime(ReaderWarmupRuntime):
    def __init__(self,config):
        model=tiny_bagel()
        names=install_memory_readers(model,start=1,end=3,rank=2,alpha=2)
        super().__init__(config,model,None,None,"cpu",names)
        self.training_overrides=[]

    def prepare(self,row,seed):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            flow=one_sample_flow()
            condition=reader_condition(flow)
        prompt=dict(prompt_hidden=condition["prompt_hidden"],prompt_mask=condition["prompt_mask"],
            content_mask=condition["content_mask"])
        return OPDCondition(row,flow,flow,condition["read_kwargs"],prompt,flow["x_t"])

    def rollout(self,row,seed):
        condition=self.prepare(row,seed)
        return [OPDState(condition,condition.noise.detach().clone(),t,index)
                for index,t in enumerate((.75,.25))]

    def _flow_kwargs(self,state,*,teacher=False):
        return {**state.condition.native,"x_t":state.sample,
                "timestep":self.model._reader_timestep(state.sample,state.timestep)}

    def _read_kwargs(self,state):
        return {**state.condition.read,"x_t":state.sample,
                "timestep":self.model._reader_timestep(state.sample,state.timestep)}

    def warmup_forward(self,state,*,memory_override=None):
        if state.condition.record.get("split")=="train":
            self.training_overrides.append(memory_override)
        return super().warmup_forward(state,memory_override=memory_override)


def test_short_warmup_runner_saves_adapter_metrics_and_gate_without_training_shuffle(tmp_path,monkeypatch):
    train=[dict(prompt_id="t",prompt="Two red cubes",category="count",split="train")]
    heldout=[dict(prompt_id=f"h{i}",prompt=f"{i+3} spheres",category="count",split="heldout")
             for i in range(2)]
    source=tmp_path/"train.jsonl"
    validation=tmp_path/"heldout.jsonl"
    source.write_text("".join(json.dumps(row)+"\n" for row in train))
    validation.write_text("".join(json.dumps(row)+"\n" for row in heldout))
    config=dict(model_path=str(tmp_path),prompt_data=str(source),heldout_prompt_data=str(validation),
        output_dir=str(tmp_path/"run"),height=32,width=32,num_steps=4,timestep_shift=3.,
        seed=42,states_per_rollout=2,num_loop_tokens=2,memory_loop_start_layer=1,
        memory_loop_end_layer=3,o_adapter_rank=2,o_adapter_alpha=2,
        learning_rate=1e-4,max_grad_norm=1.,max_steps=2,save_steps=1,eval_steps=1,
        eval_max_prompts=2)
    runtime=TinyWarmupRuntime(config)
    frozen={name:p.detach().clone() for name,p in runtime.model.named_parameters() if not p.requires_grad}
    monkeypatch.setattr(ReaderWarmupRuntime,"load_model",classmethod(lambda cls,config:runtime))
    train_warmup(config,train,heldout)
    output=Path(config["output_dir"])
    assert json.loads((output/"status.json").read_text())["status"]=="complete"
    metrics=[json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
    assert len(metrics)==2 and set(metrics[0]["per_layer"])=={"1","2"}
    assert "reader_mse" in metrics[0]["per_layer"]["1"]
    assert len(runtime.training_overrides)==4 and all(value is None for value in runtime.training_overrides)
    artifact=output/"reader_warmup_step_0000002.safetensors"
    assert inspect_warmup_checkpoint(artifact)["step"]==2
    report=json.loads((output/"warmup_gate.json").read_text())
    assert report["native_parity_max_abs"]==0. and len(report["per_state"])==4
    assert set(report["checks"])=={"readability","natural_specificity","native_parity","slot_utilization"}
    named=dict(runtime.model.named_parameters())
    assert all(torch.equal(value,named[name]) for name,value in frozen.items())
