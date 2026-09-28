from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import json
import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from qwen_latent_cot.bagel.memory_training import (
    ReplayItem, GroundingRuntime, validate_adapter_metadata, SCHEMA,
)
from qwen_latent_cot.bagel.memory_grpo import rollout_arms, validate_grpo_config
from qwen_latent_cot.bagel.memory_stage_runner import new_output
from qwen_latent_cot.bagel.modeling.bagel.bagel import Bagel


class ToyRuntime:
    """Only trajectory orchestration is mocked; real SDE and losses are used."""
    device = torch.device("cpu")
    stage = "eval"
    shape = (16,16)
    config = dict(num_steps=4, num_write_rounds=1, batch_size=2, seed=42,
                  objective="native_teacher", states_per_prompt=2)
    def __init__(self):
        self.model = SimpleNamespace(loop_memory=torch.zeros(8,2),
            prepare_image_schedule=Bagel.prepare_image_schedule, image_euler_step=Bagel.image_euler_step)
        self.inferencer = SimpleNamespace(decode_image=lambda x,s: Image.new("RGB",s,"red"))
        self.calls = []
    def autocast(self):
        return nullcontext()
    def conditions(self, record, seed):
        x = torch.randn(2,2,generator=torch.Generator().manual_seed(seed))
        return ReplayItem(record,None,None,None,x,1.,x.clone()), x
    def native_velocity(self, item):
        return item.sample * .1
    def read_memory(self, item, detach=False):
        return torch.ones(8,2) * float(item.record["id"])
    def write(self, item, memory=None, **kwargs):
        self.calls.append((item, memory.clone() if memory is not None else None, kwargs))
        v = item.sample*.1 + (self.read_memory(item) if memory is None else memory).mean() * .01
        return SimpleNamespace(final_velocity=v, write_round_velocities=(v,))
    states = GroundingRuntime.states
    dependency = GroundingRuntime.dependency


def test_native_rollout_reuses_states_and_shares_batch_timesteps():
    runtime = ToyRuntime()
    a = runtime.states({"id":"1"}, 42, state_seed=99)
    b = runtime.states({"id":"2"}, 43, state_seed=99)
    assert len(a) == len(b) == 2
    assert [it.timestep for it in a] == [it.timestep for it in b]
    assert all(torch.equal(it.target,it.sample*.1) for it in a+b)
    assert not torch.equal(a[0].sample,b[0].sample)


def test_four_arm_rollout_fixed_noise_and_write_entry_counterfactual():
    runtime = ToyRuntime()
    records = [{"id":"1"},{"id":"2"}]
    samples, trajectories, items, hashes = rollout_arms(runtime, records, 42, stochastic_steps=(1,))
    assert set(samples) == {"native","zero","shuffled","correct"}
    assert len(hashes) == 2 and len(trajectories[0]) == 1
    assert torch.isfinite(trajectories[0][0]["old_log_prob"])
    first = runtime.calls[:6]
    # zero pair, shuffled pair, correct pair: identical starting x for recipient.
    assert torch.equal(first[0][0].sample,first[2][0].sample)
    assert torch.equal(first[0][0].sample,first[4][0].sample)
    assert first[0][1].count_nonzero() == 0
    assert torch.equal(first[2][1],torch.ones(8,2)*2)
    assert first[4][1] is None
    samples2, _, _, hashes2 = rollout_arms(ToyRuntime(),records,42,stochastic_steps=(1,))
    assert hashes == hashes2
    assert all(torch.equal(samples[a][0],samples2[a][0]) for a in samples)


def test_evaluation_writes_images_metrics_and_does_not_claim_semantic_score(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("memory_eval_test", root/"scripts/evaluate/bagel_memory_causality_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.evaluate(ToyRuntime(),[{"id":"1","prompt":"one"},{"id":"2","prompt":"two"}],
                    tmp_path,generate_images=True)
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["prompts"] == 2 and not summary["semantic_scored"]
    assert summary["states"] == 4
    assert len(list(tmp_path.glob("p*/*.png"))) == 8
    assert (tmp_path/"index.html").is_file()
    metrics = [json.loads(line) for line in (tmp_path/"metrics.jsonl").read_text().splitlines()]
    assert all(row["prompt_mask"] is False for row in metrics)


def test_checkpoint_rejects_legacy_or_wrong_stage_before_model_load(tmp_path):
    adapter = tmp_path/"a.safetensors"
    adapter.touch()
    sidecar = adapter.with_suffix(".json")
    sidecar.write_text(json.dumps(dict(schema="old")))
    with pytest.raises(ValueError,match="metadata"):
        validate_adapter_metadata(dict(adapter_path=str(adapter)),"writer")
    meta = dict(schema=SCHEMA,K=8,body=[12,20],writer_read_only=True,
                lora_rank=8,lora_alpha=16,reader_o_enabled=False,stage="reader")
    sidecar.write_text(json.dumps(meta))
    assert validate_adapter_metadata(dict(adapter_path=str(adapter)),"writer")["stage"] == "reader"
    with pytest.raises(ValueError,match="initialize"):
        validate_adapter_metadata(dict(adapter_path=str(adapter)),"loop")


def test_output_protection_and_provenance(tmp_path):
    data = tmp_path/"prompts.jsonl"
    data.write_text('{"prompt":"test"}\n')
    c = dict(data_path=str(data),output_dir=str(tmp_path/"run"))
    output = new_output(c)
    manifest = json.loads((output/"run_manifest.json").read_text())
    assert manifest["input_sha256"]["data_path"]
    assert "qwen_latent_cot/bagel/memory_training.py" in manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        new_output(c)


def test_grpo_rejects_missing_reward_contract():
    with pytest.raises(ValueError,match="real GenEval"):
        validate_grpo_config({},[])
