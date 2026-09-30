"""Teacher baseline and held-out causal field evaluation (never a loss)."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .cot_teacher import sha256_file
from .opd_runtime import OPDRuntime, OPDState
from .opd_training import SCHEMA


def load_reader_checkpoint(runtime, path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (meta.get("schema") != SCHEMA or meta.get("stage") != "Phase 1A.1a"
            or meta.get("body") != [12,20] or meta.get("K") != 8
            or meta.get("reader_warmup_checkpoint_sha256") != sha256_file(runtime.config["reader_warmup_checkpoint"])
            or meta.get("trainable_names") != runtime.trainable_names):
        raise ValueError("incompatible OPD reader checkpoint")
    state = load_file(str(path))
    if set(state) != set(runtime.trainable_names):
        raise ValueError("checkpoint tensor names mismatch")
    named = dict(runtime.model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != named[name].shape:
                raise ValueError(f"checkpoint tensor shape mismatch: {name}")
            named[name].copy_(value.to(named[name]))
    return meta


@torch.no_grad()
def fixed_state_metrics(runtime, records, *, seed):
    """All arms use each prompt's same detached student-policy x_t and t."""
    if len(records) < 2 or len(records) % 2:
        raise ValueError("shuffle control requires an even number of distinct prompts")
    results = []
    for pair_index in range(0, len(records), 2):
        pair = records[pair_index:pair_index+2]
        if pair[0]["prompt"] == pair[1]["prompt"]:
            raise ValueError("shuffle donor must have a different prompt")
        states = [runtime.rollout(row, seed + pair_index // 2)[0] for row in pair]
        for index, state in enumerate(states):
            memory=runtime.read_bank(state)
            donor_state=OPDState(states[1-index].condition,state.sample,state.timestep,state.step_index)
            donor_memory=runtime.read_bank(donor_state)
            teacher = runtime.teacher_velocity(state).float()
            predictions = dict(
                native=runtime.native_velocity(state),
                correct=runtime.student_velocity(state, memory_override=memory),
                shuffled=runtime.student_velocity(state, memory_override=donor_memory),
                zero=runtime.student_velocity(state, memory_override=memory.zero_like()))
            errors = {name: float((value.float()-teacher).square().mean())
                      for name, value in predictions.items()}
            results.append(dict(prompt_id=state.condition.record["prompt_id"],
                                timestep=state.timestep, step_index=state.step_index,
                                errors=errors))
    means = {name: sum(row["errors"][name] for row in results)/len(results)
             for name in ("native","correct","shuffled","zero")}
    return dict(arm_error_mse=means, per_prompt=results,
                correct_minus_native=means["correct"]-means["native"],
                correct_minus_shuffled=means["correct"]-means["shuffled"])


@torch.no_grad()
def teacher_baseline(runtime, records, *, seeds):
    """Pre-training stable field difference; does not claim semantic gain."""
    if len(set(seeds)) < 2:
        raise ValueError("teacher baseline requires at least two seeds")
    per_seed = {}
    for seed in seeds:
        diffs = []
        for row in records:
            # Reader starts at exact zero effect, so student rollout is native.
            for state in runtime.rollout(row, int(seed)):
                native = runtime.native_velocity(state).float()
                teacher = runtime.teacher_velocity(state).float()
                diffs.append(float((teacher-native).square().mean().sqrt()))
        per_seed[str(seed)] = sum(diffs)/len(diffs)
    return per_seed


@torch.no_grad()
def _generate_pair_images(runtime, records, output, *, seed, pair_offset):
    """Same noise/NFE/CFG; each arm evolves its own trajectory at every step."""
    if len(records) < 2:
        raise ValueError("five-arm generation needs two prompts for donor memory")
    output = Path(output)
    conditions = [runtime.prepare(row, seed) for row in records[:2]]
    ts, dts = runtime.model.prepare_image_schedule(int(runtime.config["num_steps"]),
        float(runtime.config["timestep_shift"]), runtime.device)
    for index, condition in enumerate(conditions):
        folder = output / f"p{pair_offset + index:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        for arm in ("native", "teacher", "correct", "shuffled", "zero"):
            x_t = condition.noise.detach().clone()
            for step, (t, dt) in enumerate(zip(ts, dts)):
                state = OPDState(condition, x_t, float(t), step)
                if arm == "native":
                    velocity = runtime.native_velocity(state)
                elif arm == "teacher":
                    velocity = runtime.teacher_velocity(state)
                elif arm == "correct":
                    velocity = runtime.student_velocity(state)
                else:
                    donor=conditions[1-index] if arm=="shuffled" else condition
                    reference=runtime.read_bank(OPDState(donor,x_t,float(t),step))
                    memory = reference if arm == "shuffled" else reference.zero_like()
                    velocity = runtime.student_velocity(state, memory_override=memory)
                x_t = runtime.model.image_euler_step(x_t, velocity, dt).detach()
            runtime.inferencer.decode_image(x_t, runtime.shape).save(folder / f"{arm}.png")
        (folder / "prompt.txt").write_text(condition.record["prompt"]+"\n")


@torch.no_grad()
def generate_five_arm_images(runtime, records, output, *, seed):
    if len(records) < 2 or len(records) % 2:
        raise ValueError("five-arm images require an even number of prompts")
    for pair_index in range(0, len(records), 2):
        _generate_pair_images(runtime, records[pair_index:pair_index+2], output,
                              seed=seed+pair_index//2, pair_offset=pair_index)


@torch.no_grad()
def generate_teacher_baseline_images(runtime,records,output,*,seed):
    """Native versus frozen Self-CoT teacher, without a MemoryReader."""
    ts,dts=runtime.model.prepare_image_schedule(int(runtime.config["num_steps"]),
        float(runtime.config["timestep_shift"]),runtime.device)
    for index,row in enumerate(records):
        condition=runtime.prepare(row,seed+index)
        folder=Path(output)/f"p{index:03d}"
        folder.mkdir(parents=True,exist_ok=False)
        for arm in ("native","teacher"):
            sample=condition.noise.detach().clone()
            for step,(t,dt) in enumerate(zip(ts,dts)):
                state=OPDState(condition,sample,float(t),step)
                velocity=runtime.native_velocity(state) if arm=="native" else runtime.teacher_velocity(state)
                sample=runtime.model.image_euler_step(sample,velocity,dt).detach()
            runtime.inferencer.decode_image(sample,runtime.shape).save(folder/f"{arm}.png")
        (folder/"prompt.txt").write_text(row["prompt"]+"\n")
