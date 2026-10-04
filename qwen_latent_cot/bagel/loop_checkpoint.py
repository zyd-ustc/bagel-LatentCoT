"""Strict loop-only checkpoints. Native checkpoint remains an external anchor."""

import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file

FORMAT = "umm-t2i-anchored-loop-v3"


def save_loop_checkpoint(
    model,
    directory,
    *,
    step,
    model_path,
    training_depth_counts=None,
    round_training_steps=None,
    depth_curriculum=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.t2i_loop.state_dict().items()
        },
        str(directory / "loop.safetensors"),
    )
    metadata = {
        "format": FORMAT,
        "step": step,
        "model_path": str(Path(model_path).resolve()),
        "loop_config": asdict(model.t2i_loop.config),
        "native_timestep_shift": model.timestep_shift,
        "flow_timestep_distribution": "sigmoid_normal_then_native_shift",
        "alpha_semantics": "shared_across_runtime_depths",
        "gate_semantics": "native_gen_reference_plus_gated_loop_correction",
    }
    if training_depth_counts is not None:
        metadata["training_depth_counts"] = dict(training_depth_counts)
        metadata["round_training_steps"] = list(round_training_steps)
        metadata["depth_curriculum"] = depth_curriculum
    (directory / "loop.json").write_text(json.dumps(metadata, indent=2) + "\n")


def checkpoint_config(directory):
    metadata = json.loads((Path(directory) / "loop.json").read_text())
    if metadata.get("format") != FORMAT:
        raise ValueError("checkpoint belongs to an incompatible loop architecture")
    return metadata


def load_loop_checkpoint(model, directory):
    metadata = checkpoint_config(directory)
    current = asdict(model.t2i_loop.config)
    for key in (
        "loop_start_layer",
        "loop_end_layer",
        "allocated_max_loop_depth",
        "memory_slots",
        "reentry_rank",
        "reentry_adapter_type",
        "loop_mode",
    ):
        if metadata["loop_config"][key] != current[key]:
            raise ValueError(f"loop checkpoint configuration mismatch: {key}")
    model.t2i_loop.load_state_dict(
        load_file(str(Path(directory) / "loop.safetensors")), strict=True
    )
    return metadata
