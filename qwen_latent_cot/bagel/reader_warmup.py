"""Phase 1A.0: side-head MemoryReader reconstruction before Self-CoT OPD."""

import json
import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .cot_teacher import sha256_file
from .memory_training import append_json
from .opd_runtime import OPDCondition, OPDRuntime, OPDState


SCHEMA = "bagel-memory-reader-warmup-v1"
CATEGORIES = frozenset(("count", "spatial_relation", "attribute_binding",
    "multi_object_composition", "action_relation", "rare_concept",
    "reasoning_heavy_t2i"))
RESUME_SCHEMA = "bagel-reader-warmup-resume-v1"
RESUME_CONFIG_KEYS = ("seed", "height", "width", "num_steps", "timestep_shift",
    "states_per_rollout", "timestep_bucket_weights", "num_loop_tokens",
    "memory_loop_start_layer", "memory_loop_end_layer", "o_adapter_rank",
    "o_adapter_alpha", "learning_rate", "max_grad_norm", "eval_max_prompts",
    "save_steps", "eval_steps", "reader", "loss", "memory_init",
    "memory_writer_trainable", "batch_size", "cfg_text_scale", "cfg_img_scale")


@dataclass
class ReaderWarmupOutput:
    loss: torch.Tensor
    layer_losses: dict[int, torch.Tensor]
    reader_outputs: dict[int, torch.Tensor]
    prompt_targets: dict[int, torch.Tensor]
    velocity: torch.Tensor


def validate_warmup_config(source):
    config = dict(source)
    for public, internal in (("num_memory_slots","num_loop_tokens"),
                             ("memory_body_start","memory_loop_start_layer"),
                             ("memory_body_end","memory_loop_end_layer")):
        if public in config:
            if internal in config and config[public] != config[internal]:
                raise ValueError(f"conflicting {public} and {internal}")
            config[internal] = config.pop(public)
    for key in ("model_path", "prompt_data", "heldout_prompt_data", "output_dir"):
        if not config.get(key):
            raise ValueError(f"missing {key}")
    for key in ("model_path", "prompt_data", "heldout_prompt_data"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"{key}: {config[key]}")
    locked = dict(num_loop_tokens=8, memory_loop_start_layer=12,
        memory_loop_end_layer=20, memory_init="prompt_hidden_uniform",
        memory_writer_trainable=False,batch_size=1,
        o_adapter_rank=8, o_adapter_alpha=16, cfg_text_scale=1.0,
        cfg_img_scale=1.0)
    for key, value in locked.items():
        if config.get(key, value) != value:
            raise ValueError(f"Phase 1A.0 requires {key}={value}")
        config[key] = value
    reader = config.get("reader", {})
    if not isinstance(reader, dict):
        raise ValueError("reader must be a mapping")
    required_reader = dict(query_source="native_gen_q",
        memory_kv_source="frozen_read_native_kv",
        prompt_teacher_kv_source="native_prompt_cache",
        output_source="native_gen_o", inject_into_generation=False)
    for key, value in required_reader.items():
        if reader.get(key, value) != value:
            raise ValueError(f"reader.{key} must be {value}")
    for key, value in (("adapter_rank",8),("adapter_alpha",16)):
        if reader.get(key,value) != value:
            raise ValueError(f"reader.{key} must be {value}")
    if set(reader) - set(required_reader) - {"adapter_rank","adapter_alpha"}:
        raise ValueError("unknown reader controls in warm-up config")
    forbidden = {"teacher_cot_data", "source_image", "target_image", "paired_data",
        "trainable_routes", "lambda_shuffle", "lambda_dep_shuffle", "lambda_dep_zero",
        "dependency_margin", "memory_loop_repeat", "mask_prompt_kv_during_write"}
    if set(config) & forbidden:
        raise ValueError("warm-up forbids pair/CoT/ranking/loop/mask controls")
    if set(config.get("loss", {})) - {"type","layer_weights"}:
        raise ValueError("warm-up loss must contain only MSE and uniform weights")
    if config.get("loss", {}).get("type", "mse") != "mse":
        raise ValueError("warm-up only supports symmetric bank MSE")
    if config.get("loss", {}).get("layer_weights", "uniform") != "uniform":
        raise ValueError("warm-up only supports uniform body-layer weights")
    optimizer = config.get("optimizer", {})
    if set(optimizer)-{"learning_rate","betas","weight_decay","max_grad_norm"}:
        raise ValueError("unknown warm-up optimizer controls")
    if (tuple(optimizer.get("betas",(.9,.95))) != (.9,.95)
            or float(optimizer.get("weight_decay",0.)) != 0.):
        raise ValueError("warm-up optimizer requires betas=(0.9,0.95), weight_decay=0")
    config["learning_rate"] = float(optimizer.get("learning_rate", config.get("learning_rate", 1e-4)))
    config["max_grad_norm"] = float(optimizer.get("max_grad_norm", config.get("max_grad_norm", 1.0)))
    for key, default in (("seed",42),("height",512),("width",512),
                         ("num_steps",50),("states_per_rollout",2),
                         ("max_steps",5000),("save_steps",250),("eval_steps",250)):
        config.setdefault(key, default)
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["height"] % 16 or config["width"] % 16 or config["num_steps"] < 4:
        raise ValueError("invalid resolution or NFE")
    if any(not math.isfinite(config[key]) or config[key] <= 0
           for key in ("learning_rate","max_grad_norm")):
        raise ValueError("optimizer values must be positive")
    if not 1 <= config["states_per_rollout"] <= 3:
        raise ValueError("warm-up samples 1 to 3 native states per rollout")
    config.setdefault("eval_max_prompts", 8)
    if (isinstance(config["eval_max_prompts"],bool)
            or not isinstance(config["eval_max_prompts"],int) or config["eval_max_prompts"] < 2):
        raise ValueError("eval_max_prompts must be an integer >=2")
    config.setdefault("device", "auto")
    config.setdefault("timestep_shift", 3.0)
    config.setdefault("timestep_bucket_weights", [.5,.4,.1])
    weights = config["timestep_bucket_weights"]
    if len(weights) != 3 or any(float(w) < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid timestep bucket weights")
    if any(not math.isfinite(float(w)) for w in weights) or not math.isfinite(float(config["timestep_shift"])) or config["timestep_shift"] <= 0:
        raise ValueError("schedule values must be finite and timestep_shift positive")
    if config.get("resume_checkpoint") and not Path(config["resume_checkpoint"]).is_file():
        raise FileNotFoundError(f"resume_checkpoint: {config['resume_checkpoint']}")
    return config


def load_warmup_records(config):
    def read(path, split):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty {split} prompts")
        ids, prompts = set(), set()
        for row in rows:
            if any(key in row for key in ("source_image", "target_image", "reasoning_text")):
                raise ValueError("warm-up accepts semantic T2I prompts only")
            if (not isinstance(row.get("prompt_id"), str) or not row["prompt_id"]
                    or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                    or row.get("category") not in CATEGORIES
                    or row.get("split", split) != split):
                raise ValueError(f"invalid {split} warm-up record")
            if row["prompt_id"] in ids or row["prompt"] in prompts:
                raise ValueError(f"duplicate {split} prompt")
            ids.add(row["prompt_id"])
            prompts.add(row["prompt"])
        return rows, ids, prompts
    train, train_ids, train_prompts = read(config["prompt_data"], "train")
    heldout, heldout_ids, heldout_prompts = read(config["heldout_prompt_data"], "heldout")
    if train_ids & heldout_ids or train_prompts & heldout_prompts:
        raise ValueError("warm-up train and heldout prompts must be disjoint")
    if len(heldout) < 2:
        raise ValueError("warm-up causal evaluation needs two held-out prompts")
    return train, heldout


def memory_reader_reconstruction_loss(memory_output, prompt_target):
    return F.mse_loss(memory_output.float(), prompt_target.detach().float())


def layer_reconstruction(sink, start, end):
    if set(sink) != set(range(start, end)):
        raise RuntimeError("warm-up did not collect every body layer")
    losses = {layer: memory_reader_reconstruction_loss(*sink[layer])
              for layer in range(start, end)}
    return torch.stack(list(losses.values())).mean(), losses


def inspect_warmup_checkpoint(path):
    """Validate the lightweight adapter artifact before loading BAGEL weights."""
    path=Path(path)
    meta=json.loads(path.with_suffix(".json").read_text())
    if meta.get("schema")!=SCHEMA or meta.get("generation_injection") is not False:
        raise ValueError("incompatible reader warm-up checkpoint")
    body=meta.get("body",[])
    rank=meta.get("adapter_rank")
    if len(body)!=2 or not 0 <= body[0] < body[1] or not isinstance(rank,int) or rank < 1:
        raise ValueError("invalid warm-up body/rank metadata")
    names={f"language_model.model.layers.{layer}.memory_reader.output_adapter.{route}.weight"
           for layer in range(*body) for route in ("A","B")}
    try:
        state=load_file(str(path))
    except SafetensorError as exc:
        raise ValueError(f"invalid warm-up safetensors artifact: {path}") from exc
    if set(state)!=names or meta.get("trainable_names")!=sorted(names):
        raise ValueError("warm-up checkpoint must contain only all body-layer adapters")
    widths=set()
    for layer in range(*body):
        stem=f"language_model.model.layers.{layer}.memory_reader.output_adapter"
        a,b=state[stem+".A.weight"],state[stem+".B.weight"]
        if (a.ndim!=2 or b.shape!=a.T.shape or a.shape[0]!=rank
                or a.dtype!=torch.float32 or b.dtype!=torch.float32
                or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
            raise ValueError("invalid warm-up adapter shape/dtype/values")
        widths.add(a.shape[1])
    if len(widths)!=1:
        raise ValueError("warm-up adapters have inconsistent hidden widths")
    return meta


def load_warmup_checkpoint(runtime, path):
    path = Path(path)
    meta = inspect_warmup_checkpoint(path)
    expected = dict(schema=SCHEMA, K=runtime.slots,
        body=[runtime.body_start, runtime.body_end],
        memory_init="prompt_hidden_uniform", query_source="native_gen_q",
        memory_kv_source="frozen_read_native_kv",
        prompt_target="native_prompt_bank", generation_injection=False,
        adapter_rank=int(runtime.config.get("o_adapter_rank",8)),
        adapter_alpha=int(runtime.config.get("o_adapter_alpha",16)))
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("incompatible reader warm-up checkpoint")
    if meta.get("model_path") != str(Path(runtime.config["model_path"]).resolve()):
        raise ValueError("warm-up checkpoint base model mismatch")
    state = load_file(str(path))
    named = dict(runtime.model.named_parameters())
    expected_names = {name for name in named if "memory_reader.output_adapter." in name
        and runtime.body_start <= int(name.split(".layers.")[1].split(".")[0]) < runtime.body_end}
    if set(state) != expected_names or meta.get("trainable_names") != sorted(expected_names):
        raise ValueError("warm-up checkpoint adapter tensor names mismatch")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != named[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"warm-up tensor shape mismatch: {name}")
            named[name].copy_(value.to(named[name]))
    return meta


def save_warmup_checkpoint(runtime, output, step, optimizer, config):
    stem = Path(output) / f"reader_warmup_step_{step:07d}"
    tensors = {name: p.detach().cpu().contiguous() for name, p in runtime.model.named_parameters()
               if p.requires_grad}
    if set(tensors) != set(runtime.trainable_names) or any(
            "memory_reader.output_adapter." not in name for name in tensors):
        raise RuntimeError("warm-up checkpoint contains non-adapter trainables")
    save_file(tensors, str(stem.with_suffix(".safetensors")))
    meta = dict(schema=SCHEMA, stage="Phase 1A.0", step=step, K=runtime.slots,
        body=[runtime.body_start, runtime.body_end], memory_init="prompt_hidden_uniform",
        query_source="native_gen_q", memory_kv_source="frozen_read_native_kv",
        prompt_target="native_prompt_bank", generation_injection=False,
        model_path=str(Path(config["model_path"]).resolve()),
        adapter_rank=int(config.get("o_adapter_rank",8)),
        adapter_alpha=int(config.get("o_adapter_alpha",16)),
        prompt_data=str(Path(config["prompt_data"]).resolve()),
        heldout_prompt_data=str(Path(config["heldout_prompt_data"]).resolve()),
        trainable_names=sorted(tensors), prompt_data_sha256=sha256_file(config["prompt_data"]),
        heldout_prompt_data_sha256=sha256_file(config["heldout_prompt_data"]))
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    torch.save(dict(step=step, optimizer=optimizer.state_dict()), stem.with_suffix(".optimizer.pt"))
    return stem.with_suffix(".safetensors")


def inspect_warmup_resume(config, *, world_size=None):
    """CPU-only preflight; never silently fall back to weights-only recovery."""
    path = Path(config["resume_checkpoint"])
    if Path(config["output_dir"]).exists():
        raise FileExistsError("resume requires a fresh output directory; parent run is read-only")
    meta = inspect_warmup_checkpoint(path)
    step = meta.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step < config["max_steps"]:
        raise ValueError("resume step must be positive and below total max_steps")
    expected = dict(K=config["num_loop_tokens"],
        body=[config["memory_loop_start_layer"], config["memory_loop_end_layer"]],
        adapter_rank=config["o_adapter_rank"], adapter_alpha=config["o_adapter_alpha"],
        memory_init="prompt_hidden_uniform", query_source="native_gen_q",
        memory_kv_source="frozen_read_native_kv", prompt_target="native_prompt_bank",
        model_path=str(Path(config["model_path"]).resolve()))
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("resume checkpoint model/reader contract mismatch")
    source = path.parent
    previous = json.loads((source / "resolved_config.json").read_text())
    manifest = json.loads((source / "run_manifest.json").read_text())
    for key in RESUME_CONFIG_KEYS:
        left, right = previous.get(key), config.get(key)
        if key in ("reader", "loss"):
            defaults = (dict(query_source="native_gen_q", memory_kv_source="frozen_read_native_kv",
                prompt_teacher_kv_source="native_prompt_cache", output_source="native_gen_o",
                inject_into_generation=False, adapter_rank=8, adapter_alpha=16)
                if key == "reader" else dict(type="mse", layer_weights="uniform"))
            left, right = {**defaults, **(left or {})}, {**defaults, **(right or {})}
        if left != right:
            raise ValueError(f"resume config mismatch: {key}")
    previous_world = manifest.get("world_size")
    if (isinstance(previous_world, bool) or not isinstance(previous_world, int) or previous_world < 1
            or manifest.get("effective_batch_size") != previous_world
            or manifest.get("per_rank_batch_size") != 1):
        raise ValueError("invalid resume global batch/world_size provenance")
    if world_size is not None and previous_world != world_size:
        raise ValueError("resume world_size/global batch must match the original run")
    for key in ("prompt_data", "heldout_prompt_data"):
        digest = sha256_file(config[key])
        if meta.get(key+"_sha256") != digest or manifest.get(key+"_sha256") != digest:
            raise ValueError(f"resume dataset hash mismatch: {key}")
    names = json.loads((source / "trainable_routes.json").read_text())
    if sorted(names) != meta["trainable_names"] or len(set(names)) != len(names):
        raise ValueError("resume adapter parameter ordering provenance mismatch")
    state = torch.load(path.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=True)
    if state.get("step") != step:
        raise ValueError("resume optimizer/checkpoint step mismatch")
    optimizer = state.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    if len(groups) != 1 or groups[0].get("params") != list(range(len(names))):
        raise ValueError("resume optimizer parameter layout mismatch")
    group = groups[0]
    if (group.get("lr") != config["learning_rate"] or tuple(group.get("betas", ())) != (.9, .95)
            or group.get("weight_decay") != 0. or group.get("eps") != 1e-8
            or group.get("amsgrad") is not False or group.get("maximize") is not False):
        raise ValueError("resume optimizer hyperparameters mismatch")
    buffers = optimizer.get("state", {})
    tensors = load_file(str(path))
    if set(buffers) != set(range(len(names))):
        raise ValueError("resume optimizer moments are missing")
    for index, name in enumerate(names):
        item = buffers[index]
        counter = item.get("step")
        if not torch.is_tensor(counter) or counter.numel() != 1 or float(counter) != step:
            raise ValueError("resume AdamW update counter mismatch")
        for key in ("exp_avg", "exp_avg_sq"):
            value = item.get(key)
            if (not torch.is_tensor(value) or value.shape != tensors[name].shape
                    or value.dtype != tensors[name].dtype or not bool(torch.isfinite(value).all())
                    or (key == "exp_avg_sq" and bool((value < 0).any()))):
                raise ValueError("invalid resume optimizer moments")
    initial = None
    for line in (source / "heldout_diagnostics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("step") == 0:
            initial = row
            break
    if initial is None or initial.get("native_parity_max_abs", float("inf")) > 1e-6:
        raise ValueError("resume requires the original step-zero heldout baseline/parity")
    ids = [row["prompt_id"] for row in initial.get("per_state", [])]
    expected_ids = [json.loads(line)["prompt_id"] for line in Path(config["heldout_prompt_data"]).read_text().splitlines()
                    if line.strip()][:config["eval_max_prompts"]]
    if set(ids) != set(expected_ids):
        raise ValueError("resume initial heldout prompt IDs mismatch")
    warmup_gate_checks(initial, initial)  # Validate finite diagnostics, not readiness.
    rng = None
    marker = path.with_suffix(".resume.json")
    if marker.exists():
        record = json.loads(marker.read_text())
        if (record.get("schema") != RESUME_SCHEMA or record.get("step") != step
                or record.get("world_size") != previous_world or record.get("param_names") != names):
            raise ValueError("resume sidecar contract mismatch")
        for suffix in (".safetensors", ".json", ".optimizer.pt", ".resume.pt"):
            if record.get("hashes", {}).get(suffix) != sha256_file(path.with_suffix(suffix)):
                raise ValueError("resume sidecar hash mismatch")
        for name in ("resolved_config.json", "run_manifest.json", "trainable_routes.json"):
            if record.get("source_hashes", {}).get(name) != sha256_file(source / name):
                raise ValueError("resume source provenance hash mismatch")
        digest = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        if record.get("initial_sha256") != digest:
            raise ValueError("resume initial baseline hash mismatch")
        rng = torch.load(path.with_suffix(".resume.pt"), map_location="cpu", weights_only=True)
        if not isinstance(rng, list) or len(rng) != previous_world:
            raise ValueError("resume per-rank RNG count mismatch")
        for item in rng:
            validate_warmup_rng(item)
    elif path.with_suffix(".resume.pt").exists() or manifest.get("resume_schema") == RESUME_SCHEMA:
        raise ValueError("incomplete resume sidecar: completion marker missing")
    return dict(step=step, world_size=previous_world, param_names=names,
        optimizer=optimizer, initial={key:value for key,value in initial.items() if key != "step"}, rng=rng,
        mode="per_rank_rng" if rng is not None else "legacy_seeded_rollout",
        checkpoint_sha256=sha256_file(path), optimizer_sha256=sha256_file(path.with_suffix(".optimizer.pt")))


def capture_warmup_rng(device):
    import numpy as np
    numpy_state = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(), python=random.getstate(),
        numpy=(numpy_state[0], torch.tensor(numpy_state[1].astype("int64")), *numpy_state[2:]),
        torch_cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)


def validate_warmup_rng(state):
    import numpy as np
    try:
        torch.Generator().set_state(state["torch_cpu"])
        random.Random().setstate(state["python"])
        name, keys, pos, gauss, cached = state["numpy"]
        np.random.RandomState().set_state((name, keys.numpy().astype("uint32"), pos, gauss, cached))
        cuda = state["torch_cuda"]
        if cuda is not None and (not torch.is_tensor(cuda) or cuda.ndim != 1 or cuda.dtype != torch.uint8):
            raise ValueError("invalid CUDA RNG tensor")
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise ValueError("invalid resume RNG state") from exc


def restore_warmup_rng(state, device):
    import numpy as np
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    name, keys, pos, gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype("uint32"), pos, gauss, cached))
    if torch.device(device).type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError("CUDA resume checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def save_warmup_resume_state(runtime, output, step):
    """All ranks participate after eval; write the completion/hash marker last."""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    state = capture_warmup_rng(runtime.device)
    states = [state]
    if world > 1:
        states = [None] * world
        dist.all_gather_object(states, state)
    if rank == 0:
        path = Path(output) / f"reader_warmup_step_{step:07d}.safetensors"
        rng_path = path.with_suffix(".resume.pt")
        temporary = Path(str(rng_path)+".tmp")
        torch.save(states, temporary)
        temporary.replace(rng_path)
        record = dict(schema=RESUME_SCHEMA, step=step, world_size=world,
            param_names=[name for name, param in runtime.model.named_parameters() if param.requires_grad],
            hashes={suffix:sha256_file(path.with_suffix(suffix))
                    for suffix in (".safetensors", ".json", ".optimizer.pt", ".resume.pt")},
            source_hashes={name:sha256_file(Path(output)/name)
                          for name in ("resolved_config.json", "run_manifest.json", "trainable_routes.json")})
        initial = next(json.loads(line) for line in (Path(output)/"heldout_diagnostics.jsonl").read_text().splitlines()
                       if line.strip() and json.loads(line).get("step") == 0)
        record["initial_sha256"] = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        marker = path.with_suffix(".resume.json")
        temporary = Path(str(marker)+".tmp")
        temporary.write_text(json.dumps(record, indent=2)+"\n")
        temporary.replace(marker)


class ReaderWarmupRuntime(OPDRuntime):
    @classmethod
    def load_model(cls, config):
        return super().load_model(config, stage="warmup")

    @torch.no_grad()
    def prepare(self, record, seed):
        inf = self.inferencer
        with self.autocast():
            ctx = inf.init_gen_context()
            ctx = inf.update_context_text(record["prompt"], ctx,
                capture_prompt_hidden_at=self.body_start)
            contexts = dict(full=ctx, text_removed=ctx, image_removed=ctx,
                            has_visual_condition=False)
            native = inf.prepare_velocity_bundle(name="native", contexts=contexts,
                image_shape=self.shape, num_loop_tokens=0)
            read = inf.prepare_memory_read_bundle(name="read", context=ctx,
                image_shape=self.shape, num_loop_tokens=self.slots)
        template = native.flow_input["packed_init_noises"]
        noise = torch.randn(tuple(template.shape),
            generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(template)
        return OPDCondition(record, native, native, read, ctx, noise)

    @torch.no_grad()
    def rollout(self, record, seed):
        from .loop_distill import sample_replay_step_indices
        condition = self.prepare(record, seed)
        ts, dts = self.model.prepare_image_schedule(int(self.config["num_steps"]),
            float(self.config["timestep_shift"]), self.device)
        indexes = set(sample_replay_step_indices(len(dts),
            int(self.config["states_per_rollout"]),
            generator=torch.Generator().manual_seed(seed+17),
            bucket_weights=tuple(self.config["timestep_bucket_weights"])))
        sample, states = condition.noise.detach(), []
        for index, (t, dt) in enumerate(zip(ts, dts)):
            state = OPDState(condition, sample.detach(), float(t), index)
            if index in indexes:
                states.append(OPDState(condition, sample.detach().clone(), float(t), index))
            velocity = self.native_velocity(state)
            sample = self.model.image_euler_step(sample, velocity, dt).detach()
        if len(states) != int(self.config["states_per_rollout"]):
            raise RuntimeError("native rollout did not capture requested states")
        return states

    def warmup_forward(self, state, *, memory_override=None):
        with self.autocast():
            bank = self.read_bank(state) if memory_override is None else memory_override
            output = self.model.forward_memory_reader_warmup(flow_kwargs=self._flow_kwargs(state),
                memory_read_bank=bank, memory_body_start=self.body_start,
                memory_body_end=self.body_end)
        sink = {layer: (output.reader_outputs[layer], output.prompt_targets[layer])
                for layer in output.layer_losses}
        return output.velocity, bank, sink


@torch.no_grad()
def evaluate_warmup(runtime, records, *, seed):
    if len(records) < 2:
        raise ValueError("warm-up causal evaluation needs two held-out prompts")
    totals = dict(correct=[], shuffled=[], zero=[])
    parity, per_state, layer_rows = [], [], {layer:[] for layer in range(runtime.body_start,runtime.body_end)}
    for index, row in enumerate(records):
        donor_row = records[(index+1) % len(records)]
        if donor_row["prompt"] == row["prompt"]:
            raise ValueError("shuffled donor must have a distinct prompt")
        donor_condition = runtime.prepare(donor_row, seed+index)
        for state in runtime.rollout(row, seed+index):
            correct = runtime.read_bank(state)
            # Swap only the prompt while keeping the recipient x_t and t.
            donor_state = OPDState(donor_condition,state.sample,state.timestep,state.step_index)
            donor = runtime.read_bank(donor_state)
            native = runtime.native_velocity(state).float()
            targets, errors = None, {}
            for arm, bank in (("correct",correct),("shuffled",donor),("zero",correct.zero_like())):
                velocity, _, sink = runtime.warmup_forward(state, memory_override=bank)
                loss, _ = layer_reconstruction(sink, runtime.body_start, runtime.body_end)
                errors[arm] = float(loss)
                totals[arm].append(float(loss))
                parity.append(float((velocity.float()-native).abs().max()))
                if targets is None:
                    targets = {layer: pair[1].clone() for layer,pair in sink.items()}
                elif any(not torch.equal(targets[layer],pair[1]) for layer,pair in sink.items()):
                    raise RuntimeError("warm-up controls changed the recipient prompt target/query")
                if arm == "correct":
                    diagnostics = warmup_layer_metrics(runtime,sink,bank)
                    for layer,metrics in diagnostics.items():
                        layer_rows[layer].append(metrics)
            per_state.append(dict(prompt_id=row["prompt_id"],donor_prompt_id=donor_row["prompt_id"],
                timestep=state.timestep,step_index=state.step_index,errors=errors))
    layers = {str(layer):{key:sum(row[key] for row in rows)/len(rows) for key in rows[0]}
              for layer,rows in layer_rows.items()}
    return dict(heldout_error_mse={key:sum(values)/len(values) for key,values in totals.items()},
                native_parity_max_abs=max(parity),per_layer=layers,per_state=per_state)


@torch.no_grad()
def warmup_layer_metrics(runtime,sink,bank):
    result = {}
    for layer,(readout,target) in sink.items():
        readout,target=readout.detach().float(),target.detach().float()
        error=(readout-target).square().mean().sqrt()
        target_rms=target.square().mean().sqrt()
        memory=bank.states[layer].hidden_entry
        slot_stats=runtime.model.memory_slot_stats(memory)
        reader=runtime.model.language_model.model.layers[layer].memory_reader
        result[layer]=dict(reader_mse=float(error.square()),
            relative_error=float(error/target_rms.clamp_min(1e-8)),
            cosine=float(F.cosine_similarity(readout.flatten(),target.flatten(),dim=0)),
            prompt_target_rms=float(target_rms),
            memory_hidden_rms=float(memory.float().square().mean().sqrt()),
            memory_effective_rank=slot_stats["effective_rank"],
            memory_slot_cosine=slot_stats["pairwise_cosine"],**reader.diagnostics())
    return result


def warmup_gate_checks(evaluation,initial):
    error=evaluation["heldout_error_mse"]
    values=[error[key] for key in ("correct","shuffled","zero")]
    values += [initial["heldout_error_mse"]["correct"],evaluation["native_parity_max_abs"]]
    if any(not isinstance(value,(int,float)) or isinstance(value,bool)
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("warm-up gate metrics must be finite and nonnegative")
    # Operational collapse guard: essentially one slot receives all mass.
    slots=evaluation.get("per_layer",{})
    return dict(readability=error["correct"] < initial["heldout_error_mse"]["correct"],
        natural_specificity=error["correct"] < error["shuffled"],
        native_parity=evaluation["native_parity_max_abs"] <= 1e-6,
        slot_utilization=bool(slots) and all(
            math.isfinite(row.get("slot_mass_max",float("nan")))
            and row["slot_mass_max"] < .99
            and math.isfinite(row.get("slot_effective_count",float("nan")))
            and row["slot_effective_count"] > 1.05 for row in slots.values()))


def warmup_evaluation_report(runtime,config,checkpoint_path,evaluation,initial,records):
    checks=warmup_gate_checks(evaluation,initial)
    return dict(schema=SCHEMA,kind="heldout_reader_gate",
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_metadata_sha256=sha256_file(Path(checkpoint_path).with_suffix(".json")),
        model_path=str(Path(config["model_path"]).resolve()),
        prompt_data=str(Path(config["prompt_data"]).resolve()),
        prompt_data_sha256=sha256_file(config["prompt_data"]),
        heldout_prompt_data=str(Path(config["heldout_prompt_data"]).resolve()),
        heldout_prompt_data_sha256=sha256_file(config["heldout_prompt_data"]),
        prompt_ids=[row["prompt_id"] for row in records],K=runtime.slots,
        body=[runtime.body_start,runtime.body_end],num_steps=config["num_steps"],
        cfg=1.0,height=config["height"],width=config["width"],seed=config["seed"],
        states_per_rollout=config["states_per_rollout"],timestep_shift=config["timestep_shift"],
        initial_error_mse=initial["heldout_error_mse"]["correct"],
        slot_thresholds=dict(max_slot_mass=.99,min_effective_slots=1.05),
        checks=checks,ready_for_opd=all(checks.values()),**evaluation)


def check_warmup_gate(config):
    checkpoint=Path(config["reader_warmup_checkpoint"])
    report=json.loads(Path(config["reader_warmup_eval_json"]).read_text())
    meta=inspect_warmup_checkpoint(checkpoint)
    if (meta.get("K")!=8 or meta.get("body")!=[12,20]
            or meta.get("adapter_rank")!=8 or meta.get("adapter_alpha")!=16
            or meta.get("memory_init")!="prompt_hidden_uniform"
            or meta.get("query_source")!="native_gen_q"
            or meta.get("memory_kv_source")!="frozen_read_native_kv"
            or meta.get("prompt_target")!="native_prompt_bank"):
        raise ValueError("incompatible reader warm-up checkpoint")
    expected=dict(schema=SCHEMA,kind="heldout_reader_gate",K=8,body=[12,20],cfg=1.0,
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_metadata_sha256=sha256_file(checkpoint.with_suffix(".json")),
        model_path=str(Path(config["model_path"]).resolve()),num_steps=config["num_steps"],
        height=config["height"],width=config["width"],timestep_shift=config["timestep_shift"])
    if any(report.get(key)!=value for key,value in expected.items()):
        raise ValueError("warm-up heldout evidence does not match this checkpoint/run")
    source_config=dict(prompt_data=report.get("prompt_data"),
                       heldout_prompt_data=report.get("heldout_prompt_data"))
    if not all(source_config.values()):
        raise ValueError("warm-up gate requires original prompt split provenance")
    train,heldout=load_warmup_records(source_config)
    for key in ("prompt_data","heldout_prompt_data"):
        digest=sha256_file(source_config[key])
        if report.get(key+"_sha256")!=digest or meta.get(key+"_sha256")!=digest:
            raise ValueError("warm-up prompt split provenance mismatch")
    ids=report.get("prompt_ids",[])
    if len(ids)<2 or len(ids)!=len(set(ids)) or not set(ids).issubset({row["prompt_id"] for row in heldout}):
        raise ValueError("warm-up gate needs distinct heldout prompt ids")
    if set(report.get("per_layer",{}))!={str(layer) for layer in range(12,20)}:
        raise ValueError("warm-up gate is missing layer utilization metrics")
    checks=warmup_gate_checks(report,{"heldout_error_mse":{"correct":report["initial_error_mse"]}})
    if not all(checks.values()):
        raise ValueError("warm-up is not ready for OPD: "+", ".join(key for key,value in checks.items() if not value))
    return report


def synchronize_reader_parameters(params):
    """Synchronize only the small trainable side head, never the frozen BAGEL."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        for param in params:
            dist.broadcast(param.data, src=0)


def average_reader_gradients(params):
    """Custom-method forwards bypass DDP; explicitly average their gradients."""
    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    if dist.is_initialized() and dist.get_world_size() > 1:
        flag = torch.tensor(int(finite), device=params[0].device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        finite = bool(flag.item())
    if not finite:
        raise FloatingPointError("missing/nonfinite adapter gradient on at least one rank")
    if dist.is_initialized() and dist.get_world_size() > 1:
        packed = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(packed)
        packed.div_(dist.get_world_size())
        offset = 0
        for param in params:
            param.grad.copy_(packed[offset:offset+param.numel()].view_as(param))
            offset += param.numel()


def train_warmup(config, train_records, heldout_records):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    primary = rank == 0
    def barrier():
        if world > 1:
            dist.barrier()
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite warm-up output: {output}")
    resume = inspect_warmup_resume(config, world_size=world) if config.get("resume_checkpoint") else None
    start_step = resume["step"] if resume else 0
    barrier()
    runtime = ReaderWarmupRuntime.load_model(config)
    if not runtime.trainable_names or any("memory_reader.output_adapter." not in name
                                          for name in runtime.trainable_names):
        raise RuntimeError("warm-up must train only reader translation adapters")
    if resume:
        names = [name for name, param in runtime.model.named_parameters() if param.requires_grad]
        if names != resume["param_names"]:
            raise ValueError("resume live optimizer parameter ordering mismatch")
        load_warmup_checkpoint(runtime, config["resume_checkpoint"])
    if primary:
        output.mkdir(parents=True, exist_ok=False)
        (output / "resolved_config.json").write_text(json.dumps(config,indent=2)+"\n")
        (output / "trainable_routes.json").write_text(json.dumps(runtime.trainable_names,indent=2)+"\n")
        (output / "run_manifest.json").write_text(json.dumps(dict(schema=SCHEMA,
            model_path=str(Path(config["model_path"]).resolve()),seed=config["seed"],
            prompt_data_sha256=sha256_file(config["prompt_data"]),
            heldout_prompt_data_sha256=sha256_file(config["heldout_prompt_data"]),
            world_size=world,per_rank_batch_size=1,effective_batch_size=world,
            gradient_reduction="mean_before_clipping",eval_rank=0,
            code_commit=os.environ.get("BAGEL_CODE_COMMIT"),
            torch_version=torch.__version__,generation_injection=False,
            resume_schema=RESUME_SCHEMA, start_step=start_step,
            resume_checkpoint=str(Path(config["resume_checkpoint"]).resolve()) if resume else None,
            resume_mode=resume["mode"] if resume else None,
            resume_checkpoint_sha256=resume["checkpoint_sha256"] if resume else None,
            resume_optimizer_sha256=resume["optimizer_sha256"] if resume else None),indent=2)+"\n")
    barrier()
    params = [p for p in runtime.model.parameters() if p.requires_grad]
    synchronize_reader_parameters(params)
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"],
        betas=(.9,.95), weight_decay=0.)
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
    heldout = heldout_records[:max(2, int(config.get("eval_max_prompts", 8)))]
    status = dict(status="running", step=start_step)
    if primary:
        (output / "status.json").write_text(json.dumps(status)+"\n")
    try:
        if primary:
            initial = resume["initial"] if resume else evaluate_warmup(runtime, heldout, seed=int(config["seed"]))
            append_json(output / "heldout_diagnostics.jsonl", dict(step=0, **initial))
            if initial["native_parity_max_abs"] > 1e-6:
                raise RuntimeError("warm-up side head changed native velocity")
            print(json.dumps(dict(event="resume_loaded" if resume else "initial_eval_complete",
                step=start_step, world_size=world, resume_mode=resume["mode"] if resume else None)),flush=True)
        barrier()
        if resume and resume["rng"] is not None:
            restore_warmup_rng(resume["rng"][rank], runtime.device)
        for step in range(start_step+1, config["max_steps"]+1):
            record = train_records[((step-1)*world+rank) % len(train_records)]
            states = runtime.rollout(record, int(config["seed"])+step*100003+rank)
            optimizer.zero_grad(set_to_none=True)
            losses, layer_metrics, memory_metrics = [], [], []
            for state in states:
                velocity, bank, sink = runtime.warmup_forward(state)
                loss, layers = layer_reconstruction(sink, runtime.body_start, runtime.body_end)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite reader loss at step {step}")
                (loss/len(states)).backward()
                losses.append(float(loss.detach()))
                layer_metrics.append(warmup_layer_metrics(runtime,sink,bank))
                memory_metrics.append(dict(runtime.model.last_opd_memory_stats))
            average_reader_gradients(params)
            norm = torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"],
                error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step,prompt_id=record["prompt_id"],
                loss_reader_mse=sum(losses)/len(losses),grad_norm=float(norm),
                state_steps=[state.step_index for state in states],
                per_layer={str(layer):{key:sum(row[layer][key] for row in layer_metrics)/len(states)
                    for key in layer_metrics[0][layer]} for layer in layer_metrics[0]},
                **{key:sum(row[key] for row in memory_metrics)/len(states) for key in memory_metrics[0]})
            rows = [row]
            if world > 1:
                rows = [None]*world
                dist.all_gather_object(rows,row)
            if primary:
                row = dict(row)
                row["loss_reader_mse"] = sum(item["loss_reader_mse"] for item in rows)/world
                row["prompt_ids"] = [item["prompt_id"] for item in rows]
                row["state_steps_by_rank"] = [item["state_steps"] for item in rows]
                row["world_size"] = world
                row["per_rank_metrics"] = rows if world > 1 else None
                # Retain local diagnostics under per_rank_metrics; top-level loss is global.
                append_json(output / "metrics.jsonl",row)
            checkpoint_path = None
            if primary and (step % config["save_steps"] == 0 or step % config["eval_steps"] == 0 or step == config["max_steps"]):
                checkpoint_path = save_warmup_checkpoint(runtime, output, step, optimizer, config)
            if primary and (step % config["eval_steps"] == 0 or step == config["max_steps"]):
                evaluation = evaluate_warmup(runtime, heldout, seed=int(config["seed"]))
                append_json(output / "heldout_diagnostics.jsonl", dict(step=step,**evaluation))
                report=warmup_evaluation_report(runtime,config,checkpoint_path,evaluation,initial,heldout)
                (output / "warmup_gate.json").write_text(json.dumps(report,indent=2)+"\n")
                if evaluation["native_parity_max_abs"] > 1e-6:
                    raise RuntimeError("warm-up side head changed native velocity")
            if (step % config["save_steps"] == 0 or step % config["eval_steps"] == 0
                    or step == config["max_steps"]):
                save_warmup_resume_state(runtime, output, step)
            if primary:
                print(json.dumps(dict(step=step,loss_reader_mse=row["loss_reader_mse"],world_size=world)),flush=True)
                (output / "status.json").write_text(json.dumps(dict(status="running",step=step))+"\n")
            barrier()
        if primary:
            status = dict(status="complete", step=config["max_steps"],
                ready_for_opd=report["ready_for_opd"], checks=report["checks"],
                warmup_gate_json=str(output / "warmup_gate.json"))
    except Exception as exc:
        status = dict(status="failed", step=step if "step" in locals() else start_step,
            error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if primary:
            (output / "status.json").write_text(json.dumps(status,indent=2)+"\n")
