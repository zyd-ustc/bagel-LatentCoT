"""Shared v2 runtime: fixed-state A/B/D replay and four-arm inference.

Batch members are replayed sequentially to limit KV/activation memory. Shuffling
still swaps complete Read states across distinct prompts, never across slots.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from . import accelerator
from .loop import configure_loop_trainable_routes, load_loop_adapter_state_dict, loop_adapter_state_dict
from .memory_grounding import adapters_off, memory_dependency_loss, shuffle_across_batch

SCHEMA = "bagel-memory-grounding-v2"
ARMS = ("native", "zero", "shuffled", "correct")


def validate_config(config, stage):
    c = dict(config)
    if stage not in ("reader", "writer", "grpo", "loop", "eval"):
        raise ValueError(f"unknown stage: {stage}")
    locked = dict(num_loop_tokens=8, memory_loop_start_layer=12, memory_loop_end_layer=20,
                  num_read_rounds=1, loop_memory_persist=False, loop_recycle_mode="same_depth",
                  loop_update_mode="plain", lora_dropout=0.0, k_v_lora=False,
                  round0_memory_write_enabled=False, cfg_text_scale=1.0, cfg_img_scale=1.0)
    for key, value in locked.items():
        if c.get(key, value) != value:
            raise ValueError(f"v2 requires {key}={value}")
        c[key] = value
    defaults = dict(reader=["q_proj_moe_gen"], writer=["q_proj"],
                    grpo=["q_proj_moe_gen"], loop=["q_proj", "q_proj_moe_gen"], eval=[])
    c.setdefault("trainable_routes", defaults[stage])
    routes = set(c["trainable_routes"])
    allowed = set(defaults[stage])
    if stage in ("reader", "grpo") and c.get("reader_o_enabled", False):
        allowed.add("o_proj_moe_gen")
    if stage == "grpo" and c.get("joint_writer", False):
        allowed.add("q_proj")
    if stage != "eval" and (not routes or not routes <= allowed):
        raise ValueError(f"invalid {stage} trainable routes: {sorted(routes)}")
    if "o_proj_moe_gen" in routes and not c.get("reader_dependency_validated", False):
        raise ValueError("GEN-O requires a validated GEN-Q reader first")
    c.setdefault("num_write_rounds", 3 if stage == "loop" else 1)
    w = c["num_write_rounds"]
    if isinstance(w, bool) or not isinstance(w, int) or w < 1:
        raise ValueError("num_write_rounds must be a positive integer")
    if stage in ("reader", "writer", "grpo") and w != 1:
        raise ValueError("grounding and initial GRPO require one Write")
    if c.get("loop_distill", False) and not c.get("final_round_validated", False):
        raise ValueError("loop distillation requires held-out final-round validation")
    c.setdefault("objective", "native_teacher")
    if c["objective"] not in ("native_teacher", "target_flow"):
        raise ValueError("objective must be native_teacher or target_flow")
    if stage in ("reader", "grpo") and c["objective"] != "native_teacher":
        raise ValueError(f"{stage} uses native rollout states, not pair hidden targets")
    c.setdefault("state_source", "native_rollout" if c["objective"] == "native_teacher" else "target_flow")
    if c["state_source"] != ("native_rollout" if c["objective"] == "native_teacher" else "target_flow"):
        raise ValueError("state_source must match objective")
    if stage in ("writer", "grpo", "loop"):
        if not c.get("adapter_path"):
            raise ValueError("this stage requires adapter_path from the preceding stage")
        if not c.get("reader_dependency_validated", False):
            raise ValueError("preceding held-out correct-vs-shuffle gate has not been accepted")
    if stage == "loop" and not c.get("semantic_quality_validated", False):
        raise ValueError("loop supervision requires held-out semantic/quality validation")
    for key in ("model_path", "data_path", "output_dir"):
        if not c.get(key):
            raise ValueError(f"missing {key}")
    c.setdefault("batch_size", 2)
    if c["batch_size"] < 2:
        raise ValueError("counterfactual training/evaluation requires batch_size>=2")
    for key, default in (("max_steps", 5000), ("save_steps", 100), ("num_steps", 50),
                         ("states_per_prompt", 3), ("height", 512), ("width", 512)):
        c.setdefault(key, default)
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["num_steps"] < 3 or c["states_per_prompt"] > c["num_steps"] - 1:
        raise ValueError("native rollout needs enough distinct steps for states_per_prompt")
    if c["height"] % 16 or c["width"] % 16:
        raise ValueError("image dimensions must be multiples of 16")
    if stage == "loop":
        weights = c.get("write_round_weights", [1.] * w)
        if len(weights) != w or any(v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("write_round_weights must match num_write_rounds and have a positive sum")
    return c


def load_records(config):
    if config["objective"] == "target_flow":
        from qwen_latent_cot.data.phase1_pairs import load_phase1_pairs
        rows = load_phase1_pairs(config["data_path"])
        rows = [{**r, "prompt": r["instruction"]} for r in rows]
    else:
        path = Path(config["data_path"])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for i, row in enumerate(rows):
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"record {i} has no prompt")
            row.setdefault("id", str(i))
    if config.get("max_prompts"):
        rows = rows[:int(config["max_prompts"])]
    condition_keys = [(r["source_image"], r["prompt"]) if config["objective"] == "target_flow"
                      else r["prompt"] for r in rows]
    if len(rows) < config["batch_size"] or len(set(condition_keys)) != len(rows):
        raise ValueError("need enough distinct conditioning inputs for a valid shuffle control")
    return rows


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_adapter_metadata(config, stage):
    if not config.get("adapter_path"):
        return None
    checkpoint = Path(config["adapter_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    expected = dict(schema=SCHEMA, K=8, body=[12, 20], writer_read_only=True,
                    lora_rank=int(config.get("lora_rank", 8)),
                    lora_alpha=int(config.get("lora_alpha", 16)),
                    reader_o_enabled=bool(config.get("reader_o_enabled", False)))
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("adapter metadata/architecture mismatch; legacy checkpoints cannot be silently imported")
    allowed = dict(writer=("reader", "writer"), grpo=("reader", "writer", "grpo"),
                   loop=("writer", "grpo", "loop"))
    if stage in allowed and meta.get("stage") not in allowed[stage]:
        raise ValueError(f"{stage} cannot initialize from stage {meta.get('stage')}")
    return meta


@dataclass
class ReplayItem:
    record: dict
    native: object
    student: object
    read: object
    sample: torch.Tensor
    timestep: float
    target: torch.Tensor


class GroundingRuntime:
    def __init__(self, config, stage):
        from .backbone import BagelBackbone
        from .inferencer import InterleaveInferencer
        from .modeling._bagel_utils import ImageTransform
        from safetensors.torch import load_file
        self.config, self.stage = config, stage
        self.adapter_metadata = validate_adapter_metadata(config, stage)
        self.device = accelerator.resolve_device(config.get("device", "auto"))
        if not accelerator.is_accelerator(self.device):
            raise RuntimeError("full BAGEL requires CUDA or Ascend NPU; CPU is for unit tests only")
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        else:
            torch.npu.set_device(self.device)
        accelerator.manual_seed_all(int(config.get("seed", 42)))
        backbone = BagelBackbone({**config, "disable_visual_gen": False, "disable_gen_expert": False,
                                  "loop_depth": 1 + config["num_write_rounds"]}).load()
        self.model, self.vae = backbone.bagel, backbone.vae_model
        include_o = bool(config.get("reader_o_enabled", False))
        backbone.apply_loop_trainable_policy(start_layer=12, end_layer=20,
            rank=int(config.get("lora_rank", 8)), alpha=int(config.get("lora_alpha", 16)),
            dropout=0.0, gen_attention_o_lora=include_o, k_v_lora=False)
        if config.get("adapter_path"):
            checkpoint = Path(config["adapter_path"])
            load_loop_adapter_state_dict(self.model, load_file(str(checkpoint)))
        for name, module in self.model.named_modules():
            if getattr(module, "is_loop_lora", False) and name.endswith(".q_proj"):
                # v2 writer means Read memory rows only, not UND updates in Write.
                module.write_enabled = False
        if stage == "eval":
            self.model.requires_grad_(False)
            self.trainable_names = []
        else:
            self.trainable_names = configure_loop_trainable_routes(self.model, config["trainable_routes"])
        self.model.to(self.device).eval()
        self.vae.to(self.device).eval().requires_grad_(False)
        ids = backbone.token_ids
        self.inferencer = InterleaveInferencer(model=self.model, vae_model=self.vae,
            tokenizer=backbone.tokenizer, vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14), new_token_ids={
                "bos_token_id": int(ids.im_start), "eos_token_id": int(ids.im_end),
                "start_of_image": int(ids.vision_start), "end_of_image": int(ids.vision_end)})
        self.shape = (config["height"], config["width"])

    def autocast(self):
        return accelerator.autocast_for(self.device)

    @torch.no_grad()
    def conditions(self, record, seed):
        from PIL import Image
        inf = self.inferencer
        with adapters_off(self.model), self.autocast():
            context = inf.init_gen_context()
            if self.config["objective"] == "target_flow":
                with Image.open(record["source_image"]) as im:
                    source = im.convert("RGB").resize(self.shape[::-1])
                context = inf.update_context_image(source, context, vae=True, vit=True)
            context = inf.update_context_text(record["prompt"], context)
            contexts = dict(full=context, text_removed=context, image_removed=context,
                            has_visual_condition=False)
            native = inf.prepare_velocity_bundle(name="native", contexts=contexts,
                                                  image_shape=self.shape, num_loop_tokens=0)
            template = native.flow_input["packed_init_noises"]
            noise = torch.randn(template.shape, generator=torch.Generator().manual_seed(seed)).to(template)
            student = inf.prepare_velocity_bundle(name="student", contexts=contexts,
                    image_shape=self.shape, init_noise=noise, num_loop_tokens=8)
            read = inf.prepare_memory_read_bundle(name="read", context=context,
                                                  image_shape=self.shape, num_loop_tokens=8)
        return ReplayItem(record, native, student, read, noise, 1.0, torch.empty(0)), noise

    def kwargs(self, item, *, native=False):
        kwargs = self.inferencer.build_image_velocity_kwargs(x_t=item.sample, timestep=item.timestep,
            condition=item.native if native else item.student, cfg_text_scale=1., cfg_img_scale=1.,
            cfg_interval=(0., 1.), cfg_renorm_min=0., cfg_renorm_type="global")
        if not native:
            for name in tuple(kwargs):
                if name.startswith("within_step_loop_"):
                    kwargs.pop(name)
            kwargs.update(packed_loop_token_indexes=item.student.flow_input["packed_loop_token_indexes"],
                          loop_memory=self.model.loop_memory[:8], memory_loop_start=12, memory_loop_end=20)
        return kwargs

    def native_velocity(self, item):
        with torch.no_grad(), adapters_off(self.model), self.autocast():
            return self.model._forward_flow(**self.kwargs(item, native=True)).detach()

    def read_memory(self, item, *, detach=False):
        with (torch.no_grad() if detach else nullcontext()), self.autocast():
            kwargs = self.inferencer.build_memory_read_kwargs(x_t=item.sample,
                timestep=item.timestep, condition=item.read, memory_loop_start=12, memory_loop_end=20)
            return self.model.forward_memory_read(**kwargs).memory_read

    def write(self, item, memory=None, *, mask=False, rounds=None, attention=None):
        with self.autocast():
            return self.model.forward_loop_supervised(**self.kwargs(item),
                num_write_rounds=rounds or self.config["num_write_rounds"],
                write_memory_override=memory, mask_prompt_kv_during_write=mask,
                attention_mass_sink=attention)

    @torch.no_grad()
    def states(self, record, seed, *, state_seed=None):
        """One native trajectory supplies several replays; never reroll per replay."""
        from .loop_pair_ground import prepare_flow_training_state, sample_weighted_timestep
        from .loop_distill import sample_replay_step_indices
        from PIL import Image
        item, noise = self.conditions(record, seed)
        # A batch shares selected timesteps; only prompt/noise/content differs.
        generator = torch.Generator().manual_seed((seed if state_seed is None else state_seed) + 17)
        if self.config["objective"] == "target_flow":
            with Image.open(record["target_image"]) as im:
                image = im.convert("RGB").resize(self.shape[::-1])
            with self.autocast():
                clean = self.inferencer.encode_image(image, self.shape)
            t = sample_weighted_timestep(generator, bucket_weights=tuple(self.config.get("timestep_bucket_weights", (.5,.3,.2))))
            item.sample, item.target = prepare_flow_training_state(clean, t, noise.to(clean))
            item.timestep = t
            return [item]
        ts, dts = self.model.prepare_image_schedule(self.config["num_steps"],
            float(self.config.get("timestep_shift", 3.0)), self.device)
        selected = sample_replay_step_indices(len(dts), self.config["states_per_prompt"],
            generator=generator, bucket_weights=tuple(self.config.get("timestep_bucket_weights", (.5,.3,.2))))
        result = []
        for i, (t, dt) in enumerate(zip(ts, dts)):
            item.timestep = float(t)
            v = self.native_velocity(item)
            if i in selected:
                result.append(ReplayItem(record, item.native, item.student, item.read,
                                          item.sample.detach().clone(), float(t), v.clone()))
            item.sample = self.model.image_euler_step(item.sample, v, dt)
        return result

    def dependency(self, items, *, masks=None, generator=None, detach_read=False, attention=None):
        memories = torch.stack([self.read_memory(item, detach=detach_read) for item in items])
        shuffled, donors = shuffle_across_batch(memories, generator=generator)
        predictions = {name: [] for name in ("correct", "shuffled", "zero")}
        for i, item in enumerate(items):
            for name, memory in (("correct", memories[i]), ("shuffled", shuffled[i]),
                                 ("zero", torch.zeros_like(memories[i]))):
                out = self.write(item, memory, mask=bool(masks[i]) if masks else False,
                                 attention=attention if name == "correct" else None)
                predictions[name].append(out.final_velocity)
        values = {name: torch.stack(v) for name, v in predictions.items()}
        c = self.config
        loss = memory_dependency_loss(correct_velocity=values["correct"],
            shuffled_velocity=values["shuffled"], zero_velocity=values["zero"],
            teacher_velocity=torch.stack([it.target for it in items]),
            lambda_teacher=float(c.get("lambda_teacher", c.get("lambda_effect", 1.))),
            lambda_dep_shuffle=float(c.get("lambda_dep_shuffle", .5)),
            lambda_dep_zero=float(c.get("lambda_dep_zero", .25)),
            lambda_direction=float(c.get("lambda_direction", .1)), margin=float(c.get("dependency_margin", .02)))
        metrics = {k: float(v.detach()) for k, v in asdict_tensors(loss).items()}
        metrics.update(donors=donors.tolist(), sample_ids=[str(it.record["id"]) for it in items],
                       timesteps=[it.timestep for it in items], state_hashes=[tensor_hash(it.sample) for it in items])
        for name in ("shuffled", "zero"):
            metrics[f"relative_dv_{name}"] = float((values["correct"] - values[name]).detach().float().norm()
                                                   / values[name].detach().float().norm().clamp_min(1e-8))
        return loss.loss, metrics

    def save(self, output, step, optimizer=None):
        from safetensors.torch import save_file
        stem = Path(output) / f"{self.stage}_step_{step:07d}"
        save_file(loop_adapter_state_dict(self.model), str(stem.with_suffix(".safetensors")))
        meta = dict(schema=SCHEMA, stage=self.stage, step=step, K=8, body=[12,20],
                    num_read_rounds=1, num_write_rounds=self.config["num_write_rounds"],
                    total_body_passes=1+self.config["num_write_rounds"], writer_read_only=True,
                    lora_rank=int(self.config.get("lora_rank",8)), lora_alpha=int(self.config.get("lora_alpha",16)),
                    reader_o_enabled=bool(self.config.get("reader_o_enabled",False)), config=self.config)
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        if optimizer is not None:
            torch.save(dict(step=step, optimizer=optimizer.state_dict(), config=self.config),
                       stem.with_suffix(".optimizer.pt"))


def asdict_tensors(value):
    # dataclasses.asdict deep-copies tensors and fails on non-leaf autograd values.
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
