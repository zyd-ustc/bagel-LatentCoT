"""Frozen BAGEL T0 teacher and on-policy memory-reader student runtime."""

from dataclasses import dataclass

import torch

from . import accelerator
from .memory_reader import install_memory_readers
from .cot_teacher import teacher_condition


@dataclass
class OPDCondition:
    record: dict
    native: object
    teacher: object
    read: object
    prompt_context: dict
    noise: torch.Tensor


@dataclass
class OPDState:
    condition: OPDCondition
    sample: torch.Tensor
    timestep: float
    step_index: int


class OPDRuntime:
    def __init__(self, config, model, vae, inferencer, device, trainable_names):
        self.config = config
        self.model, self.vae, self.inferencer = model, vae, inferencer
        self.device = torch.device(device)
        self.trainable_names = trainable_names
        self.shape = (int(config["height"]), int(config["width"]))
        self.body_start = int(config["memory_loop_start_layer"])
        self.body_end = int(config["memory_loop_end_layer"])
        self.slots = int(config["num_loop_tokens"])

    @classmethod
    def load_model(cls, config):
        from .backbone import BagelBackbone
        from .inferencer import InterleaveInferencer
        from .modeling._bagel_utils import ImageTransform
        device = accelerator.resolve_device(config.get("device", "auto"))
        if not accelerator.is_accelerator(device):
            raise RuntimeError("BAGEL OPD requires CUDA or NPU; CPU is for tiny unit tests")
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
        else:
            torch.npu.set_device(device)
        seed = int(config.get("seed", 42))
        accelerator.manual_seed_all(seed)
        loaded = BagelBackbone({**config, "disable_visual_gen": False,
                                "disable_gen_expert": False, "loop_depth": 2}).load()
        model, vae = loaded.bagel, loaded.vae_model
        names = install_memory_readers(model,
            start=int(config["memory_loop_start_layer"]),
            end=int(config["memory_loop_end_layer"]),
            rank=int(config.get("o_adapter_rank", 8)),
            alpha=int(config.get("o_adapter_alpha", 16)))
        # OPD uses inference-mode MoT routing while still backpropagating
        # through frozen downstream layers to the new reader branch.
        model.language_model.model.gradient_checkpointing = True
        model.to(device).eval()
        vae.to(device).eval().requires_grad_(False)
        ids = loaded.token_ids
        inferencer = InterleaveInferencer(model=model, vae_model=vae,
            tokenizer=loaded.tokenizer, vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14), new_token_ids={
                "bos_token_id": int(ids.im_start), "eos_token_id": int(ids.im_end),
                "start_of_image": int(ids.vision_start), "end_of_image": int(ids.vision_end)})
        return cls(config, model, vae, inferencer, device, names)

    def autocast(self):
        return accelerator.autocast_for(self.device)

    @torch.no_grad()
    def prepare(self, record, seed):
        inf = self.inferencer
        with self.autocast():
            prompt_ctx = inf.init_gen_context()
            prompt_ctx = inf.update_context_text(record["prompt"], prompt_ctx,
                capture_prompt_hidden_at=self.body_start)
            teacher_ctx = inf.init_gen_context()
            teacher_ctx = inf.update_context_text(
                teacher_condition(record["prompt"], record["reasoning_text"]), teacher_ctx)
            def bundle(name, ctx, slots):
                contexts = dict(full=ctx, text_removed=ctx, image_removed=ctx,
                                has_visual_condition=False)
                return inf.prepare_velocity_bundle(name=name, contexts=contexts,
                                                   image_shape=self.shape, num_loop_tokens=slots)
            native = bundle("native", prompt_ctx, 0)
            teacher = bundle("teacher", teacher_ctx, 0)
            read = inf.prepare_memory_read_bundle(name="read", context=prompt_ctx,
                                                  image_shape=self.shape, num_loop_tokens=self.slots)
        template = native.flow_input["packed_init_noises"]
        noise = torch.randn(tuple(template.shape), generator=torch.Generator().manual_seed(seed),
                            dtype=torch.float32).to(template)
        return OPDCondition(record, native, teacher, read, prompt_ctx, noise)

    def _flow_kwargs(self, state, *, teacher=False):
        return self.inferencer.build_image_velocity_kwargs(
            x_t=state.sample, timestep=state.timestep,
            condition=state.condition.teacher if teacher else state.condition.native,
            cfg_text_scale=1., cfg_img_scale=1., cfg_interval=(0., 1.),
            cfg_renorm_min=0., cfg_renorm_type="global")

    def _read_kwargs(self, state):
        return self.inferencer.build_memory_read_kwargs(
            x_t=state.sample, timestep=state.timestep,
            condition=state.condition.read, memory_loop_start=self.body_start,
            memory_loop_end=self.body_end)

    @torch.no_grad()
    def teacher_velocity(self, state):
        with self.autocast():
            return self.model._forward_flow(**self._flow_kwargs(state, teacher=True)).detach()

    @torch.no_grad()
    def native_velocity(self, state):
        with self.autocast():
            return self.model._forward_flow(**self._flow_kwargs(state)).detach()

    def student_velocity(self, state, *, return_memory=False, memory_override=None):
        with self.autocast():
            if memory_override is not None:
                if not isinstance(memory_override, (tuple, list)):
                    raise ValueError("OPD override must contain one memory tensor per reader layer")
                return self.model._forward_flow(**self._flow_kwargs(state),
                    opd_memory_hidden=tuple(memory.detach() for memory in memory_override),
                    opd_reader_start=self.body_start, opd_reader_end=self.body_end)
            ctx = state.condition.prompt_context
            return self.model.forward_memory_opd_velocity(
                x_t=state.sample, timestep=state.timestep,
                condition=dict(prompt_hidden=ctx["prompt_hidden"],
                               prompt_mask=ctx["prompt_mask"],
                               content_mask=ctx["content_mask"], num_slots=self.slots,
                               read_kwargs=self._read_kwargs(state),
                               flow_kwargs=self._flow_kwargs(state)),
                memory_body_start=self.body_start,
                memory_body_end=self.body_end, return_memory=return_memory)

    @torch.no_grad()
    def rollout(self, record, seed):
        from .loop_distill import sample_replay_step_indices
        condition = self.prepare(record, seed)
        ts, dts = self.model.prepare_image_schedule(int(self.config["num_steps"]),
            float(self.config["timestep_shift"]), self.device)
        indexes = set(sample_replay_step_indices(len(dts),
            int(self.config["states_per_rollout"]),
            generator=torch.Generator().manual_seed(seed + 17),
            bucket_weights=tuple(self.config["timestep_bucket_weights"])))
        x_t, states = condition.noise.detach(), []
        for index, (t, dt) in enumerate(zip(ts, dts)):
            state = OPDState(condition, x_t.detach(), float(t), index)
            if index in indexes:
                states.append(OPDState(condition, x_t.detach().clone(), float(t), index))
            velocity = self.student_velocity(state)
            x_t = self.model.image_euler_step(x_t, velocity, dt).detach()
        if len(states) != int(self.config["states_per_rollout"]):
            raise RuntimeError("on-policy rollout did not capture requested states")
        return states

    def reader_diagnostics(self):
        layers = self.model.language_model.model.layers[self.body_start:self.body_end]
        outputs = [layer.memory_reader.last_output_rms for layer in layers]
        native = [layer.memory_reader.last_native_attn_rms for layer in layers]
        entropies = [layer.memory_reader.last_attention_entropy for layer in layers]
        peaks = [layer.memory_reader.last_attention_max for layer in layers]
        return dict(memory_reader_output_rms=sum(outputs)/len(outputs),
                    memory_reader_native_attn_ratio=sum(
                        o/max(n, 1e-8) for o,n in zip(outputs,native))/len(outputs),
                    gen_to_memory_attention_entropy=sum(entropies)/len(entropies),
                    gen_to_memory_attention_max=sum(peaks)/len(peaks))
