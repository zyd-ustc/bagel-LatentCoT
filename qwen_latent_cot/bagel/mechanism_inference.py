"""Frozen six-arm velocity engine and same-state counterfactual probes."""
from __future__ import annotations

import torch

from .memory_mechanism import MODES, control_for
from .modeling.bagel.qwen2_navit import NaiveCache


class MemoryMechanismEngine:
    """One packed pair, read-only prompt cache, fresh M at every velocity call."""

    def __init__(self, inferencer, prompts, noises, image_shape):
        if len(prompts) != 2 or len(set(prompts)) != 2 or len(noises) != 2:
            raise ValueError("mechanism evaluation requires a pair of distinct prompts")
        self.inferencer = inferencer
        self.model = model = inferencer.model
        if model.training or not model.use_moe:
            raise ValueError("mechanism evaluation requires frozen MoT eval mode")
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("freeze model parameters before mechanism evaluation")
        if any(hasattr(m, "lora_A") for m in model.modules()):
            raise ValueError("LoRA modules are forbidden in the six-arm protocol")
        if model.loop_memory is None or model.loop_memory.shape[0] != 8:
            raise ValueError("six-arm protocol requires exactly K=8 allocated memory slots")
        if len(model.language_model.model.layers) < 20:
            raise ValueError("six-arm protocol requires body [12,20)")
        model.language_model.model.enable_taylorseer = False
        self.device = inferencer.device
        self.noise = torch.cat(noises).to(self.device)
        self.prompts = tuple(prompts)
        self.image_shape = tuple(image_shape)
        self.cache = NaiveCache(model.config.llm_config.num_hidden_layers)
        prompt_input, lens, ropes = model.prepare_prompts(
            curr_kvlens=[0, 0], curr_rope=[0, 0], prompts=prompts,
            tokenizer=inferencer.tokenizer, new_token_ids=inferencer.new_token_ids,
        )
        self.cache = model.forward_cache_update_text(self.cache, **self.move(prompt_input))
        self.empty_cache = NaiveCache(model.config.llm_config.num_hidden_layers)
        self.layouts = {}
        self.cfg_layouts = {}
        for k in (0, 8):
            packed = model.prepare_vae_latent(
                curr_kvlens=lens, curr_rope=ropes, image_sizes=[image_shape] * 2,
                new_token_ids=inferencer.new_token_ids, num_loop_tokens=k,
            )
            if packed["packed_init_noises"].shape != self.noise.shape:
                raise ValueError("fixed noise disagrees with native latent geometry")
            self.layouts[k] = self.move(packed)
            self.cfg_layouts[k] = self.move(model.prepare_vae_latent_cfg(
                curr_kvlens=[0, 0], curr_rope=[0, 0], image_sizes=[image_shape] * 2,
                num_loop_tokens=k,
            ))
        self.lengths = [int(x) for x in self.layouts[0]["packed_vae_seqlens"].tolist()]

    def move(self, values):
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in values.items()}

    @torch.inference_mode()
    def velocity(self, x_t, timestep, memory_control_mode, *, diagnostics=False, rounds=2):
        control = control_for(memory_control_mode)
        model = self.model
        k = 8 if control.present else 0
        packed = self.layouts[k]
        cfg = self.cfg_layouts[k]
        text_idx = packed["packed_text_indexes"]
        gen_idx = packed["packed_vae_token_indexes"]
        mem_idx = packed["packed_loop_token_indexes"]
        text = model.language_model.model.embed_tokens(packed["packed_text_ids"])
        hidden = text.new_zeros((int(packed["packed_seqlens"].sum()), model.hidden_size))
        hidden[text_idx] = text
        if k:
            hidden[mem_idx] = model.loop_memory.to(hidden).repeat(2, 1)
        t = torch.full((x_t.shape[0],), float(timestep), device=x_t.device)
        vae = (model.vae2llm(x_t) + model.time_embedder(t)
               + model.latent_pos_embed(packed["packed_vae_position_ids"]))
        hidden[gen_idx] = vae.to(hidden)
        routed_text = model.mot_und_route_indexes(text_idx, mem_idx)

        def branch(unconditional, sink):
            output = model.language_model.forward_inference(
                packed_query_sequence=hidden.clone(),
                query_lens=packed["packed_seqlens"],
                packed_query_position_ids=(cfg["cfg_packed_position_ids"] if unconditional else packed["packed_position_ids"]),
                packed_query_indexes=(cfg["cfg_packed_query_indexes"] if unconditional else packed["packed_indexes"]),
                past_key_values=self.empty_cache if unconditional else self.cache,
                key_values_lens=cfg["cfg_key_values_lens"] if unconditional else packed["key_values_lens"],
                packed_key_value_indexes=(cfg["cfg_packed_key_value_indexes"] if unconditional else packed["packed_key_value_indexes"]),
                update_past_key_values=False, is_causal=False, mode="gen",
                packed_vae_token_indexes=gen_idx, packed_text_indexes=routed_text,
                packed_memory_token_indexes=mem_idx,
                memory_loop_start=12, memory_loop_end=20,
                memory_control_mode=memory_control_mode, mechanism_diagnostics=sink,
                memory_control_rounds=rounds,
            )
            return model.llm2vae(output.packed_query_sequence)[gen_idx]

        sink = {} if diagnostics else None
        conditional = branch(False, sink)
        # Native BAGEL guidance schedule; independent per-sample global norm.
        scale = 4.0 if 0.4 < float(timestep) <= 1.0 else 1.0
        unconditional = branch(True, None) if scale > 1 else None
        velocity = model._combine_cfg_velocities(
            conditional, unconditional, None, cfg_text_scale=scale,
            cfg_img_scale=1.0, cfg_renorm_min=0.0, cfg_renorm_type="sample_global",
            vae_seqlens=packed["packed_vae_seqlens"],
        )
        if not torch.isfinite(velocity).all():
            raise FloatingPointError(f"nonfinite velocity for {memory_control_mode}")
        return velocity, sink


PAIRS = {"topology": ("static_null", "native"),
         "dynamic": ("zero_dynamic", "static_null"),
         "content": ("normal", "zero_dynamic"),
         "specificity": ("normal", "shuffled_dynamic"),
         "joint": ("normal", "frozen_correct")}


def mechanism_metrics(velocities, diagnostics, lengths, step, timestep):
    """All pairwise relative magnitudes use the SAME native velocity norm."""
    results = []
    chunks = {mode: tensor.detach().float().cpu().split(lengths)
              for mode, tensor in velocities.items()}
    for sample, native in enumerate(chunks["native"]):
        base_norm = float(native.norm())
        row = {"sample": sample, "step": int(step), "t": float(timestep),
               "v_native_norm": base_norm, "hidden_branch": "conditional"}
        denom = base_norm + 1e-12
        for name, (left, right) in PAIRS.items():
            norm = float((chunks[left][sample] - chunks[right][sample]).norm())
            row[f"delta_{name}_norm"] = norm
            row[f"relative_{name}"] = norm / denom
        delta_d = (chunks["normal"][sample] - native).flatten()
        delta_c = (chunks["zero_dynamic"][sample] - native).flatten()
        norm_product = float(delta_d.norm() * delta_c.norm())
        row["cos_D_vs_C"] = (max(-1.0, min(1.0, float(torch.dot(delta_d, delta_c)) / norm_product))
                                  if norm_product > 1e-24 else None)
        row["arms"] = {}
        for mode in MODES:
            arm = {"relative_velocity_vs_native": float((chunks[mode][sample] - native).norm()) / denom}
            for field in ("gen_body_hidden", "gen_suffix_hidden"):
                ref = diagnostics["native"][field].split(lengths)[sample].float()
                value = diagnostics[mode][field].split(lengths)[sample].float()
                arm[f"relative_{field}"] = float((value - ref).norm() / (ref.norm() + 1e-12))
            for field in ("memory_read", "memory_write_input"):
                if field in diagnostics[mode]:
                    memory = diagnostics[mode][field].reshape(len(lengths), -1)
                    arm[f"{field}_norm"] = float(memory[sample].float().norm())
            row["arms"][mode] = arm
        results.append(row)
    return results


@torch.inference_mode()
def probe_base_trajectory(engine, timesteps, dts, on_step):
    """Only native advances x_t; other arms are counterfactual evaluations."""
    x_t = engine.noise.clone()
    for step, (t, dt) in enumerate(zip(timesteps, dts)):
        velocities, diagnostics = {}, {}
        native_device_velocity = None
        for mode in MODES:
            velocity, diag = engine.velocity(x_t.clone(), t, mode, diagnostics=True)
            if mode == "native":
                native_device_velocity = velocity
            velocities[mode] = velocity.detach().float().cpu()
            diagnostics[mode] = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                                 for key, value in diag.items()}
        rows = mechanism_metrics(velocities, diagnostics, engine.lengths, step, t)
        on_step(step, x_t.detach().cpu().clone(), float(t), rows, diagnostics)
        x_t = engine.model.image_euler_step(x_t, native_device_velocity, dt)
    return x_t


@torch.inference_mode()
def generate_trajectory(engine, timesteps, dts, mode):
    x_t = engine.noise.clone()
    for t, dt in zip(timesteps, dts):
        velocity, _ = engine.velocity(x_t, t, mode)
        x_t = engine.model.image_euler_step(x_t, velocity, dt)
    return x_t
