"""GEN-only flow training on 64 devices. Native BAGEL stays frozen.

Each card holds the frozen generator and one or more caption-image pairs.
The step trains the state-free residual and the gate. Memory slots and the
UND reader are not in the optimizer. Sampling times stay inside the loop
schedule, so a closed gate still receives a gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path

import torch

from .config import LoopConfig
from .legacy import refuse_legacy_checkpoint
from .loop import AnchoredDualStateLoop
from .loss import anchored_loss
from .native_path import BranchContext, pack_flow_hidden
from ..diagnostics import distributed as parallel
from ..diagnostics.data import file_hash
from ..diagnostics.native import device_inputs, empty_cache, text_source
from ..diagnostics.numerics import autocast_for
from ..diagnostics.record import FrozenGuard, write_json
from ..diagnostics.schedule import native_times
from ..interaction.data import PairSampler, load_pairs
from .schedule import in_loop_schedule


DEVICES = 64


@dataclass
class GenOnlyTrainConfig:
    """Launch defaults for the 64-card GEN-only run. They are not a tuned recipe."""

    steps: int = 500
    global_batch: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    warmup_steps: int = 20
    save_steps: tuple[int, ...] = (100, 250, 500)
    seed: int = 17
    resolution: int = 512
    sampler_steps: int = 50
    timestep_shift: float = 3.0
    posterior_seed: int = 0

    def __post_init__(self):
        if isinstance(self.save_steps, list):
            self.save_steps = tuple(int(step) for step in self.save_steps)

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.global_batch < DEVICES or self.global_batch % DEVICES:
            raise ValueError("global_batch must be a positive multiple of 64")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning rate, weight decay, and gradient clip are out of range")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if any(step < 1 or step > self.steps for step in self.save_steps):
            raise ValueError("save_steps must lie inside the training budget")
        if self.resolution < 1 or self.sampler_steps < 2 or self.timestep_shift <= 0.0:
            raise ValueError("resolution and the native sampler grid are out of range")

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["save_steps"] = list(self.save_steps)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "GenOnlyTrainConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown GEN-only train fields: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "GenOnlyTrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def require_gen_only_loop(config: LoopConfig) -> None:
    if config.variant != "gen_only":
        raise ValueError("this entry trains variant=gen_only")
    if not config.enabled:
        raise ValueError("gen_only training requires the loop to be enabled")
    if config.force_native_bypass or config.force_full_zero_path:
        raise ValueError("gen_only training runs the correction")
    if config.feedback_mode != "correct":
        raise ValueError("gen_only has no memory state to intervene on")


def require_sixty_four(world: int, global_batch: int) -> None:
    if world != DEVICES:
        raise ValueError(f"GEN-only training expects {DEVICES} devices, received {world}")
    if global_batch % world:
        raise ValueError("global_batch must be divisible by the device count")


def select_gen_only_parameters(loop: AnchoredDualStateLoop) -> list[torch.nn.Parameter]:
    """Train the state-free residual and the gate. Freeze memory parameters."""
    selected = loop.trainable_parameters()
    live = {id(parameter) for parameter in selected}
    for parameter in loop.parameters():
        parameter.requires_grad_(id(parameter) in live)
    if not selected:
        raise RuntimeError("gen_only exposed no trainable parameters")
    return selected


def scheduled_flow_times(config: LoopConfig, sampler_steps: int, timestep_shift: float) -> list[float]:
    """Native sampler nodes that enter the correction. Late nodes stay out."""
    times = native_times({"sampler_steps": sampler_steps, "timestep_shift": timestep_shift})
    selected = [float(value) for value in times if in_loop_schedule(config, flow_time=float(value))]
    if not selected:
        raise ValueError("the loop schedule contains no native sampler node")
    return selected


def correction_loss(loop: AnchoredDualStateLoop, output, target: torch.Tensor, vae_indexes: torch.Tensor):
    return anchored_loss(
        output.velocity, target, output.native_velocity, output.delta, output.anchor, vae_indexes,
        lambda_anchor=loop.config.lambda_anchor, lambda_entry=loop.config.lambda_entry,
        round_velocities=output.round_velocities,
        round_loss_weights=loop.config.round_loss_weights,
    )


def posterior_seed_for(path: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{file_hash(path)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@torch.no_grad()
def encode_target(bundle, path: str, resolution: int, posterior_seed: int) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    from ..bagel.accelerator import seeded_context

    device = bundle.model.language_model.model.embed_tokens.weight.device
    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
    with seeded_context(device, posterior_seed), autocast_for(device):
        latent = bundle.vae.encode(pixels.to(next(bundle.vae.parameters()).dtype))
    patch = bundle.model.latent_patch_size
    _, channels, height, width = latent.shape
    return torch.einsum(
        "chpwq->hwpqc", latent[0].reshape(channels, height // patch, patch, width // patch, patch),
    ).reshape(-1, patch * patch * channels)


@torch.no_grad()
def conditional_flow(bundle, prompt: str, resolution: int) -> dict:
    """One conditional GEN packing. Text CFG is not part of this loss."""
    model = bundle.model
    decoder = model.language_model.model
    device = decoder.embed_tokens.weight.device
    with autocast_for(device):
        source, _, positions = text_source(bundle, [prompt])
    cache = empty_cache(decoder)
    for index, item in enumerate(source):
        cache.key_cache[index], cache.value_cache[index] = item.keys, item.values
    flow = model.prepare_vae_latent(
        source[0].lengths, positions, [(resolution, resolution)], bundle.token_ids,
    )
    flow.pop("packed_init_noises", None)
    flow = device_inputs(flow, device)
    flow["past_key_values"] = cache
    return flow


def conditional_branch(flow: dict, flow_time: float) -> BranchContext:
    return BranchContext(
        name="cond",
        query_lens=flow["packed_seqlens"],
        packed_query_position_ids=flow["packed_position_ids"],
        packed_query_indexes=flow["packed_indexes"],
        past_key_values=flow["past_key_values"],
        key_values_lens=flow["key_values_lens"],
        packed_key_value_indexes=flow["packed_key_value_indexes"],
        packed_text_indexes=flow["packed_text_indexes"],
        packed_vae_token_indexes=flow["packed_vae_token_indexes"],
        flow_time=flow_time,
    )


def paired_latents(x0: torch.Tensor, flow_time: float, noise_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    epsilon = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(device=x0.device, dtype=x0.dtype)
    latent = epsilon if flow_time == 1.0 else (1.0 - flow_time) * x0 + flow_time * epsilon
    return latent, epsilon.float() - x0.float()


def forward_correction(loop, model, latent, flow_time: float, flow: dict):
    branch = conditional_branch(flow, flow_time)
    hidden = pack_flow_hidden(
        model, latent, torch.full((latent.shape[0],), flow_time, device=latent.device),
        flow["packed_text_ids"], flow["packed_text_indexes"], flow["packed_vae_token_indexes"],
        flow["packed_vae_position_ids"], flow["packed_seqlens"],
    )
    device = latent.device if isinstance(latent.device, torch.device) else torch.device(latent.device)
    with autocast_for(device):
        return loop.forward_branch(
            hidden, branch, flow_time=flow_time, velocity_head=model.llm2vae, train=True,
        )


def _checkpoint_path(output: Path, step: int) -> Path:
    return output / f"step_{step:06d}.pt"


def save_checkpoint(output: Path, loop, optimizer, sampler, train: GenOnlyTrainConfig, step: int) -> None:
    blob = loop.export_state()
    blob.update(
        variant="gen_only", step=step, train=train.to_dict(),
        optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
    )
    torch.save(blob, _checkpoint_path(output, step))


def place_optimizer_state(optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=parameter.device)


def load_checkpoint(path: str | Path, loop, optimizer, sampler) -> int:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    refuse_legacy_checkpoint(blob)
    if blob.get("variant") != "gen_only":
        raise ValueError("resume requires a gen_only checkpoint")
    loop.load_state(blob)
    optimizer.load_state_dict(blob["optimizer"])
    place_optimizer_state(optimizer)
    sampler.load_state_dict(blob["sampler"])
    return int(blob["step"])


def train_gen_only(loop, bundle, rows, train: GenOnlyTrainConfig, output: Path, *, resume: str | None = None) -> None:
    from ..bagel.accelerator import max_memory_allocated, reset_peak_memory_stats, synchronize
    from ..diagnostics.training_logging import configure_logging

    require_gen_only_loop(loop.config)
    train.validate()
    require_sixty_four(parallel.world_size(), train.global_batch)
    output = Path(output)
    parameters = select_gen_only_parameters(loop)
    loop.train()
    if resume:
        sampler = PairSampler(rows, train.seed)
        optimizer = torch.optim.AdamW(parameters, lr=train.learning_rate, weight_decay=train.weight_decay)
        start = load_checkpoint(resume, loop, optimizer, sampler)
        select_gen_only_parameters(loop)
    else:
        parallel.broadcast_parameters(loop)
        sampler = PairSampler(rows, train.seed)
        optimizer = torch.optim.AdamW(parameters, lr=train.learning_rate, weight_decay=train.weight_decay)
        start = 0
    bundle.model.eval()
    bundle.vae.eval()
    guard = FrozenGuard(bundle.model, bundle.vae)
    times = scheduled_flow_times(loop.config, train.sampler_steps, train.timestep_shift)
    logger = configure_logging(output)
    reset_peak_memory_stats(bundle.model.language_model.model.embed_tokens.weight.device)
    started_at = time.perf_counter()
    device = bundle.model.language_model.model.embed_tokens.weight.device
    if parallel.rank() == 0:
        write_json(output / "run.json", {
            "experiment": "anchored_dual_state_loop",
            "variant": "gen_only",
            "devices": parallel.world_size(),
            "loop": loop.config.to_dict(),
            "train": train.to_dict(),
            "scheduled_flow_times": times,
            "train_pairs": len(rows),
            "trainable": sum(parameter.numel() for parameter in parameters),
        })
    parallel.barrier()
    for step in range(start + 1, train.steps + 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = sampler.take(train.global_batch)
        draws = [
            (sampler.rng.choice(times), sampler.rng.randrange(2**63 - 1))
            for _ in batch
        ]
        totals = [0.0, 0.0, 0.0]
        for index in range(parallel.rank(), train.global_batch, parallel.world_size()):
            row = batch[index]
            flow_time, noise_seed = draws[index]
            x0 = encode_target(
                bundle, row["image"], train.resolution,
                posterior_seed_for(row["image"], train.posterior_seed),
            )
            latent, target = paired_latents(x0, flow_time, noise_seed)
            flow = conditional_flow(bundle, row["caption"], train.resolution)
            output = forward_correction(loop, bundle.model, latent, flow_time, flow)
            loss, parts = correction_loss(
                loop, output, target, flow["packed_vae_token_indexes"],
            )
            (loss / train.global_batch).backward()
            totals[0] += float(parts["total"])
            totals[1] += float(parts["final"])
            totals[2] += float(parts.get("error_fell_1", 0.0))
            del output, loss, latent, target, flow, x0
        parallel.sum_gradients(loop)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, train.gradient_clip, error_if_nonfinite=True,
        )
        learning_rate = train.learning_rate * min(1.0, step / max(1, train.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        guard.check()
        loop.assert_native_frozen()
        reduced = parallel.sum_statistics(totals, device)
        synchronize(device)
        record = {
            "step": step,
            "loss": reduced[0] / train.global_batch,
            "flow": reduced[1] / train.global_batch,
            "rounds_error_fell": reduced[2] / train.global_batch,
            "gate": float(loop.gate.raw.detach()),
            "gradient_norm": float(grad_norm),
            "learning_rate": learning_rate,
            "step_seconds": time.perf_counter() - step_started,
            "elapsed_seconds": time.perf_counter() - started_at,
            "peak_memory_gib": parallel.max_statistics([max_memory_allocated(device) / 1024**3], device)[0],
        }
        if parallel.rank() == 0:
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            logger.info("step=%d %s", step, json.dumps(record))
        if step in train.save_steps or step == train.steps:
            if parallel.rank() == 0:
                save_checkpoint(output, loop, optimizer, sampler, train, step)
            parallel.barrier()
