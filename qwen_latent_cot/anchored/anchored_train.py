"""Two-stage anchored curriculum on 64 devices. Native BAGEL stays frozen.

Stage 1 trains the state-free residual and the gate. Stage 2 loads those
weights, freezes that residual, the gate, and the C2-observe reader, and
trains the feedback updater plus ``q_proj``, ``k_proj``, ``v_proj``, and
``mem_out``. ``q_norm``, ``k_norm``, and ``norm_m`` stay at initialization.

The validation probe is one fixed draw from the generation val split. It
records whether the first correction's mean square is below the native mean
square, and, in stage 2, whether that dual error is below the frozen GEN-only
path and whether shuffled and zero are higher. Those flags are logs. They do
not stop the run and they do not show that a generated image improved.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
import time
from pathlib import Path

import torch

from .gen_only_train import (
    DEVICES,
    conditional_flow,
    encode_target,
    forward_correction,
    paired_latents,
    place_optimizer_state,
    posterior_seed_for,
    require_gen_only_loop,
    require_sixty_four,
    sample_timesteps,
    schedule_time_window,
    scheduled_flow_times,
)
from .legacy import refuse_legacy_checkpoint
from .loop import AnchoredDualStateLoop
from .loss import anchored_loss
from ..diagnostics import distributed as parallel
from ..diagnostics.record import FrozenGuard, write_json


GEN_ONLY_PREFIXES = (
    "gate.",
    "adapter.norm_h.",
    "adapter.norm_p.",
    "adapter.h_down.",
    "adapter.p_down.",
    "adapter.time_down.",
    "adapter.up.",
)
DUAL_EXTRA_PREFIXES = (
    "updater.",
    "adapter.q_proj.",
    "adapter.k_proj.",
    "adapter.v_proj.",
    "adapter.mem_out.",
)
TRAINER = "anchored_curriculum"


@dataclass
class CurriculumStage:
    """One block of the 400 + 600 schedule. Counts are launch settings."""

    name: str
    steps: int
    learning_rate: float
    output_learning_rate: float
    warmup_steps: int = 20
    correction_rounds: int = 1
    second_round_at: int | None = None
    save_steps: tuple[int, ...] = ()

    def __post_init__(self):
        if isinstance(self.save_steps, list):
            self.save_steps = tuple(int(step) for step in self.save_steps)
        if self.second_round_at is not None:
            self.second_round_at = int(self.second_round_at)

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["save_steps"] = list(self.save_steps)
        return payload


@dataclass
class AnchoredTrainConfig:
    """Shared batch and the two stages. This is not a measured recipe."""

    global_batch: int = 64
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    seed: int = 17
    resolution: int = 512
    sampler_steps: int = 50
    timestep_shift: float = 3.0
    posterior_seed: int = 0
    lambda_improve: float = 1.0
    improvement_margin: float = 1e-4
    val_images: int = 128
    stages: tuple[CurriculumStage, ...] = ()

    def __post_init__(self):
        if isinstance(self.stages, list):
            self.stages = tuple(
                stage if isinstance(stage, CurriculumStage) else CurriculumStage(**stage)
                for stage in self.stages
            )

    def validate(self) -> None:
        if self.global_batch < DEVICES or self.global_batch % DEVICES:
            raise ValueError("global_batch must be a positive multiple of 64")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("weight decay or gradient clip is out of range")
        if self.lambda_improve < 0.0 or self.improvement_margin < 0.0:
            raise ValueError("improvement loss weights must be non-negative")
        if self.resolution < 1 or self.sampler_steps < 2 or self.timestep_shift <= 0.0:
            raise ValueError("resolution and the native sampler grid are out of range")
        if self.val_images < DEVICES * 2 or self.val_images % DEVICES:
            raise ValueError("val_images must be a multiple of 64 and at least two images per device")
        if len(self.stages) != 2 or [stage.name for stage in self.stages] != ["gen_only", "dual"]:
            raise ValueError("the curriculum is a gen_only stage followed by a dual stage")
        for stage in self.stages:
            if stage.steps < 1:
                raise ValueError("stage steps must be positive")
            if stage.learning_rate <= 0.0 or stage.output_learning_rate <= 0.0:
                raise ValueError("stage learning rates must be positive")
            if stage.warmup_steps < 0:
                raise ValueError("warmup_steps must be non-negative")
            if stage.correction_rounds != 1:
                raise ValueError("each stage starts at one correction round")
            if any(step < 1 or step > stage.steps for step in stage.save_steps):
                raise ValueError("save_steps must lie inside the stage budget")
        first, second = self.stages
        if first.second_round_at is not None:
            raise ValueError("the gen_only stage stays at one correction round")
        if second.second_round_at is None or not 1 <= second.second_round_at <= second.steps:
            raise ValueError("second_round_at must fall inside the dual stage")

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["stages"] = [stage.to_dict() for stage in self.stages]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "AnchoredTrainConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown curriculum fields: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "AnchoredTrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def correction_rounds_for(stage: CurriculumStage, step: int) -> int:
    """Rounds used on this stage step. ``second_round_at`` is the first step at 2."""
    if stage.second_round_at is not None and step >= stage.second_round_at:
        return 2
    return stage.correction_rounds


def stage_prefixes(stage: CurriculumStage) -> tuple[str, ...]:
    if stage.name == "gen_only":
        return GEN_ONLY_PREFIXES
    if stage.name == "dual":
        return DUAL_EXTRA_PREFIXES
    raise ValueError(f"unknown curriculum stage {stage.name}")


def output_prefixes(stage: CurriculumStage) -> tuple[str, ...]:
    """Low learning rate on the stage's output projection."""
    if stage.name == "gen_only":
        return ("adapter.up.",)
    return ("adapter.mem_out.",)


def apply_stage_parameters(loop: AnchoredDualStateLoop, stage: CurriculumStage) -> list[dict]:
    """Turn on this stage's parameters and return AdamW groups.

    Stage 2 leaves the stage-1 residual and the gate frozen. Attention norms
    and ``adapter.norm_m`` stay frozen as well. The returned groups carry
    ``base_lr`` so a later warmup can scale them separately.
    """
    prefixes = stage_prefixes(stage)
    outputs = output_prefixes(stage)
    base, output = [], []
    for name, parameter in loop.named_parameters():
        selected = name.startswith(prefixes)
        parameter.requires_grad_(selected)
        if not selected:
            continue
        if name.startswith(outputs):
            output.append(parameter)
        else:
            base.append(parameter)
    if not base or not output:
        raise RuntimeError("curriculum stage exposed an empty parameter group")
    loop.config.variant = stage.name
    loop.config.correction_rounds = stage.correction_rounds
    loop.config.validate(len(loop.decoder.layers))
    return [
        {"params": base, "lr": stage.learning_rate, "base_lr": stage.learning_rate, "name": "base"},
        {
            "params": output,
            "lr": stage.output_learning_rate,
            "base_lr": stage.output_learning_rate,
            "name": "output",
        },
    ]


def require_stage2_reader(checkpoint: str, source_mode: str) -> Path:
    """Check the reader path before either stage spends a step. Do not load it."""
    if source_mode != "gen_full_query":
        raise ValueError("stage 2 requires observation_source_mode=gen_full_query")
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError("stage 2 observation checkpoint is missing")
    return path


def prepare_dual_reader(loop: AnchoredDualStateLoop, checkpoint: str) -> None:
    """Load the frozen C2-observe copy. Native BAGEL is not written."""
    path = require_stage2_reader(checkpoint, loop.config.observation_source_mode)
    loop.config.variant = "dual"
    loop.config.observation_checkpoint = str(path)
    loop.config.allow_unloaded_observer = False
    loop.config.validate(len(loop.decoder.layers))
    loop.observer.load_checkpoint(str(path), source_mode="gen_full_query")
    loop.observer.freeze()
    if not loop.observer.loaded or loop.observer.source_mode != "gen_full_query":
        raise RuntimeError("stage 2 reader did not stay on gen_full_query")
    live = [name for name, parameter in loop.observer.named_parameters() if parameter.requires_grad]
    if live:
        raise RuntimeError("observation reader is trainable: " + live[0])


def _checkpoint_path(output: Path, stage: str, step: int) -> Path:
    return output / f"{stage}_step_{step:06d}.pt"


def save_checkpoint(output: Path, loop, optimizer, sampler, train: AnchoredTrainConfig, stage: str, step: int) -> None:
    blob = loop.export_state()
    blob.update(
        trainer=TRAINER, variant=loop.config.variant, stage=stage, step=step,
        train=train.to_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
    )
    torch.save(blob, _checkpoint_path(output, stage, step))


def load_checkpoint(path: str | Path, loop, sampler) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    refuse_legacy_checkpoint(blob)
    if blob.get("trainer") != TRAINER:
        raise ValueError("resume requires an anchored curriculum checkpoint")
    loop.load_state(blob)
    sampler.load_state_dict(blob["sampler"])
    return blob


def _resume_cursor(blob: dict, stages: tuple[CurriculumStage, ...]) -> tuple[int, int]:
    name = blob.get("stage")
    indexes = [index for index, stage in enumerate(stages) if stage.name == name]
    if len(indexes) != 1:
        raise ValueError("resume checkpoint has no curriculum stage")
    index = indexes[0]
    step = int(blob["step"])
    if step >= stages[index].steps:
        return index + 1, 0
    if step < 0:
        raise ValueError("resume step is negative")
    return index, step


def _learning_scale(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, step / warmup)


def correction_mse(output, target: torch.Tensor) -> tuple[float, float]:
    """Unweighted mean square of the native velocity and of the first correction.

    A bypass has no correction velocity. Its first-correction error is the
    native error, so the comparison does not count as an improvement.
    """
    native = output.round_velocities[0]
    mse_0 = float((native.detach().float() - target.detach().float()).pow(2).mean())
    if len(output.round_velocities) < 2:
        return mse_0, mse_0
    corrected = output.round_velocities[1]
    mse_1 = float((corrected.detach().float() - target.detach().float()).pow(2).mean())
    return mse_0, mse_1


def stage1_val_flags(mse_0: float, mse_1: float) -> dict:
    """Strict comparison on one probe. This does not certify the stage."""
    return {"v1_below_v0": bool(mse_1 < mse_0)}


def stage2_val_flags(
    dual: float,
    gen_only: float,
    shuffled: float,
    zero: float,
    *,
    shuffled_measured: bool,
) -> dict:
    """Dual against the frozen GEN-only path, then the two content ablations.

    ``shuffled_rose`` stays false when the gate bypassed and no donor existed.
    None of these flags is a training stop.
    """
    return {
        "dual_below_gen_only": bool(dual < gen_only),
        "zero_rose": bool(zero > dual),
        "shuffled_rose": bool(shuffled_measured and shuffled > dual),
        "shuffled_measured": bool(shuffled_measured),
    }


def freeze_val_probe(source, train: AnchoredTrainConfig, t_min: float, t_max: float):
    """One fixed generation-val draw. Later checks reuse these rows and times."""
    rows = source.take(train.val_images)
    if len(rows) != train.val_images:
        raise RuntimeError("validation probe did not return the requested rows")
    pairs = train.val_images // 2
    generator = torch.Generator().manual_seed(train.seed + 1009)
    times = sample_timesteps(
        pairs, timestep_shift=train.timestep_shift, device=torch.device("cpu"),
        generator=generator, t_min=t_min, t_max=t_max,
    )
    noise = random.Random(train.seed + 1009)
    seeds = [noise.randrange(2**63 - 1) for _ in range(train.val_images)]
    return rows, [float(value) for value in times], seeds


def _statistic_keys(rounds: int) -> tuple[str, ...]:
    keys = ("total", "final", "improve", "mse_0", "mse_1", "error_fell_1")
    if rounds >= 2:
        return keys + ("mse_2",)
    return keys


def _probe_forward(loop, bundle, row, flow_time, noise_seed, resolution, posterior_seed, *, variant, mode, donor):
    saved = (loop.config.variant, loop.config.correction_rounds, loop.config.feedback_mode)
    loop.config.variant = variant
    loop.config.correction_rounds = 1
    loop.config.feedback_mode = mode
    try:
        x0 = encode_target(
            bundle, row["image"], resolution, posterior_seed_for(row["image"], posterior_seed),
        )
        latent, target = paired_latents(x0, flow_time, noise_seed)
        flow = conditional_flow(bundle, row["caption"], resolution)
        output = forward_correction(
            loop, bundle.model, latent, flow_time, flow, train=False, feedback_donor=donor,
        )
        return correction_mse(output, target), output.feedback
    finally:
        loop.config.variant, loop.config.correction_rounds, loop.config.feedback_mode = saved


def evaluate_val_probe(loop, bundle, probe, train: AnchoredTrainConfig, stage: str) -> dict:
    """Eval-mode probe. The first correction is the measured velocity on every arm."""
    rows, times, seeds = probe
    pair_count = len(rows) // 2
    names = ("mse_0", "mse_1") if stage == "gen_only" else (
        "mse_0", "mse_1", "gen_only", "shuffled", "zero", "shuffled_measured",
    )
    totals = {name: 0.0 for name in names}
    was_training = loop.training
    loop.eval()
    try:
        with torch.no_grad():
            for pair_index in range(parallel.rank(), pair_count, parallel.world_size()):
                flow_time = times[pair_index]
                left = rows[2 * pair_index]
                right = rows[2 * pair_index + 1]
                specs = (
                    (left, seeds[2 * pair_index], stage, "correct", None),
                    (right, seeds[2 * pair_index + 1], stage, "correct", None),
                )
                measured = []
                for row, noise_seed, variant, mode, donor in specs:
                    (mse_0, mse_1), feedback = _probe_forward(
                        loop, bundle, row, flow_time, noise_seed, train.resolution,
                        train.posterior_seed, variant=variant, mode=mode, donor=donor,
                    )
                    totals["mse_0"] += mse_0
                    totals["mse_1"] += mse_1
                    measured.append(feedback)
                if stage != "dual":
                    continue
                for row, noise_seed, feedback in (
                    (left, seeds[2 * pair_index], measured[1]),
                    (right, seeds[2 * pair_index + 1], measured[0]),
                ):
                    ready = feedback is not None
                    donor = feedback if ready else None
                    mode = "shuffled" if ready else "correct"
                    (_mse_0, mse_1), _feedback = _probe_forward(
                        loop, bundle, row, flow_time, noise_seed, train.resolution,
                        train.posterior_seed, variant="dual", mode=mode, donor=donor,
                    )
                    totals["shuffled"] += mse_1
                    totals["shuffled_measured"] += 1.0 if ready else 0.0
                for row, noise_seed in (
                    (left, seeds[2 * pair_index]),
                    (right, seeds[2 * pair_index + 1]),
                ):
                    (_mse_0, gen_mse), _feedback = _probe_forward(
                        loop, bundle, row, flow_time, noise_seed, train.resolution,
                        train.posterior_seed, variant="gen_only", mode="correct", donor=None,
                    )
                    (_zero_0, zero_mse), _feedback = _probe_forward(
                        loop, bundle, row, flow_time, noise_seed, train.resolution,
                        train.posterior_seed, variant="dual", mode="zero", donor=None,
                    )
                    totals["gen_only"] += gen_mse
                    totals["zero"] += zero_mse
    finally:
        if was_training:
            loop.train()
        else:
            loop.eval()
    device = bundle.model.language_model.model.embed_tokens.weight.device
    reduced = parallel.sum_statistics([totals[name] for name in names], device)
    averaged = {
        name: reduced[index] / train.val_images for index, name in enumerate(names)
    }
    record = {
        "stage": stage,
        "val_images": train.val_images,
        "val_mse_0": averaged["mse_0"],
        "val_mse_1": averaged["mse_1"],
        **stage1_val_flags(averaged["mse_0"], averaged["mse_1"]),
    }
    if stage == "dual":
        measured = averaged["shuffled_measured"] == 1.0
        record.update({
            "val_gen_only_mse_1": averaged["gen_only"],
            "val_shuffled_mse_1": averaged["shuffled"],
            "val_zero_mse_1": averaged["zero"],
            **stage2_val_flags(
                averaged["mse_1"], averaged["gen_only"], averaged["shuffled"], averaged["zero"],
                shuffled_measured=measured,
            ),
        })
    return record


def train_anchored(
    loop,
    bundle,
    pairs,
    train: AnchoredTrainConfig,
    output: Path,
    *,
    observation_checkpoint: str,
    val_pairs,
    resume: str | None = None,
) -> None:
    from ..bagel.accelerator import max_memory_allocated, reset_peak_memory_stats, synchronize
    from ..diagnostics.training_logging import configure_logging

    require_gen_only_loop(loop.config)
    train.validate()
    require_sixty_four(parallel.world_size(), train.global_batch)
    reader = require_stage2_reader(observation_checkpoint, loop.config.observation_source_mode)
    output = Path(output)
    sampler = pairs
    blob = None
    if resume:
        blob = load_checkpoint(resume, loop, sampler)
        saved = (blob.get("observation") or {}).get("checkpoint")
        if saved and str(Path(saved)) != str(reader):
            raise ValueError("resume observation checkpoint differs from the launch path")
        stage_index, stage_step = _resume_cursor(blob, train.stages)
        if stage_index >= len(train.stages):
            raise ValueError("this curriculum checkpoint is already complete")
    else:
        parallel.broadcast_parameters(loop)
        stage_index, stage_step = 0, 0
    bundle.model.eval()
    bundle.vae.eval()
    guard = FrozenGuard(bundle.model, bundle.vae)
    times = scheduled_flow_times(loop.config, train.sampler_steps, train.timestep_shift)
    t_min, t_max = schedule_time_window(loop.config)
    probe = freeze_val_probe(val_pairs, train, t_min, t_max)
    logger = configure_logging(output)
    device = bundle.model.language_model.model.embed_tokens.weight.device
    reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    if parallel.rank() == 0:
        write_json(output / "run.json", {
            "experiment": "anchored_dual_state_loop",
            "trainer": TRAINER,
            "devices": parallel.world_size(),
            "loop": loop.config.to_dict(),
            "train": train.to_dict(),
            "scheduled_flow_times": times,
            "time_sampling": "shifted_interval",
            "time_window": [t_min, t_max],
            "train_pairs": sampler.pair_count,
            "text_field": sampler.text_field,
            "source_kind": sampler.source_kind,
            "corpus_fingerprint": sampler.fingerprint,
            "val_pairs": val_pairs.pair_count,
            "val_images": train.val_images,
            "observation_checkpoint": str(reader),
            "resume_stage_index": stage_index,
            "resume_step": stage_step,
        })
    parallel.barrier()
    for index, stage in enumerate(train.stages):
        if index < stage_index:
            continue
        groups = apply_stage_parameters(loop, stage)
        if stage.name == "dual":
            prepare_dual_reader(loop, str(reader))
            if parallel.rank() == 0:
                write_json(output / "stage2.json", loop.observer.status())
            parallel.barrier()
        loop.train()
        parameters = [parameter for group in groups for parameter in group["params"]]
        optimizer = torch.optim.AdamW(groups, weight_decay=train.weight_decay)
        start = stage_step if index == stage_index else 0
        if blob is not None and index == stage_index and start > 0:
            optimizer.load_state_dict(blob["optimizer"])
            place_optimizer_state(optimizer)
        for step in range(start + 1, stage.steps + 1):
            rounds = correction_rounds_for(stage, step)
            if loop.config.correction_rounds != rounds:
                loop.config.correction_rounds = rounds
                loop.config.validate(len(loop.decoder.layers))
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batch = sampler.take(train.global_batch)
            generator = torch.Generator()
            generator.manual_seed(sampler.rng.randrange(2**31))
            flow_times = sample_timesteps(
                len(batch), timestep_shift=train.timestep_shift, device=device,
                generator=generator, t_min=t_min, t_max=t_max,
            )
            draws = [
                (float(flow_times[item]), sampler.rng.randrange(2**63 - 1))
                for item in range(len(batch))
            ]
            keys = _statistic_keys(rounds)
            totals = [0.0] * len(keys)
            for item in range(parallel.rank(), train.global_batch, parallel.world_size()):
                row = batch[item]
                flow_time, noise_seed = draws[item]
                x0 = encode_target(
                    bundle, row["image"], train.resolution,
                    posterior_seed_for(row["image"], train.posterior_seed),
                )
                latent, target = paired_latents(x0, flow_time, noise_seed)
                flow = conditional_flow(bundle, row["caption"], train.resolution)
                output_step = forward_correction(loop, bundle.model, latent, flow_time, flow)
                if not output_step.corrected:
                    raise RuntimeError("training flow time fell outside the correction window")
                loss, parts = anchored_loss(
                    output_step.velocity, target, output_step.native_velocity, output_step.delta,
                    output_step.anchor, flow["packed_vae_token_indexes"],
                    lambda_anchor=loop.config.lambda_anchor, lambda_entry=loop.config.lambda_entry,
                    round_velocities=output_step.round_velocities,
                    round_loss_weights=loop.config.round_loss_weights,
                    lambda_improve=train.lambda_improve,
                    improvement_margin=train.improvement_margin,
                )
                (loss / train.global_batch).backward()
                for key_index, key in enumerate(keys):
                    totals[key_index] += float(parts[key])
                del output_step, loss, latent, target, flow, x0
            if stage.name == "dual":
                loop.observer.freeze()
            parallel.sum_gradients(loop)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, train.gradient_clip, error_if_nonfinite=True,
            )
            scale = _learning_scale(step, stage.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            optimizer.step()
            guard.check()
            loop.assert_native_frozen()
            reduced = parallel.sum_statistics(totals, device)
            synchronize(device)
            averaged = {
                key: reduced[key_index] / train.global_batch for key_index, key in enumerate(keys)
            }
            record = {
                "stage": stage.name,
                "step": step,
                "correction_rounds": rounds,
                "loss": averaged["total"],
                "flow": averaged["final"],
                "improve": averaged["improve"],
                "mse_0": averaged["mse_0"],
                "mse_1": averaged["mse_1"],
                "error_fell_1": averaged["error_fell_1"],
                "gate": float(loop.gate.raw.detach()),
                "gradient_norm": float(grad_norm),
                "learning_rate": stage.learning_rate * scale,
                "output_learning_rate": stage.output_learning_rate * scale,
                "step_seconds": time.perf_counter() - step_started,
                "elapsed_seconds": time.perf_counter() - started_at,
                "peak_memory_gib": parallel.max_statistics(
                    [max_memory_allocated(device) / 1024**3], device,
                )[0],
            }
            if "mse_2" in averaged:
                record["mse_2"] = averaged["mse_2"]
            if parallel.rank() == 0:
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                logger.info("stage=%s step=%d %s", stage.name, step, json.dumps(record))
            if step in stage.save_steps or step == stage.steps:
                val_record = evaluate_val_probe(loop, bundle, probe, train, stage.name)
                val_record["step"] = step
                val_record["gate"] = float(loop.gate.raw.detach())
                if parallel.rank() == 0:
                    with (output / "val.jsonl").open("a") as stream:
                        stream.write(json.dumps(val_record) + "\n")
                    logger.info("val stage=%s step=%d %s", stage.name, step, json.dumps(val_record))
                    save_checkpoint(output, loop, optimizer, sampler, train, stage.name, step)
                parallel.barrier()


