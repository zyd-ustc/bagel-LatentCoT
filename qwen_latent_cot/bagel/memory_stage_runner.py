"""Command implementation shared by the four v2 stage entrypoints."""
import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
import yaml

from .memory_grounding import write_prompt_mask_probability
from .memory_training import (SCHEMA, GroundingRuntime, append_json, asdict_tensors,
                              load_records, validate_config, validate_adapter_metadata)
from .loop_supervision import loop_supervision_loss
from .memory_distributed import (DistributedContext, RankInfo, RankBatchSampler,
                                 aggregate_rank_metrics, distributed_contract)

LOGGER = logging.getLogger("bagel.memory_v2")


def arguments(stage):
    configs = dict(reader="memory_reader_grounding", writer="memory_writer_effect",
                   grpo="memory_grpo", loop="loop_supervision_r4", eval="memory_reader_grounding")
    p = argparse.ArgumentParser(description=f"BAGEL memory grounding v2: {stage}")
    p.add_argument("--config", default=("configs/evaluation/memory_causality.yaml" if stage == "eval"
                                       else f"configs/training/{configs[stage]}.yaml"))
    for name in ("model-path", "data-path", "output-dir", "adapter-path", "device"):
        p.add_argument(f"--{name}", default=None)
    for name in ("max-steps", "max-prompts", "num-write-rounds", "batch-size"):
        p.add_argument(f"--{name}", type=int, default=None)
    p.add_argument("--validate-only", action="store_true", help="validate config/data/paths without loading weights")
    if stage == "eval":
        p.add_argument("--generate-images", action="store_true")
    return p.parse_args()


def preflight(args, stage):
    config = yaml.safe_load(Path(args.config).read_text()) or {}
    for key in ("model_path", "data_path", "output_dir", "adapter_path", "device",
                "max_steps", "max_prompts", "num_write_rounds", "batch_size"):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    config = validate_config(config, stage)
    distributed_contract(config, stage)
    for key in ("model_path", "data_path", "adapter_path"):
        if config.get(key) and not Path(config[key]).exists():
            raise FileNotFoundError(f"{key}: {config[key]}")
    if stage == "eval" and config.get("adapter_path"):
        meta = json.loads(Path(config["adapter_path"]).with_suffix(".json").read_text())
        if args.num_write_rounds is None:
            config["num_write_rounds"] = int(meta["num_write_rounds"])
        for key in ("lora_rank", "lora_alpha", "reader_o_enabled"):
            config[key] = meta[key]
    config = validate_config(config, stage)
    validate_adapter_metadata(config, stage)
    records = load_records(config)
    config.update(distributed_contract(config, stage, len(records)))
    if stage == "grpo":
        from .memory_grpo import validate_grpo_config
        validate_grpo_config(config, records)
        for key in ("diffusion_rm_repo", "flux_rm_config", "flux_rm_checkpoint"):
            if not Path(config[key]).exists():
                raise FileNotFoundError(f"{key}: {config[key]}")
    return config, records


def new_output(config):
    output = Path(config["output_dir"]).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix or overwrite a previous run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(dict(schema=SCHEMA, **config), indent=2) + "\n")
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for pattern in ("qwen_latent_cot/bagel/**/*.py", "scripts/train/bagel_*memory*.py",
                    "scripts/train/bagel_loop_supervision.py", "scripts/evaluate/bagel_memory_causality_eval.py",
                    "configs/training/memory*.yaml", "configs/training/loop_supervision*.yaml"):
        for path in root.glob(pattern):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
              for key in ("data_path", "adapter_path") if config.get(key)}
    (output / "run_manifest.json").write_text(json.dumps(dict(schema=SCHEMA,
        source_sha256=sources, input_sha256=inputs, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, config=config, git_revision="not_assumed"), indent=2))
    return output


def checked_step(runtime, optimizer, loss, config, distributed=None):
    distributed = distributed or DistributedContext()
    distributed.fail_if(not bool(torch.isfinite(loss)), "non-finite training loss", FloatingPointError)
    loss.backward()
    missing, unexpected = [], []
    for name, p in runtime.model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing.append(name)
        if not p.requires_grad and p.grad is not None:
            unexpected.append(name)
    distributed.fail_if(bool(missing or unexpected),
        f"gradient routing failed: missing={missing[:5]}, unexpected={unexpected[:5]}")
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    distributed.average_gradients(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(config.get("max_grad_norm", 1.)), error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def main(stage):
    args = arguments(stage)
    config, records = preflight(args, stage)
    info = RankInfo.from_env()
    logging.basicConfig(level=logging.INFO if info.rank == 0 else logging.WARNING,
                        format=f"%(asctime)s [rank {info.rank}] %(levelname)s %(message)s")
    if args.validate_only:
        if info.rank == 0:
            print(json.dumps(dict(stage=stage, records=len(records), config=config), indent=2))
        return
    distributed = DistributedContext.start(config)
    output = None
    try:
        distributed.verify_inputs(config, records)
        output = Path(distributed.primary_call(lambda: str(new_output(config))))
        runtime_config = dict(config)
        if distributed.active:
            runtime_config["device"] = str(distributed.device)
        LOGGER.info("loading %s: world_size=%d per_rank_batch=%d global_batch=%d lr=%s",
                    stage, info.world_size, config["batch_size"], config["global_batch_size"],
                    config.get("learning_rate", 5e-6))
        runtime = GroundingRuntime(runtime_config, stage)
        distributed.synchronize_initial_adapters(runtime.model)
        distributed.primary_call(lambda: (output / "trainable_routes.json").write_text(
            json.dumps(runtime.trainable_names, indent=2)))
        train_stage(runtime, records, output, config, distributed)
    except Exception as exc:
        if distributed.primary and output is not None:
            (output / "status.json").write_text(json.dumps(dict(status="failed", stage=stage,
                world_size=info.world_size, error=f"{type(exc).__name__}: {exc}"), indent=2))
        raise
    finally:
        distributed.close()


def train_stage(runtime, records, output, config, distributed):
    """Shared train loop; accepts a tiny runtime for multi-process integration tests."""
    stage = runtime.stage
    optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad],
        lr=float(config.get("learning_rate", 5e-6)), betas=(.9, .95), weight_decay=0.)
    if stage == "grpo":
        from .memory_grpo import train_grpo
        train_grpo(runtime, records, optimizer, output)
        return
    seed = int(config.get("seed", 42))
    info = distributed.info
    generator = torch.Generator().manual_seed(seed + info.rank * 1009)
    cached, batch_index = [], 0
    batch_size = config["batch_size"]
    sampler = RankBatchSampler(len(records), batch_size, rank=info.rank,
        world_size=info.world_size, seed=seed, shuffle=bool(config.get("shuffle_data", False)))
    started = time.monotonic()
    for step in range(1, config["max_steps"] + 1):
        step_started = time.monotonic()
        if not cached:
            selected = [records[index] for index in sampler.indices(batch_index)]
            # A complete rollout is shared by states_per_prompt updates.
            batch_seed = seed + batch_index * 100003
            LOGGER.info("preparing native/target states: batch=%d next_step=%d", batch_index, step)
            samples = [runtime.states(row, batch_seed + info.rank * batch_size + j,
                                     state_seed=batch_seed) for j, row in enumerate(selected)]
            expected_states = config["states_per_prompt"] if config["objective"] == "native_teacher" else 1
            distributed.fail_if(any(len(values) != expected_states for values in samples),
                                "rollout returned an unexpected state count")
            cached = [list(items) for items in zip(*samples)]
            batch_index += 1
        items = cached.pop(0)
        optimizer.zero_grad(set_to_none=True)
        if stage == "loop":
            results = [runtime.write(item) for item in items]
            velocities = [torch.stack([out.write_round_velocities[r] for out in results])
                          for r in range(config["num_write_rounds"])]
            obj = loop_supervision_loss(velocities, torch.stack([it.target for it in items]),
                weights=config.get("write_round_weights", [1.] * len(velocities)),
                loop_distill=config.get("loop_distill", False),
                final_round_validated=config.get("final_round_validated", False),
                lambda_loop_distill=float(config.get("lambda_loop_distill", .2)),
                lambda_loop_monotonic=float(config.get("lambda_loop_monotonic", .1)),
                monotonic_margin=float(config.get("loop_monotonic_margin", 0.)))
            loss = obj.loss
            metrics = {k: v.detach().cpu().tolist() for k, v in asdict_tensors(obj).items()}
            metrics.update(sample_ids=[str(it.record["id"]) for it in items],
                           timesteps=[it.timestep for it in items])
        else:
            prob = write_prompt_mask_probability(step - 1,
                start=float(config.get("write_prompt_mask_prob_start", .5 if stage == "reader" else 0.)),
                end=float(config.get("write_prompt_mask_prob_end", .1 if stage == "reader" else 0.)),
                decay_steps=int(config.get("write_prompt_mask_decay_steps", 3000)))
            masks = (torch.rand(len(items), generator=generator) < prob).tolist()
            loss, metrics = runtime.dependency(items, masks=masks, generator=generator,
                                               detach_read=stage == "reader")
            metrics.update(write_prompt_mask_probability=prob, mask_samples=masks)
        norm = checked_step(runtime, optimizer, loss, config, distributed)
        metrics.update(step=step, grad_norm=norm)
        metrics.update(rank=info.rank, step_seconds=time.monotonic()-step_started)
        rows = distributed.gather(metrics)
        global_metrics = aggregate_rank_metrics(rows)
        global_metrics.update(step=step, world_size=info.world_size,
            global_batch_size=batch_size*info.world_size, per_rank_batch_size=batch_size,
            elapsed_seconds=time.monotonic()-started)
        distributed.primary_call(lambda: append_json(output / "metrics.jsonl", global_metrics))
        LOGGER.info("stage=%s step=%d/%d global_loss=%.6f grad_norm=%.6f world=%d",
                    stage, step, config["max_steps"], global_metrics["loss"], norm, info.world_size)
        if step % config["save_steps"] == 0 or step == config["max_steps"]:
            distributed.primary_call(lambda: runtime.save(output, step, optimizer))
    distributed.primary_call(lambda: (output / "status.json").write_text(json.dumps(dict(
        status="complete", stage=stage, steps=config["max_steps"], world_size=info.world_size,
        global_batch_size=batch_size*info.world_size, heldout_gate="not_evaluated"), indent=2)))
