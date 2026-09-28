"""V2 on-policy Flow-GRPO with paired correct-memory/shuffled-memory rewards."""
from dataclasses import replace
import json
import logging

import torch

from .flow_grpo import sde_step_with_logprob, clipped_grpo_loss, gaussian_mean_kl
from .loop import loop_adapter_state_dict, load_loop_adapter_state_dict
from .memory_grounding import causal_reward_advantages
from .memory_training import ARMS, append_json, tensor_hash


def validate_grpo_config(c, records):
    required = ("geneval_url", "diffusion_rm_repo", "flux_rm_config", "flux_rm_checkpoint")
    if any(not c.get(k) for k in required):
        raise ValueError("GRPO requires real GenEval and FLUX quality reward services/weights")
    if c.get("group_size", 4) < 2 or c.get("noise_level", .8) <= 0:
        raise ValueError("GRPO requires group_size>=2 and positive SDE noise")
    indices = c.get("sde_step_indices", [])
    if not indices or len(indices) != len(set(indices)) or any(
        not isinstance(i, int) or not 0 <= i < c["num_steps"] - 2 for i in indices
    ):
        raise ValueError("select unique, nonterminal stochastic step indexes")
    if any(not r.get("vqa_list") or len(r.get("skills", [])) != len(r["vqa_list"]) for r in records):
        raise ValueError("GRPO requires per-atom GenEval metadata (vqa_list and skills)")


@torch.no_grad()
def rollout_arms(runtime, records, seed, *, stochastic_steps=(), noise_level=.8):
    """Paired initial and transition noise; independent four-arm trajectories.

    At every step shuffled memory is a whole-sample derangement of the current
    shuffled-arm Read batch. Only correct-arm transitions are optimized.
    """
    prepared = [runtime.conditions(r, seed + i) for i, r in enumerate(records)]
    items = [pair[0] for pair in prepared]
    initial = [pair[1] for pair in prepared]
    samples = {a: [v.clone() for v in initial] for a in ARMS}
    ts, dts = runtime.model.prepare_image_schedule(runtime.config["num_steps"],
        float(runtime.config.get("timestep_shift", 3.)), runtime.device)
    trajectories = [[] for _ in items]
    generator = torch.Generator().manual_seed(seed + 7919)
    for index, (t, dt) in enumerate(zip(ts, dts)):
        noises = [torch.randn(x.shape, generator=generator).to(x) for x in initial]
        for arm in ARMS:
            current = [replace(item, sample=samples[arm][i], timestep=float(t)) for i, item in enumerate(items)]
            if arm == "shuffled":
                memory = [runtime.read_memory(it, detach=True) for it in current]
            for i, item in enumerate(current):
                if arm == "native":
                    v = runtime.native_velocity(item)
                else:
                    override = (torch.zeros_like(runtime.model.loop_memory[:8]) if arm == "zero"
                                else memory[(i + 1) % len(items)] if arm == "shuffled" else None)
                    v = runtime.write(item, override).final_velocity
                if index in stochastic_steps:
                    transition = sde_step_with_logprob(v, timestep=t, next_timestep=t-dt,
                        sample=item.sample, noise=noises[i], noise_level=noise_level)
                    next_x = transition.next_sample
                    if arm == "correct":
                        trajectories[i].append(dict(sample=item.sample.detach().cpu(), timestep=float(t),
                            next_timestep=float(t-dt), next_sample=next_x.detach().cpu(),
                            old_log_prob=transition.log_prob.detach().cpu(), step=index))
                else:
                    next_x = runtime.model.image_euler_step(item.sample, v, dt)
                samples[arm][i] = next_x.detach()
    return samples, trajectories, items, [tensor_hash(x) for x in initial]


def train_grpo(runtime, records, optimizer, output):
    from .rewards import GenEvalRewardClient, FluxLatentReward, audit_bagel_flux_vae_contract
    c = runtime.config
    validate_grpo_config(c, records)
    semantic = GenEvalRewardClient(c["geneval_url"])
    semantic.check_available()
    audit_bagel_flux_vae_contract(runtime.vae)
    quality = FluxLatentReward(diffusion_rm_repo=c["diffusion_rm_repo"],
        config_path=c["flux_rm_config"], checkpoint_path=c["flux_rm_checkpoint"], device=runtime.device)
    reference = {k: v.clone() for k, v in loop_adapter_state_dict(runtime.model).items()}
    group_size = int(c.get("group_size", 4))
    for step in range(1, c["max_steps"] + 1):
        record = records[(step - 1) % len(records)]
        donor = records[step % len(records)]
        groups, latents, images = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
        for g in range(group_size):
            seed = int(c.get("seed", 42)) + step * 100003 + g
            samples, trajectories, items, hashes = rollout_arms(runtime, [record, donor], seed,
                stochastic_steps=tuple(c["sde_step_indices"]), noise_level=float(c.get("noise_level", .8)))
            groups.append((items[0], trajectories[0], seed, hashes[0]))
            with torch.no_grad(), runtime.autocast():
                for arm in ARMS:
                    latents[arm].append(samples[arm][0])
                    images[arm].append(runtime.inferencer.decode_image(samples[arm][0], runtime.shape))
        semantic_scores = {a: semantic.score(images[a], [record] * group_size) for a in ARMS}
        quality_scores = {a: quality.score(latents[a], [record["prompt"]] * group_size,
                                           image_shape=runtime.shape) for a in ("correct", "native")}
        objective, advantage, terms = causal_reward_advantages(semantic_scores["correct"],
            semantic_scores["shuffled"], quality_scores["correct"], quality_scores["native"],
            lambda_memory=float(c.get("lambda_memory", 1.)), lambda_quality=float(c.get("lambda_quality", 1.)),
            quality_tolerance=float(c.get("quality_tolerance", 0.)))
        if not torch.isfinite(objective).all():
            raise FloatingPointError("non-finite reward")
        # Compute fixed-reference means before constructing the policy graphs.
        current_adapter = {k: v.clone() for k, v in loop_adapter_state_dict(runtime.model).items()}
        references = []
        try:
            load_loop_adapter_state_dict(runtime.model, reference)
            with torch.no_grad():
                for item, trajectory, _, _ in groups:
                    local = []
                    for tr in trajectory:
                        replay = replace(item, sample=tr["sample"].to(runtime.device), timestep=tr["timestep"])
                        v = runtime.write(replay).final_velocity
                        transition = sde_step_with_logprob(v, sample=replay.sample,
                            timestep=tr["timestep"], next_timestep=tr["next_timestep"],
                            next_sample=tr["next_sample"].to(runtime.device), noise_level=float(c.get("noise_level", .8)))
                        local.append(transition.mean.detach().cpu())
                    references.append(local)
        finally:
            load_loop_adapter_state_dict(runtime.model, current_adapter)
        optimizer.zero_grad(set_to_none=True)
        # Accumulate per transition; keep only one expensive BAGEL graph alive.
        count = sum(len(g[1]) for g in groups)
        total_loss, total_kl = 0., 0.
        for g, (item, trajectory, _, _) in enumerate(groups):
            for j, tr in enumerate(trajectory):
                replay = replace(item, sample=tr["sample"].to(runtime.device), timestep=tr["timestep"])
                v = runtime.write(replay).final_velocity
                transition = sde_step_with_logprob(v, sample=replay.sample,
                    timestep=tr["timestep"], next_timestep=tr["next_timestep"],
                    next_sample=tr["next_sample"].to(runtime.device), noise_level=float(c.get("noise_level", .8)))
                pg = clipped_grpo_loss(transition.log_prob, tr["old_log_prob"].to(runtime.device),
                    advantage[g].to(runtime.device), clip_range=float(c.get("clip_range", 1e-5)))
                std = transition.std * (tr["timestep"] - tr["next_timestep"]) ** .5
                kl = gaussian_mean_kl(transition.mean, references[g][j].to(runtime.device), std)
                loss = (pg + float(c.get("kl_weight", .01)) * kl) / count
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite GRPO loss")
                loss.backward()
                total_loss += float(loss.detach())
                total_kl += float(kl.detach()) / count
        params = [p for p in runtime.model.parameters() if p.requires_grad]
        if any(p.grad is None for p in params):
            raise RuntimeError("GRPO has missing trainable gradients")
        norm = torch.nn.utils.clip_grad_norm_(params, float(c.get("max_grad_norm", 1.)), error_if_nonfinite=True)
        optimizer.step()
        row = dict(step=step, sample_id=record["id"], donor_id=donor["id"], loss=total_loss,
            kl=total_kl, grad_norm=float(norm), objective=objective.tolist(),
            semantic={a: s.tolist() for a, s in semantic_scores.items()},
            quality={a: s.tolist() for a, s in quality_scores.items()},
            semantic_dependency_gap=terms.semantic_delta.tolist(), advantage=advantage.tolist(),
            seeds=[g[2] for g in groups], initial_noise_hashes=[g[3] for g in groups])
        append_json(output / "metrics.jsonl", row)
        logging.info("GRPO step=%d/%d correct-shuffle=%.6f", step, c["max_steps"], float(terms.semantic_delta.mean()))
        if step % c["save_steps"] == 0 or step == c["max_steps"]:
            runtime.save(output, step, optimizer)
    (output / "status.json").write_text(json.dumps(dict(status="complete", stage="grpo", steps=c["max_steps"],
                                                      heldout_gate="not_evaluated")))
