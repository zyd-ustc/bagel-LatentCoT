#!/usr/bin/env python3
"""V2 fixed-state causality and optional same-seed four-arm image evaluation."""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from qwen_latent_cot.bagel.memory_stage_runner import arguments, preflight, new_output
from qwen_latent_cot.bagel.memory_training import ARMS, GroundingRuntime, append_json
from qwen_latent_cot.bagel.memory_grpo import rollout_arms


@torch.no_grad()
def evaluate(runtime, records, output, *, generate_images=False):
    config = runtime.config
    if len(records) % config["batch_size"]:
        raise ValueError("evaluation prompts must form complete distinct-prompt batches (no silent dropping)")
    all_metrics, galleries = [], []
    semantic, quality = None, None
    image_scores = []
    if config.get("geneval_url"):
        if not generate_images:
            raise ValueError("semantic evaluation requires --generate-images")
        from qwen_latent_cot.bagel.rewards import GenEvalRewardClient
        semantic = GenEvalRewardClient(config["geneval_url"])
        semantic.check_available()
    quality_keys = ("diffusion_rm_repo", "flux_rm_config", "flux_rm_checkpoint")
    if any(config.get(key) for key in quality_keys):
        if not generate_images or not all(config.get(key) for key in quality_keys):
            raise ValueError("quality evaluation requires images and all FLUX reward paths")
        from qwen_latent_cot.bagel.rewards import FluxLatentReward, audit_bagel_flux_vae_contract
        audit_bagel_flux_vae_contract(runtime.vae)
        quality = FluxLatentReward(diffusion_rm_repo=config["diffusion_rm_repo"],
            config_path=config["flux_rm_config"], checkpoint_path=config["flux_rm_checkpoint"], device=runtime.device)
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        batch_seed = int(config.get("seed", 42)) + start
        states = [runtime.states(row, batch_seed + j, state_seed=batch_seed) for j, row in enumerate(batch)]
        for state_index, items in enumerate(zip(*states)):
            attention = []
            _, metrics = runtime.dependency(list(items), detach_read=True, attention=attention,
                generator=torch.Generator().manual_seed(int(config.get("seed", 42)) + start + state_index))
            # Full-suffix per-Write flow error, not intermediate body hidden error.
            out = [runtime.write(item) for item in items]
            metrics["write_round_errors"] = [float(torch.stack([
                (result.write_round_velocities[r].float() - item.target.float()).square().mean()
                for result, item in zip(out, items)]).mean()) for r in range(config["num_write_rounds"])]
            metrics.update(batch_start=start, state_index=state_index, attention=attention,
                           prompt_mask=False, target="native_velocity" if config["objective"] == "native_teacher" else "target_flow")
            append_json(output / "metrics.jsonl", metrics)
            all_metrics.append(metrics)
        if generate_images:
            samples, _, _, hashes = rollout_arms(runtime, batch, int(config.get("seed",42)) + start)
            decoded = {a: [] for a in ARMS}
            with runtime.autocast():
                for j, record in enumerate(batch):
                    folder = output / f"p{start+j:04d}"
                    folder.mkdir()
                    for arm in ARMS:
                        image = runtime.inferencer.decode_image(samples[arm][j], runtime.shape)
                        image.save(folder / f"{arm}.png")
                        decoded[arm].append(image)
                    manifest = dict(id=record["id"], prompt=record["prompt"], arms=list(ARMS),
                        seed=int(config.get("seed",42))+start+j, initial_noise_hash=hashes[j],
                        donor_id=batch[(j+1)%len(batch)]["id"], num_write_rounds=config["num_write_rounds"],
                        cfg_text_scale=1., prompt_mask=False)
                    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
                    galleries.append((folder.name, record["prompt"]))
            sem = {a: semantic.score(decoded[a], batch).tolist() for a in ARMS} if semantic else {}
            qual = {a: quality.score(samples[a], [r["prompt"] for r in batch],
                                     image_shape=runtime.shape).tolist() for a in ARMS} if quality else {}
            if sem or qual:
                for j, record in enumerate(batch):
                    scores = dict(id=record["id"], semantic={a:v[j] for a,v in sem.items()},
                                  quality={a:v[j] for a,v in qual.items()})
                    image_scores.append(scores)
                    append_json(output / "image_scores.jsonl", scores)
        print(f"[{start+len(batch)}/{len(records)}] fixed-state evaluation complete", flush=True)
    keys = ("error_correct", "error_shuffled", "error_zero", "dependency_gap_shuffle",
            "dependency_gap_zero", "direction_cosine", "relative_dv_shuffled", "relative_dv_zero")
    summary = {key: sum(row[key] for row in all_metrics) / len(all_metrics) for key in keys}
    summary.update(prompts=len(records), states=len(all_metrics) * config["batch_size"],
        shuffle_positive_batch_state_fraction=sum(r["dependency_gap_shuffle"] > 0 for r in all_metrics) / len(all_metrics),
        semantic_scored=semantic is not None, quality_scored=quality is not None,
        gate="requires held-out consistency and quality review",
        cfg_text_scale=1., note="paired batch-state means; not confidence intervals or semantic scores")
    if semantic:
        summary["semantic_correct_minus_shuffled"] = sum(
            r["semantic"]["correct"] - r["semantic"]["shuffled"] for r in image_scores) / len(image_scores)
    if quality:
        summary["quality_correct_minus_native"] = sum(
            r["quality"]["correct"] - r["quality"]["native"] for r in image_scores) / len(image_scores)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if galleries:
        page = "<!doctype html><meta charset='utf-8'><title>Memory causality v2</title><style>img{width:24%;}section{margin:24px 0}</style>"
        page += "<h1>native / zero / shuffled / correct</h1><p>CFG=1; prompt KV preserved; only Write-entry memory changes.</p>"
        for folder, prompt in galleries:
            page += f"<section><h2>{html.escape(prompt)}</h2>"
            page += "".join(f"<a href='{folder}/{a}.png'><img src='{folder}/{a}.png' title='{a}'></a>" for a in ARMS)
            page += "</section>"
        (output / "index.html").write_text(page)
    (output / "status.json").write_text(json.dumps(dict(status="complete", **summary), indent=2))


def main():
    args = arguments("eval")
    config, records = preflight(args, "eval")
    if len(records) % config["batch_size"]:
        raise ValueError("prompt count must be divisible by batch_size")
    if args.validate_only:
        print(json.dumps(dict(config=config, records=len(records)), indent=2))
        return
    output = new_output(config)
    runtime = GroundingRuntime(config, "eval")
    evaluate(runtime, records, output, generate_images=args.generate_images)


if __name__ == "__main__":
    main()
