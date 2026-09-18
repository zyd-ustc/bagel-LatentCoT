#!/usr/bin/env python3
"""Generate an official high-atomicity GenEval2 subset with frozen BAGEL."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--benchmark-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-atom-count", type=int, default=5)
    parser.add_argument(
        "--required-skill",
        action="append",
        choices=("object", "count", "attribute", "position", "verb"),
        default=[],
        help="Repeat to require several skills in every selected prompt.",
    )
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    return parser.parse_args()


def _load_hard_rows(args: argparse.Namespace) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(args.benchmark_data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = set(args.required_skill)
    selected = [
        row
        for row in rows
        if int(row["atom_count"]) >= int(args.min_atom_count)
        and required.issubset(set(row["skills"]))
    ]
    # Highest-compositionality cases first; file order breaks ties reproducibly.
    selected.sort(key=lambda row: -int(row["atom_count"]))
    if int(args.max_prompts) > 0:
        selected = selected[: int(args.max_prompts)]
    if not selected:
        raise ValueError("the GenEval2 filters selected no prompts")
    return selected


def _initial_noise(model, image_shape: tuple[int, int], seed: int) -> torch.Tensor:
    height, width = image_shape
    rows = (height // int(model.latent_downsample)) * (
        width // int(model.latent_downsample)
    )
    channels = int(model.latent_channel) * int(model.latent_patch_size) ** 2
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((rows, channels), generator=generator, dtype=torch.float32)


def _contexts(inferencer, prompt: str):
    full = inferencer.init_gen_context()
    visual_only = deepcopy(full)
    full = inferencer.update_context_text(prompt, full)
    return full, visual_only, deepcopy(full)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("BAGEL generation requires CUDA")
    rows = _load_hard_rows(args)

    from qwen_latent_cot.bagel.backbone import BagelBackbone
    from qwen_latent_cot.bagel.inferencer import InterleaveInferencer

    backbone = BagelBackbone(
        {
            "model_path": args.model_path,
            "disable_visual_gen": False,
            "disable_gen_expert": False,
            "num_image_tokens": 4900,
        }
    ).load()
    assert backbone.bagel is not None and backbone.vae_model is not None
    device = torch.device(args.device)
    backbone.bagel.to(device).eval().requires_grad_(False)
    backbone.vae_model.to(device).eval().requires_grad_(False)
    ids = backbone.token_ids
    inferencer = InterleaveInferencer(
        model=backbone.bagel,
        vae_model=backbone.vae_model,
        tokenizer=backbone.tokenizer,
        vae_transform=None,
        vit_transform=None,
        new_token_ids={
            "bos_token_id": int(ids.im_start),
            "eos_token_id": int(ids.im_end),
            "start_of_image": int(ids.vision_start),
            "end_of_image": int(ids.vision_end),
        },
    )

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (int(args.image_height), int(args.image_width))
    image_map: dict[str, str] = {}
    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        seed = int(args.seed) + index
        full, visual_only, text_only = _contexts(inferencer, prompt)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            image = inferencer.gen_image(
                image_shape,
                full,
                cfg_text_precontext=visual_only,
                cfg_img_precontext=text_only,
                cfg_text_scale=float(args.cfg_text_scale),
                cfg_img_scale=1.0,
                num_timesteps=int(args.num_steps),
                timestep_shift=float(args.timestep_shift),
                init_noise=_initial_noise(backbone.bagel, image_shape, seed),
            )
        image_path = (image_dir / f"{index:04d}.png").resolve()
        image.save(image_path)
        image_map[prompt] = str(image_path)
        print(f"[{index + 1}/{len(rows)}] {image_path}", flush=True)

    selected_path = output_dir / "benchmark_hard.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "image_paths.json").write_text(
        json.dumps(image_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "schema": "bagel_geneval2_hard_zero_shot_v1",
        "model_path": str(Path(args.model_path).resolve()),
        "benchmark_source": str(Path(args.benchmark_data).resolve()),
        "selected_benchmark": str(selected_path.resolve()),
        "image_paths": str((output_dir / "image_paths.json").resolve()),
        "prompt_count": len(rows),
        "min_atom_count": int(args.min_atom_count),
        "required_skills": list(args.required_skill),
        "seed": int(args.seed),
        "generation": {
            "loop_state": "off",
            "image_shape": list(image_shape),
            "num_steps": int(args.num_steps),
            "cfg_text_scale": float(args.cfg_text_scale),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
