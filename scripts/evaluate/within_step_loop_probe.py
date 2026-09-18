#!/usr/bin/env python3
"""Zero-shot probe: repeat a shared middle block inside one denoising step.

This is the BAGEL analogue of naive Looped MMDiT: layers ``[start, end)`` are
re-applied ``L`` times on the same hidden state within a single FM step, with a
damping factor ``alpha`` on the residual::

    h <- h + alpha * (body(h) - h)

``L=1`` is exact parity with the native model (the loop branch is skipped).

The SenseTime Looped MMDiT blog reports that naive looping degrades beyond the
training depth and erodes token position information. This probe measures the
same thing on frozen BAGEL, zero-shot, without any training.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from bagel_common import (
    add_native_model_args,
    load_native_bagel,
    make_noise,
    parse_csv_floats,
    pixel_mae,
    relative_l2,
    run_full_trajectory,
    shifted_schedule,
    stable_noise_seed,
)
from qwen_latent_cot.bagel import accelerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--repeat-list", default="1,2,4,8")
    parser.add_argument("--damping-list", default="1.0,0.5")
    parser.add_argument("--loop-start-layer", type=int, default=10)
    parser.add_argument("--loop-end-layer", type=int, default=18)
    parser.add_argument("--save-latents", action="store_true")
    parser.set_defaults(num_steps=20)
    return parser.parse_args()


def load_prompts(args) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(
                    str(json.loads(line)["prompt"]).strip()
                    if line.startswith("{")
                    else line
                )
    elif str(args.prompt).strip():
        prompts.append(str(args.prompt).strip())
    if not prompts:
        raise ValueError("provide --prompt or --prompt-file")
    if int(args.max_prompts) > 0:
        prompts = prompts[: int(args.max_prompts)]
    return prompts


def t2i_bundle(inferencer, prompt: str, image_shape, init_noise):
    full = inferencer.update_context_text(prompt, inferencer.init_gen_context())
    empty = inferencer.init_gen_context()
    contexts = {
        "full": full,
        "text_removed": empty,
        "image_removed": full,
        "has_visual_condition": False,
    }
    return inferencer.prepare_image_condition_bundle(
        name="t2i", contexts=contexts, image_shape=image_shape, init_noise=init_noise
    )


def write_gallery(output_dir: Path, rows: list[dict], configs: list[tuple],
                  loop_layers: tuple) -> None:
    header = "".join(f"<th>{'L=%d' % L} α={a:g}</th>" for L, a in configs)
    cells = []
    for row in rows:
        tds = "".join(
            f"<td><img src='{html.escape(src)}'><div class='cap'>rel-L2 {rl:.3f}<br>MAE {ma:.1f}</div></td>"
            for src, rl, ma in row["cells"]
        )
        cells.append(
            f"<tr><td class='prompt'>{html.escape(row['prompt'])}</td>{tds}</tr>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>Within-step loop probe</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
table{{border-collapse:collapse}} td,th{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:150px;height:150px;object-fit:contain;background:#000}}
.cap{{font-size:11px;color:#aaa;text-align:center}}
.prompt{{max-width:240px;color:#ffd479}}</style>
<h1>Within-step repeated-block probe (frozen BAGEL, zero-shot)</h1>
<p>looped layers [{loop_layers[0]}, {loop_layers[1]}); rel-L2 / pixel MAE are vs L=1 for the same prompt and noise.</p>
<table><tr><th>prompt</th>{header}</tr>{"".join(cells)}</table>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    repeats = [int(v) for v in parse_csv_floats(args.repeat_list)]
    dampings = parse_csv_floats(args.damping_list)
    configs = [(L, a) for L in repeats for a in dampings]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (int(args.image_height), int(args.image_width))

    print(f"[run] {len(prompts)} prompts x {len(configs)} configs", flush=True)
    print("[model] loading frozen native BAGEL checkpoint", flush=True)
    backbone, inferencer = load_native_bagel(args)
    model = backbone.bagel
    assert model is not None

    timesteps, dts = shifted_schedule(int(args.num_steps), float(args.timestep_shift))
    rows = []
    records = []

    for index, prompt in enumerate(prompts):
        seed = stable_noise_seed(
            int(args.seed), prompt, schema="bagel-within-step-loop-v1"
        )
        init_noise = make_noise(model, image_shape, seed).to(
            accelerator.resolve_device(args.device)
        )
        bundle = t2i_bundle(inferencer, prompt, image_shape, init_noise)
        ref_latent = None
        cells = []
        for L, alpha in configs:
            tag = f"L{L}_a{alpha:g}"
            loop_kwargs = {
                "within_step_loop_start": int(args.loop_start_layer),
                "within_step_loop_end": int(args.loop_end_layer),
                "within_step_loop_repeat": int(L),
                "within_step_loop_damping": float(alpha),
            }
            latent = run_full_trajectory(
                inferencer,
                model,
                bundle,
                init_noise,
                timesteps,
                dts,
                args,
                within_step_loop=loop_kwargs,
            )
            image = inferencer.decode_image(latent, image_shape)
            name = f"p{index:03d}_{tag}"
            image.save(output_dir / f"{name}.png")
            if L == 1:
                # Parity gate: L=1 must be bit-identical to the native path.
                native = run_full_trajectory(
                    inferencer, model, bundle, init_noise, timesteps, dts, args
                )
                if not torch.equal(native, latent):
                    raise AssertionError(
                        f"L=1 is not parity with the native path (p{index:03d}): "
                        f"max|Δ|={float((native - latent).abs().max())}"
                    )
                print(f"[parity] p{index:03d} L=1 == native OK", flush=True)
                ref_latent = latent
            if ref_latent is None:
                raise RuntimeError(
                    "repeat list must start at L=1 to form the reference"
                )
            rl = relative_l2(latent, ref_latent)
            ma = pixel_mae(image, inferencer.decode_image(ref_latent, image_shape))
            cells.append((f"{name}.png", rl, ma))
            if args.save_latents:
                torch.save(latent.cpu(), output_dir / f"{name}.pt")
            print(
                f"[p{index:03d}] {tag} rel_l2_vs_L1={rl:.4f} mae={ma:.2f}", flush=True
            )
        rows.append({"prompt": prompt, "cells": cells, "noise_seed": int(seed)})
        records.append(
            {
                "prompt": prompt,
                "noise_seed": int(seed),
                "arms": [
                    {
                        "config": f"L{L}_a{alpha:g}",
                        "relative_l2_vs_L1": c[1],
                        "pixel_mae_vs_L1": c[2],
                    }
                    for (L, alpha), c in zip(configs, cells)
                ],
            }
        )

    manifest = {
        "schema": "bagel_within_step_loop_probe_v1",
        "prompts": prompts,
        "configs": [f"L{L}_a{alpha:g}" for L, alpha in configs],
        "loop_layers": [int(args.loop_start_layer), int(args.loop_end_layer)],
        "num_steps": int(args.num_steps),
        "image_shape": list(image_shape),
        "records": records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_gallery(output_dir, rows, configs, (int(args.loop_start_layer), int(args.loop_end_layer)))
    print(f"[done] gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
