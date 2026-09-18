#!/usr/bin/env python3
"""Multi-trajectory reflection loop on frozen BAGEL (zero-shot).

Per prompt and truncation ``t_stop``:

  1. draft   : run Euler from t=1 down to t_stop, estimate x0_hat, decode preview
  2. reflect : full UND pass over the preview (ViT only) -> text reflection
  3. regen   : full Euler from the SAME initial noise, conditioned on
               [prompt] + optional [x0_hat as a raw VAE latent] + optional [reflection text]

Arms:
  baseline                native T2I, no extra condition
  draft_tXX               truncated draft (preview of what UND sees)
  regen_latent_tXX        round 2 conditioned on prompt + x0_hat latent
  regen_reflection_tXX    round 2 conditioned on prompt + reflection text
  regen_both_tXX          round 2 conditioned on prompt + x0_hat latent + reflection text

Everything is BAGEL-native: the latent condition enters through the VAE/GEN K/V
path (no pixel decode + re-encode) and the reflection enters as native text K/V.

Prompts come from ``--prompt`` (single) or ``--prompt-file`` (one prompt per
line, or GenEval2-style JSONL with a ``prompt`` field). Output layout::

  <out>/p000/baseline.png, draft_t0.9.png, regen_*_t0.9.png, reflection_t0.9.txt
  <out>/index.html, run_manifest.json
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
    run_truncated_draft,
    shifted_schedule,
    stable_noise_seed,
)
from qwen_latent_cot.bagel import accelerator


REFLECTION_SYSTEM_PROMPT = """You are the understanding side of a text-to-image model.
You inspect a partially denoised draft image and audit it against the target prompt.
Report concrete mismatches: object count, position, colour/shape attributes, and details.
Then state a short, positive regeneration instruction that fixes the mismatches while
preserving everything already correct. Do not invent requirements.
Keep it under six lines."""

REFLECTION_USER_TEMPLATE = """Target prompt:
{prompt}

Inspect the draft image and report the mismatches, then give the regeneration instruction."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument(
        "--prompt", default="", help="Single prompt (or use --prompt-file)."
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="One prompt per line, or GenEval2-style JSONL with a 'prompt' field.",
    )
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument(
        "--regen-noise",
        default="fresh",
        choices=["fresh", "same"],
        help=(
            "fresh: round 2 resamples noise (true multi-trajectory) and runs a "
            "matched no-condition control; same: round 2 reuses round-1 noise "
            "(isolates the condition causally, but the prompt-dominated flow "
            "barely moves)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument(
        "--truncations",
        default="0.9",
        help="CSV of draft truncation timesteps.",
    )
    parser.add_argument(
        "--regen-variants",
        default="latent,reflection,both",
        help="CSV subset of latent,reflection,both.",
    )
    parser.add_argument("--reflection-max-tokens", type=int, default=1000)
    parser.add_argument("--reflection-sample", action="store_true")
    parser.add_argument("--reflection-temperature", type=float, default=0.3)
    parser.add_argument("--reflection-system-prompt", default="",
        help="Official understanding uses no system prompt; empty = in-distribution.")
    parser.add_argument("--reflection-user-template", default=REFLECTION_USER_TEMPLATE)
    parser.add_argument(
        "--regen-cfg-img-scale",
        type=float,
        default=2.0,
        help="Official editing uses 2.0; 1.0 disables the image-CFG branch entirely.",
    )
    parser.add_argument("--regen-cfg-interval-min", type=float, default=0.0)
    parser.add_argument("--regen-cfg-renorm-type", default="text_channel")
    return parser.parse_args()


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in str(value).split(",") if part.strip()]
    invalid = sorted(set(variants) - {"latent", "reflection", "both"})
    if invalid:
        raise ValueError(f"invalid --regen-variants: {invalid}")
    if not variants:
        raise ValueError("--regen-variants must not be empty")
    return variants


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        for line in (
            Path(args.prompt_file).expanduser().read_text(encoding="utf-8").splitlines()
        ):
            line = line.strip()
            if not line:
                continue
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


def build_condition_bundle(
    inferencer,
    *,
    name: str,
    prompt: str,
    image_shape,
    init_noise,
    latent=None,
    reflection: str | None = None,
):
    """Native BAGEL triple-CFG bundle for [prompt] + [latent] + [reflection]."""

    def make(*, with_prompt: bool, with_latent: bool, with_reflection: bool):
        ctx = inferencer.init_gen_context()
        if with_prompt:
            ctx = inferencer.update_context_text(prompt, ctx)
        if with_latent and latent is not None:
            ctx = inferencer.update_context_vae_latent(
                latent, image_shape, ctx, timestep=0.0
            )
        if with_reflection and reflection:
            ctx = inferencer.update_context_text(reflection, ctx)
        return ctx

    contexts = {
        "full": make(with_prompt=True, with_latent=True, with_reflection=True),
        "text_removed": make(
            with_prompt=False, with_latent=True, with_reflection=False
        ),
        "image_removed": make(
            with_prompt=True, with_latent=False, with_reflection=True
        ),
        "has_visual_condition": latent is not None,
    }
    return inferencer.prepare_image_condition_bundle(
        name=name,
        contexts=contexts,
        image_shape=image_shape,
        init_noise=init_noise,
    )


def write_gallery(output_dir: Path, rows: list[dict]) -> None:
    cells = []
    for row in rows:
        cell = ["<figure><div class='imgs'>"]
        for src, label in row["panels"]:
            cell.append(
                f"<div><img src='{html.escape(src)}'><div class='cap'>{html.escape(label)}</div></div>"
            )
        cell.append("</div>")
        if row.get("reflection"):
            cell.append(f"<pre class='reflect'>{html.escape(row['reflection'])}</pre>")
        cell.append(
            f"<figcaption><b>{html.escape(row['arm'])}</b></figcaption></figure>"
        )
        cells.append("".join(cell))
    page = f"""<!doctype html><meta charset='utf-8'>
<title>Multi-trajectory reflection loop</title>
<style>body{{font:14px system-ui;background:#111;color:#eee;margin:24px}}
.grid{{display:flex;gap:16px;flex-wrap:wrap}}
figure{{margin:0;background:#1a1a1a;padding:10px;border-radius:8px;max-width:760px}}
.imgs{{display:flex;gap:8px;flex-wrap:wrap}}
img{{width:200px;height:200px;object-fit:contain;background:#000}}
.cap{{text-align:center;font-size:12px;color:#aaa}}
.reflect{{white-space:pre-wrap;max-width:740px;background:#000;padding:8px;border-radius:4px;font-size:12px}}</style>
<h1>Multi-trajectory reflection loop (frozen BAGEL, zero-shot)</h1>
<div class='grid'>{"".join(cells)}</div>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def run_prompt(
    inferencer,
    model,
    args,
    *,
    prompt: str,
    prompt_dir: Path,
    timesteps,
    dts,
    truncations: list[float],
    variants: list[str],
) -> dict:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (int(args.image_height), int(args.image_width))
    device = accelerator.resolve_device(args.device)
    same_noise = str(args.regen_noise) == "same"
    regen_cfg = {
        "cfg_img_scale": float(args.regen_cfg_img_scale),
        "cfg_interval": (float(args.regen_cfg_interval_min), float(args.cfg_interval_max)),
        "cfg_renorm_type": str(args.regen_cfg_renorm_type),
    }

    base_seed = stable_noise_seed(int(args.seed), prompt, schema="bagel-multitraj-reflection-v1")
    base_noise = make_noise(model, image_shape, base_seed).to(device)
    regen_seed = stable_noise_seed(int(args.seed), prompt, schema="bagel-multitraj-reflection-regen-v1")
    regen_noise = make_noise(model, image_shape, regen_seed).to(device)

    def save(name, image):
        image.save(prompt_dir / f"{name}.png")
        return f"{prompt_dir.name}/{name}.png"

    baseline_bundle = build_condition_bundle(
        inferencer, name="baseline", prompt=prompt, image_shape=image_shape,
        init_noise=base_noise,
    )
    baseline_latent = run_full_trajectory(
        inferencer, model, baseline_bundle, base_noise, timesteps, dts, args
    )
    baseline_image = inferencer.decode_image(baseline_latent, image_shape)
    baseline_rel = save("baseline", baseline_image)
    print(f"[baseline] {prompt_dir.name} base_seed={base_seed}", flush=True)

    if same_noise:
        control_latent, control_image, control_rel = baseline_latent, baseline_image, baseline_rel
    else:
        control_bundle = build_condition_bundle(
            inferencer, name="control", prompt=prompt, image_shape=image_shape,
            init_noise=regen_noise,
        )
        control_latent = run_full_trajectory(
            inferencer, model, control_bundle, regen_noise, timesteps, dts, args,
            cfg=regen_cfg,
        )
        control_image = inferencer.decode_image(control_latent, image_shape)
        control_rel = save("control", control_image)
        print(f"[control] {prompt_dir.name} regen_seed={regen_seed}", flush=True)

    round2_noise = base_noise if same_noise else regen_noise
    arms: list[dict] = [
        {
            "arm": "control",
            "regen_noise": "same" if same_noise else "fresh",
            "relative_l2_vs_baseline": relative_l2(control_latent, baseline_latent),
            "pixel_mae_vs_baseline": pixel_mae(control_image, baseline_image),
        }
    ]
    panels = [(baseline_rel, "baseline (noise A)")]
    if not same_noise:
        panels.append((control_rel, "control (noise B, no condition)"))
    reflection_text = ""

    with torch.no_grad():
        for t_stop in truncations:
            tag = f"t{float(t_stop):g}"
            print(f"[draft] {prompt_dir.name} {tag}", flush=True)
            x0_hat, actual_t = run_truncated_draft(
                inferencer, model, baseline_bundle, base_noise, timesteps, dts, args,
                float(t_stop),
            )
            preview = inferencer.decode_image(x0_hat, image_shape)
            preview_rel = save(f"draft_{tag}", preview)

            user_text = str(args.reflection_user_template).format(prompt=prompt)
            reflection_text = inferencer.generate_image_reflection(
                preview,
                user_text=user_text,
                system_prompt=str(args.reflection_system_prompt) or None,
                max_length=int(args.reflection_max_tokens),
                do_sample=bool(args.reflection_sample),
                temperature=float(args.reflection_temperature),
            )
            (prompt_dir / f"reflection_{tag}.txt").write_text(
                reflection_text + "\n", encoding="utf-8"
            )
            print(f"[reflect] {prompt_dir.name} {tag}: {reflection_text[:140]!r}", flush=True)

            panels.append((preview_rel, f"draft @{actual_t:.3f}"))
            arms.append(
                {
                    "arm": f"draft_{tag}",
                    "actual_t": actual_t,
                    "reflection": reflection_text,
                    "relative_l2_vs_baseline": relative_l2(x0_hat, baseline_latent),
                }
            )

            for variant in variants:
                use_latent = variant in ("latent", "both")
                use_reflection = variant in ("reflection", "both")
                name = f"regen_{variant}_{tag}"
                bundle = build_condition_bundle(
                    inferencer, name=name, prompt=prompt, image_shape=image_shape,
                    init_noise=round2_noise,
                    latent=x0_hat.detach() if use_latent else None,
                    reflection=reflection_text if use_reflection else None,
                )
                final_latent = run_full_trajectory(
                    inferencer, model, bundle, round2_noise, timesteps, dts, args,
                    cfg=regen_cfg,
                )
                image = inferencer.decode_image(final_latent, image_shape)
                image_rel = save(name, image)
                panels.append((image_rel, variant))
                arms.append(
                    {
                        "arm": name,
                        "actual_t": actual_t,
                        "use_latent": use_latent,
                        "use_reflection": use_reflection,
                        "relative_l2_vs_baseline": relative_l2(final_latent, baseline_latent),
                        "pixel_mae_vs_baseline": pixel_mae(image, baseline_image),
                        "relative_l2_vs_control": relative_l2(final_latent, control_latent),
                        "pixel_mae_vs_control": pixel_mae(image, control_image),
                    }
                )
                print(
                    f"[regen] {prompt_dir.name} {name} "
                    f"mae_vs_control={arms[-1]['pixel_mae_vs_control']:.2f} "
                    f"mae_vs_baseline={arms[-1]['pixel_mae_vs_baseline']:.2f}",
                    flush=True,
                )

    return {
        "prompt": prompt,
        "baseline_noise_seed": int(base_seed),
        "regen_noise_seed": int(regen_seed),
        "regen_noise": "same" if same_noise else "fresh",
        "baseline_image": baseline_rel,
        "control_image": control_rel,
        "reflection": reflection_text,
        "arms": arms,
        "_panels": panels,
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    variants = parse_variants(args.regen_variants)
    truncations = parse_csv_floats(args.truncations)
    if not truncations:
        raise ValueError("--truncations must not be empty")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (int(args.image_height), int(args.image_width))
    if image_shape[0] % 16 or image_shape[1] % 16:
        raise ValueError("image dimensions must be divisible by 16")

    print(f"[run] {len(prompts)} prompts, truncations={truncations}", flush=True)
    print("[model] loading frozen native BAGEL checkpoint (no adapter)", flush=True)
    backbone, inferencer = load_native_bagel(args)
    model = backbone.bagel
    assert model is not None

    timesteps, dts = shifted_schedule(int(args.num_steps), float(args.timestep_shift))

    records = []
    gallery_rows = []
    for index, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"p{index:03d}"
        record = run_prompt(
            inferencer,
            model,
            args,
            prompt=prompt,
            prompt_dir=prompt_dir,
            timesteps=timesteps,
            dts=dts,
            truncations=truncations,
            variants=variants,
        )
        panels = record.pop("_panels")
        records.append(record)
        gallery_rows.append(
            {
                "arm": f"p{index:03d}: {prompt[:70]}",
                "panels": panels,
                "reflection": record["reflection"],
            }
        )

    manifest = {
        "schema": "bagel_multitraj_reflection_loop_v2",
        "prompt_count": len(prompts),
        "prompts": prompts,
        "seed": int(args.seed),
        "image_shape": list(image_shape),
        "num_steps": int(args.num_steps),
        "timestep_shift": float(args.timestep_shift),
        "cfg": {
            "text_scale": float(args.cfg_text_scale),
            "img_scale": float(args.cfg_img_scale),
            "interval": [float(args.cfg_interval_min), float(args.cfg_interval_max)],
        },
        "truncations": [float(t) for t in truncations],
        "regen_variants": variants,
        "reflection_system_prompt": str(args.reflection_system_prompt),
        "reflection_user_template": str(args.reflection_user_template),
        "reflection_max_tokens": int(args.reflection_max_tokens),
        "records": records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_gallery(output_dir, gallery_rows)
    print(f"[done] gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
