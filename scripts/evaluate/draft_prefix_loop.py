#!/usr/bin/env python3
"""Zero-shot draft-prefix reflection loop.

Every visible I_r is decode(x0_hat) at t_A, including r>=1.
r=0: T2I prefix to t_A.
r>0: official UND, then official Editing CFG/image-cond, fresh ε, prefix to t_A.

--edit-stop full restores the v5 50-step probe (UND then sees a finished image).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Optional

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
from qwen_latent_cot.bagel.modeling._bagel_utils import pil_img2rgb


# Official Understanding cell in inference.ipynb: a natural question, not a form.
REFLECT_QUERY = """This picture is supposed to show: {prompt}

What is wrong with the object counts or positions compared to that target? Answer in plain sentences, then say how to fix it."""

TEMPLATE_ECHO = (
    "imperative sentence",
    "one specific error",
    "fixes that error",
    "one concrete error",
    "<no changes",
    "no changes needed",
    "no fix needed",
)
NO_OP_INSTRUCTION = (
    "no change",
    "no changes",
    "no fix",
    "no correction",
    "none needed",
    "nothing to",
)
DONE_PHRASES = (
    "already correct",
    "no issues",
    "are correct compared",
    "counts and positions are correct",
    "object counts and positions are correct",
    "nothing to fix",
    "no further",
)
GEN_TEXT_MODES = ("a_only", "p_plus_a", "p_only")
EDIT_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)
UND_HYPER = dict(
    understanding_output=True,
    max_think_token_n=1000,
    do_sample=False,
)
EDIT_CFG = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": (0.0, 1.0),
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
}
EDIT_STOPS = ("prefix", "full")


def says_already_correct(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in DONE_PHRASES)


def is_usable_reflection(text: str) -> bool:
    """Official UND is free-form. Drop template copies and no-op answers."""

    text = (text or "").strip()
    if len(text) < 8:
        return False
    lowered = text.lower()
    if any(token in lowered for token in TEMPLATE_ECHO):
        return False
    if any(token in lowered for token in NO_OP_INSTRUCTION):
        return False
    if says_already_correct(text):
        return False
    return True


def parse_edit_stop(value: str) -> str:
    stop = str(value).strip()
    if stop not in EDIT_STOPS:
        raise ValueError(f"invalid --edit-stop: {value!r} (use prefix or full)")
    return stop


def parse_gen_text_modes(value: str) -> list[str]:
    modes = [part.strip() for part in str(value).split(",") if part.strip()]
    invalid = [mode for mode in modes if mode not in GEN_TEXT_MODES]
    if invalid:
        raise ValueError(f"invalid --gen-text modes: {invalid}")
    if not modes:
        raise ValueError("--gen-text must not be empty")
    return modes


def compose_gen_text(mode: str, prompt: str, instruction: Optional[str]) -> Optional[str]:
    """r>0 GEN text. None means skip this prefix (no usable instruction)."""

    if mode == "p_only":
        return str(prompt)
    instruction = (instruction or "").strip()
    if not instruction:
        return None
    if mode == "a_only":
        return instruction
    if mode == "p_plus_a":
        return f"{prompt}\n\nEdit instruction: {instruction}"
    raise ValueError(f"unknown gen-text mode: {mode}")


def build_gen_bundle(
    inferencer,
    *,
    name: str,
    text: Optional[str],
    image_shape,
    init_noise,
    draft_image=None,
):
    """r=0: text-only T2I. r>0: official VAE+ViT encode of the decoded draft + text."""

    encoded = None
    if draft_image is not None:
        encoded = inferencer.vae_transform.resize_transform(pil_img2rgb(draft_image))

    def make(*, with_text: bool, with_image: bool):
        ctx = inferencer.init_gen_context()
        if with_image and encoded is not None:
            ctx = inferencer.update_context_image(
                encoded, ctx, vae=True, vit=True
            )
        if with_text and text:
            ctx = inferencer.update_context_text(text, ctx)
        return ctx

    has_image = encoded is not None
    return inferencer.prepare_image_condition_bundle(
        name=name,
        contexts={
            "full": make(with_text=True, with_image=has_image),
            "text_removed": make(with_text=False, with_image=has_image),
            "image_removed": make(with_text=True, with_image=False),
            "has_visual_condition": has_image,
        },
        image_shape=image_shape,
        init_noise=init_noise,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_model_args(parser)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--truncations",
        default="0.8",
        help="t_A. v2 picked 0.8: UND started emitting INSTRUCTION, draft still prefix.",
    )
    parser.add_argument(
        "--gen-text",
        default="a_only",
        help="CSV of a_only (GEN never sees P) and p_plus_a (GEN still sees P).",
    )
    parser.add_argument(
        "--edit-stop",
        default="prefix",
        help="prefix: r>0 also stops at t_A and decodes x0_hat. full: v5 50-step edit.",
    )
    parser.add_argument("--reflect-query", default=REFLECT_QUERY)
    parser.add_argument("--reflection-max-tokens", type=int, default=1000)
    parser.add_argument("--reflection-sample", action="store_true")
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


def write_gallery(
    output_dir: Path,
    rows: list[dict],
    *,
    truncations,
    rounds,
    gen_text_modes: list[str],
    edit_stop: str,
) -> None:
    blocks = []
    for row in rows:
        tds = "".join(
            f"<td><img src='{html.escape(src)}'><div class='cap'>{html.escape(cap)}</div></td>"
            for src, cap in row["panels"]
        )
        blocks.append(
            "<h2 class='prompt'>"
            + html.escape(row["prompt"])
            + "</h2><table><tr>"
            + tds
            + "</tr></table>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>Draft-prefix reflection loop</title>
<style>body{{font:13px system-ui;background:#111;color:#eee;margin:20px}}
table{{border-collapse:collapse;margin-bottom:24px}} td{{padding:6px;border:1px solid #333;vertical-align:top}}
img{{width:140px;height:140px;object-fit:contain;background:#000}}
.cap{{font-size:11px;color:#aaa;text-align:center;max-width:140px}}
.prompt{{color:#ffd479;font-size:16px}}</style>
<h1>Draft-prefix loop — edit_stop={html.escape(edit_stop)}</h1>
<p>modes={html.escape(",".join(gen_text_modes))} t={html.escape(",".join(str(t) for t in truncations))} rounds={int(rounds)} — prefix means every I_r is x0_hat @ t_A</p>
{"".join(blocks)}"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    truncations = parse_csv_floats(args.truncations)
    gen_text_modes = parse_gen_text_modes(args.gen_text)
    edit_stop = parse_edit_stop(args.edit_stop)
    if not truncations:
        raise ValueError("--truncations must not be empty")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (int(args.image_height), int(args.image_width))

    print(
        f"[run] draft-prefix  prompts={len(prompts)} t={truncations} "
        f"rounds={int(args.rounds)} gen_text={gen_text_modes} "
        f"edit_stop={edit_stop}  GEN sees VAE(draft)+ViT, fresh ε",
        flush=True,
    )
    print("[model] loading frozen native BAGEL", flush=True)
    backbone, inferencer = load_native_bagel(args)
    model = backbone.bagel
    assert model is not None
    device = accelerator.resolve_device(args.device)
    timesteps, dts = shifted_schedule(int(args.num_steps), float(args.timestep_shift))

    rows, records = [], []
    for index, prompt in enumerate(prompts):
        pdir = output_dir / f"p{index:03d}"
        pdir.mkdir(parents=True, exist_ok=True)
        seed = stable_noise_seed(int(args.seed), prompt, schema="bagel-draft-prefix-v4")
        init_noise = make_noise(model, image_shape, seed).to(device)

        baseline_bundle = build_gen_bundle(
            inferencer, name="baseline", text=prompt, image_shape=image_shape,
            init_noise=init_noise,
        )
        baseline_latent = run_full_trajectory(
            inferencer, model, baseline_bundle, init_noise, timesteps, dts, args
        )
        baseline = inferencer.decode_image(baseline_latent, image_shape)
        baseline.save(pdir / "baseline.png")
        panels = [(f"p{index:03d}/baseline.png", "baseline T2I(P,ε)")]
        print(f"[p{index:03d}] baseline ok", flush=True)

        t_records = []
        for t_stop in truncations:
            tag = f"t{float(t_stop):g}"
            bundle0 = build_gen_bundle(
                inferencer, name=f"{tag}_r0", text=prompt,
                image_shape=image_shape, init_noise=init_noise,
            )
            x0_0, actual_t0 = run_truncated_draft(
                inferencer, model, bundle0, init_noise, timesteps, dts, args,
                float(t_stop),
            )
            draft0 = inferencer.decode_image(x0_0, image_shape)
            draft0.save(pdir / f"draft_{tag}_r0.png")
            panels.append((f"p{index:03d}/draft_{tag}_r0.png", f"{tag} r0 (P, no draft cond)"))
            print(f"[p{index:03d}] {tag} r0 ok t={actual_t0:.3f}", flush=True)

            mode_records = []
            for mode in gen_text_modes:
                last_draft = draft0
                drafts = [
                    {
                        "round": 0,
                        "actual_t": actual_t0,
                        "gen_text": prompt,
                        "reflection": None,
                        "skipped": False,
                        "official_edit": False,
                        "pixel_mae_vs_draft0": 0.0,
                    }
                ]
                for r in range(1, int(args.rounds) + 1):
                    und = inferencer(
                        image=last_draft,
                        text=str(args.reflect_query).format(prompt=prompt),
                        understanding_output=True,
                        max_think_token_n=int(args.reflection_max_tokens),
                        do_sample=bool(args.reflection_sample),
                    )
                    reflection = (und.get("text") or "").strip()
                    (pdir / f"reflection_{tag}_{mode}_r{r}.txt").write_text(
                        reflection + "\n", encoding="utf-8"
                    )
                    text = compose_gen_text(mode, prompt, reflection)
                    done = says_already_correct(reflection)
                    skipped = (
                        text is None
                        or not is_usable_reflection(reflection)
                        or done
                    )
                    if skipped:
                        draft = last_draft
                        actual_t = None
                        print(
                            f"[p{index:03d}] {tag} {mode} r{r} "
                            f"{'STOP already-correct' if done else 'SKIP bad UND'}\n"
                            f"    und: {reflection[:180]!r}",
                            flush=True,
                        )
                    elif edit_stop == "full":
                        edited = inferencer(image=last_draft, text=text, **EDIT_HYPER)
                        draft = edited["image"]
                        actual_t = 0.0
                        print(
                            f"[p{index:03d}] {tag} {mode} r{r} official edit 50 steps\n"
                            f"    und: {reflection[:180]!r}",
                            flush=True,
                        )
                    else:
                        edit_seed = stable_noise_seed(
                            int(args.seed),
                            f"{prompt}|{mode}|r{r}",
                            schema="bagel-draft-edit-v7",
                        )
                        edit_noise = make_noise(model, image_shape, edit_seed).to(device)
                        bundle = build_gen_bundle(
                            inferencer,
                            name=f"{tag}_{mode}_r{r}",
                            text=text,
                            image_shape=image_shape,
                            init_noise=edit_noise,
                            draft_image=last_draft,
                        )
                        x0_hat, actual_t = run_truncated_draft(
                            inferencer,
                            model,
                            bundle,
                            edit_noise,
                            timesteps,
                            dts,
                            args,
                            float(t_stop),
                            cfg=EDIT_CFG,
                        )
                        draft = inferencer.decode_image(x0_hat, image_shape)
                        print(
                            f"[p{index:03d}] {tag} {mode} r{r} "
                            f"prefix edit t={actual_t:.3f} fresh ε\n"
                            f"    und: {reflection[:180]!r}",
                            flush=True,
                        )
                    name = f"draft_{tag}_{mode}_r{r}"
                    draft.save(pdir / f"{name}.png")
                    panels.append(
                        (
                            f"p{index:03d}/{name}.png",
                            f"{tag} {mode} r{r}: {(reflection or 'NO UND')[:60]}",
                        )
                    )
                    drafts.append(
                        {
                            "round": r,
                            "gen_text": None if skipped else text,
                            "reflection": reflection,
                            "skipped": skipped,
                            "stopped": done,
                            "official_edit": not skipped,
                            "edit_stop": edit_stop,
                            "actual_t": actual_t,
                            "pixel_mae_vs_draft0": pixel_mae(draft, draft0),
                        }
                    )
                    last_draft = draft
                    if done:
                        for rest in range(r + 1, int(args.rounds) + 1):
                            copy_name = f"draft_{tag}_{mode}_r{rest}"
                            draft.save(pdir / f"{copy_name}.png")
                            panels.append(
                                (
                                    f"p{index:03d}/{copy_name}.png",
                                    f"{tag} {mode} r{rest}: STOP",
                                )
                            )
                            drafts.append(
                                {
                                    "round": rest,
                                    "gen_text": None,
                                    "reflection": None,
                                    "skipped": True,
                                    "stopped": True,
                                    "official_edit": False,
                                    "pixel_mae_vs_draft0": pixel_mae(draft, draft0),
                                }
                            )
                        break

                final = last_draft
                final.save(pdir / f"final_{tag}_{mode}.png")
                panels.append(
                    (f"p{index:03d}/final_{tag}_{mode}.png", f"final {tag} {mode}")
                )
                mode_records.append(
                    {
                        "mode": mode,
                        "final_mae_vs_baseline": pixel_mae(final, baseline),
                        "drafts": drafts,
                    }
                )
                print(
                    f"[p{index:03d}] {tag} {mode} final mae_vs_baseline="
                    f"{mode_records[-1]['final_mae_vs_baseline']:.2f}",
                    flush=True,
                )

            t_records.append(
                {"t_stop": float(t_stop), "actual_t0": actual_t0, "modes": mode_records}
            )

        rows.append({"prompt": prompt, "panels": panels})
        records.append({"prompt": prompt, "noise_seed": int(seed), "truncations": t_records})

    manifest = {
        "schema": "bagel_draft_prefix_loop_v7",
        "gen_text_modes": gen_text_modes,
        "truncations": [float(t) for t in truncations],
        "rounds": int(args.rounds),
        "edit_stop": edit_stop,
        "und": "official_understanding_output",
        "edit": (
            "official_inferencer_call_full_50"
            if edit_stop == "full"
            else "official_edit_cfg_prefix_x0hat_at_t_A"
        ),
        "prompts": prompts,
        "image_shape": list(image_shape),
        "t2i_cfg": {
            "cfg_text_scale": float(args.cfg_text_scale),
            "cfg_img_scale": float(args.cfg_img_scale),
            "cfg_interval": [float(args.cfg_interval_min), float(args.cfg_interval_max)],
            "cfg_renorm_type": str(args.cfg_renorm_type),
        },
        "edit_cfg": {
            "cfg_img_scale": EDIT_CFG["cfg_img_scale"],
            "cfg_interval": list(EDIT_CFG["cfg_interval"]),
            "cfg_renorm_type": EDIT_CFG["cfg_renorm_type"],
        },
        "records": records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_gallery(
        output_dir, rows, truncations=truncations, rounds=int(args.rounds),
        gen_text_modes=gen_text_modes, edit_stop=edit_stop,
    )
    print(f"[done] gallery={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
